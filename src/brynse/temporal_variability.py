"""Validated multi-horizon temporal-variability evidence extraction."""

from __future__ import annotations

import importlib
import importlib.util
import math
import statistics
from dataclasses import dataclass
from typing import Any

from brynse.models import DecodedPcm

_VALIDATED_SAMPLE_RATE = 16000
_SCALES_SECONDS = (8, 16, 24, 40)


class TemporalVariabilityAnalysisError(RuntimeError):
    """Raised when validated temporal-variability evidence cannot be produced."""


@dataclass(frozen=True)
class TemporalVariabilityEvidence:
    """Descriptive B7.3e multi-horizon temporal-variability evidence."""

    pre_profile_pairwise: tuple[float, ...]
    post_profile_pairwise: tuple[float, ...]
    variability_ratio_8s: float
    variability_ratio_16s: float
    variability_ratio_24s: float
    variability_ratio_40s: float

    def __post_init__(self) -> None:
        if len(self.pre_profile_pairwise) != 5:
            raise ValueError("pre_profile_pairwise must contain five bins")
        if len(self.post_profile_pairwise) != 5:
            raise ValueError("post_profile_pairwise must contain five bins")

        values = (
            *self.pre_profile_pairwise,
            *self.post_profile_pairwise,
            self.variability_ratio_8s,
            self.variability_ratio_16s,
            self.variability_ratio_24s,
            self.variability_ratio_40s,
        )
        if any(value < 0 for value in values):
            raise ValueError("temporal variability values must be non-negative")


class TemporalVariabilityAnalyzer:
    """Port the B7.3b/e validated TVC extraction into read-only product code."""

    def __init__(
        self,
        *,
        side_seconds: float = 40.0,
        guard_seconds: float = 2.0,
        block_seconds: float = 4.0,
        profile_bin_seconds: float = 8.0,
        required_sample_rate: int = _VALIDATED_SAMPLE_RATE,
    ) -> None:
        if side_seconds <= 0:
            raise ValueError("side_seconds must be greater than zero")
        if guard_seconds < 0:
            raise ValueError("guard_seconds must be non-negative")
        if block_seconds <= 0:
            raise ValueError("block_seconds must be greater than zero")
        if profile_bin_seconds <= 0:
            raise ValueError("profile_bin_seconds must be greater than zero")
        if required_sample_rate <= 0:
            raise ValueError("required_sample_rate must be greater than zero")

        blocks_per_side = side_seconds / block_seconds
        blocks_per_bin = profile_bin_seconds / block_seconds
        bins_per_side = side_seconds / profile_bin_seconds

        if not math.isclose(blocks_per_side, round(blocks_per_side)):
            raise ValueError("side_seconds must be an exact multiple of block_seconds")
        if not math.isclose(blocks_per_bin, round(blocks_per_bin)):
            raise ValueError("profile_bin_seconds must be an exact multiple of block_seconds")
        if not math.isclose(bins_per_side, round(bins_per_side)):
            raise ValueError("side_seconds must be an exact multiple of profile_bin_seconds")
        if round(blocks_per_bin) < 2:
            raise ValueError("profile bins need at least two blocks")
        if round(bins_per_side) != 5:
            raise ValueError("validated TVC extraction requires exactly five profile bins")

        self._side_seconds = side_seconds
        self._guard_seconds = guard_seconds
        self._block_seconds = block_seconds
        self._profile_bin_seconds = profile_bin_seconds
        self._required_sample_rate = required_sample_rate

    @property
    def available(self) -> bool:
        """Return whether the optional NumPy-backed shadow analyzer can run."""
        return importlib.util.find_spec("numpy") is not None

    @property
    def required_preroll_seconds(self) -> float:
        return self._side_seconds + self._guard_seconds

    @property
    def required_postroll_seconds(self) -> float:
        return self._side_seconds + self._guard_seconds

    @property
    def required_sample_rate(self) -> int:
        return self._required_sample_rate

    def analyze(
        self,
        *,
        pcm: DecodedPcm,
        boundary_time_seconds: float,
        absolute_start_time_seconds: float = 0.0,
    ) -> TemporalVariabilityEvidence | None:
        """Return exact B7.3e evidence, or ``None`` when context is incomplete."""
        if boundary_time_seconds < 0:
            raise ValueError("boundary_time_seconds must be non-negative")
        if absolute_start_time_seconds < 0:
            raise ValueError("absolute_start_time_seconds must be non-negative")
        if pcm.sample_rate != self._required_sample_rate:
            raise ValueError(
                f"temporal variability PCM sample rate must be {self._required_sample_rate} Hz"
            )
        if pcm.channels != 1 or pcm.sample_width_bytes != 2:
            raise ValueError("temporal variability requires mono 16-bit PCM")

        try:
            np = importlib.import_module("numpy")
        except ModuleNotFoundError as exc:
            raise TemporalVariabilityAnalysisError(
                "NumPy is required for temporal variability shadow analysis"
            ) from exc

        relative_boundary = boundary_time_seconds - absolute_start_time_seconds
        if relative_boundary < 0:
            raise ValueError("boundary_time_seconds precedes absolute_start_time_seconds")

        duration_seconds = pcm.sample_count / pcm.sample_rate
        before_start = relative_boundary - self._guard_seconds - self._side_seconds
        after_start = relative_boundary + self._guard_seconds
        after_end = after_start + self._side_seconds
        if before_start < 0 or after_end > duration_seconds:
            return None

        samples = np.frombuffer(pcm.data, dtype="<i2").astype(np.float64) / 32768.0
        block_samples = round(self._block_seconds * pcm.sample_rate)
        blocks_per_side = round(self._side_seconds / self._block_seconds)
        blocks_per_bin = round(self._profile_bin_seconds / self._block_seconds)

        before_sample = round(before_start * pcm.sample_rate)
        after_sample = round(after_start * pcm.sample_rate)

        pre_blocks = tuple(
            samples[
                before_sample + index * block_samples : before_sample + (index + 1) * block_samples
            ]
            for index in range(blocks_per_side)
        )
        post_blocks = tuple(
            samples[
                after_sample + index * block_samples : after_sample + (index + 1) * block_samples
            ]
            for index in range(blocks_per_side)
        )
        if any(block.size != block_samples for block in (*pre_blocks, *post_blocks)):
            return None

        raw = np.vstack(
            [self._block_features(block, pcm.sample_rate, np) for block in pre_blocks]
            + [self._block_features(block, pcm.sample_rate, np) for block in post_blocks]
        )
        features = self._robust_normalize(raw, np)
        pre_features = features[:blocks_per_side]
        post_features = features[blocks_per_side:]

        pre_profile = tuple(
            self._segment_variability(
                pre_features[index : index + blocks_per_bin],
                np,
            )
            for index in range(0, blocks_per_side, blocks_per_bin)
        )
        post_profile = tuple(
            self._segment_variability(
                post_features[index : index + blocks_per_bin],
                np,
            )
            for index in range(0, blocks_per_side, blocks_per_bin)
        )

        ratios = {
            seconds: self._contraction_ratio(
                pre_profile,
                post_profile,
                seconds // round(self._profile_bin_seconds),
            )
            for seconds in _SCALES_SECONDS
        }

        return TemporalVariabilityEvidence(
            pre_profile_pairwise=pre_profile,
            post_profile_pairwise=post_profile,
            variability_ratio_8s=ratios[8],
            variability_ratio_16s=ratios[16],
            variability_ratio_24s=ratios[24],
            variability_ratio_40s=ratios[40],
        )

    @staticmethod
    def _frames(signal: Any, frame_len: int, hop: int, np: Any) -> Any:
        if signal.size < frame_len:
            raise TemporalVariabilityAnalysisError("insufficient audio for feature frame")
        count = 1 + (signal.size - frame_len) // hop
        return np.lib.stride_tricks.as_strided(
            signal,
            shape=(count, frame_len),
            strides=(signal.strides[0] * hop, signal.strides[0]),
            writeable=False,
        ).copy()

    @classmethod
    def _spectrum(cls, signal: Any, rate: int, np: Any) -> tuple[Any, Any]:
        n_fft = 512
        frame_len = round(rate * 0.025)
        hop = round(rate * 0.010)
        frames = cls._frames(signal, frame_len, hop, np)
        frames *= np.hanning(frame_len)
        spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
        return np.abs(spectrum) ** 2, np.fft.rfftfreq(n_fft, d=1.0 / rate)

    @staticmethod
    def _hz_to_mel(value: Any, np: Any) -> Any:
        return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)

    @staticmethod
    def _mel_to_hz(value: Any, np: Any) -> Any:
        return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)

    @classmethod
    def _mel_bank(cls, rate: int, np: Any, n_mels: int = 40) -> Any:
        n_fft = 512
        frequencies = np.fft.rfftfreq(n_fft, d=1.0 / rate)
        points = cls._mel_to_hz(
            np.linspace(
                cls._hz_to_mel(40.0, np),
                cls._hz_to_mel(min(rate / 2.0, 7600.0), np),
                n_mels + 2,
            ),
            np,
        )
        bank = np.zeros((n_mels, frequencies.size))
        for index in range(n_mels):
            left, center, right = points[index : index + 3]
            rising = (frequencies >= left) & (frequencies <= center)
            falling = (frequencies >= center) & (frequencies <= right)
            if center > left:
                bank[index, rising] = (frequencies[rising] - left) / (center - left)
            if right > center:
                bank[index, falling] = (right - frequencies[falling]) / (right - center)
        return bank

    @staticmethod
    def _dct(np: Any, n_mfcc: int = 13, n_mels: int = 40) -> Any:
        k = np.arange(n_mfcc)[:, None]
        n = np.arange(n_mels)[None, :]
        matrix = np.cos(np.pi / n_mels * (n + 0.5) * k)
        matrix[0] *= 1 / math.sqrt(2)
        matrix *= math.sqrt(2 / n_mels)
        return matrix

    @classmethod
    def _block_features(cls, signal: Any, rate: int, np: Any) -> Any:
        rms = math.sqrt(float(np.mean(signal * signal)) + 1e-12)
        log_rms = math.log(rms + 1e-12)

        power, frequencies = cls._spectrum(signal, rate, np)
        mel = np.maximum(power @ cls._mel_bank(rate, np).T, 1e-12)
        coefficients = np.log(mel) @ cls._dct(np).T
        mfcc = np.median(coefficients[:, 1:], axis=0)

        valid = (frequencies >= 55) & (frequencies <= min(rate / 2, 5000))
        frequencies = frequencies[valid]
        selected_power = power[:, valid]
        midi = 69 + 12 * np.log2(frequencies / 440)
        pitch_classes = np.mod(np.rint(midi).astype(int), 12)
        chroma = np.zeros((power.shape[0], 12))
        for note in range(12):
            mask = pitch_classes == note
            if np.any(mask):
                chroma[:, note] = np.sum(selected_power[:, mask], axis=1)

        chroma /= np.maximum(np.sum(chroma, axis=1, keepdims=True), 1e-12)
        chroma_vector = np.median(chroma, axis=0)
        norm = np.linalg.norm(chroma_vector)
        if norm > 1e-12:
            chroma_vector /= norm

        return np.concatenate(([log_rms], mfcc, chroma_vector))

    @staticmethod
    def _robust_normalize(matrix: Any, np: Any) -> Any:
        center = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - center), axis=0) * 1.4826
        q1 = np.quantile(matrix, 0.25, axis=0)
        q3 = np.quantile(matrix, 0.75, axis=0)
        scale = np.maximum.reduce(
            [
                mad,
                (q3 - q1) / 1.349,
                np.ptp(matrix, axis=0) * 0.05,
                np.full(matrix.shape[1], 1e-9),
            ]
        )
        return (matrix - center) / scale

    @staticmethod
    def _distance(left: Any, right: Any, np: Any) -> float:
        return float(np.sqrt(np.mean((left - right) ** 2)))

    @classmethod
    def _segment_variability(cls, segment: Any, np: Any) -> float:
        if len(segment) < 2:
            return 0.0
        values = [
            cls._distance(segment[left], segment[right], np)
            for left in range(len(segment))
            for right in range(left + 1, len(segment))
        ]
        return float(np.median(values))

    @staticmethod
    def _contraction_ratio(
        pre: tuple[float, ...],
        post: tuple[float, ...],
        bins: int,
    ) -> float:
        if bins < 1 or bins > len(pre) or bins > len(post):
            raise ValueError(f"invalid temporal variability scale: {bins} bins")
        pre_mean = statistics.fmean(pre[-bins:])
        post_mean = statistics.fmean(post[:bins])
        return post_mean / max(pre_mean, 1e-9)
