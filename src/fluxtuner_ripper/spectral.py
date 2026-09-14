"""Spectral-distribution evidence extraction from decoded PCM."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from fluxtuner_ripper.models import DecodedPcm

_SPECTRAL_METRICS = ("centroid", "spread", "entropy", "flatness", "rolloff")


class SpectralAnalysisError(RuntimeError):
    """Raised when FFmpeg cannot produce spectral-distribution evidence."""


@dataclass(frozen=True)
class SpectralDistributionEvidence:
    """Descriptive spectral context around one candidate boundary."""

    between: float
    before_sequential_variability: float
    after_sequential_variability: float
    before_distribution_spread: float
    after_distribution_spread: float

    def __post_init__(self) -> None:
        values = (
            ("between", self.between),
            ("before_sequential_variability", self.before_sequential_variability),
            ("after_sequential_variability", self.after_sequential_variability),
            ("before_distribution_spread", self.before_distribution_spread),
            ("after_distribution_spread", self.after_distribution_spread),
        )
        for name, value in values:
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


class FfmpegSpectralDistributionAnalyzer:
    """Measure before/after spectral distributions around a PCM boundary."""

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        side_seconds: float = 12.0,
        guard_seconds: float = 2.0,
        block_seconds: float = 2.0,
    ) -> None:
        if side_seconds <= 0:
            raise ValueError("side_seconds must be greater than zero")
        if guard_seconds < 0:
            raise ValueError("guard_seconds must be non-negative")
        if block_seconds <= 0:
            raise ValueError("block_seconds must be greater than zero")
        if block_seconds > side_seconds:
            raise ValueError("block_seconds must not exceed side_seconds")
        if not math.isclose(
            side_seconds / block_seconds,
            round(side_seconds / block_seconds),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("side_seconds must be an exact multiple of block_seconds")

        self._ffmpeg_binary = ffmpeg_binary
        self._side_seconds = side_seconds
        self._guard_seconds = guard_seconds
        self._block_seconds = block_seconds

    @property
    def required_preroll_seconds(self) -> float:
        return self._side_seconds + self._guard_seconds

    @property
    def required_postroll_seconds(self) -> float:
        return self._side_seconds + self._guard_seconds

    def analyze(
        self,
        *,
        pcm: DecodedPcm,
        boundary_time_seconds: float,
        absolute_start_time_seconds: float = 0.0,
    ) -> SpectralDistributionEvidence | None:
        """Return spectral evidence, or ``None`` when context is incomplete."""
        if boundary_time_seconds < 0:
            raise ValueError("boundary_time_seconds must be non-negative")
        if absolute_start_time_seconds < 0:
            raise ValueError("absolute_start_time_seconds must be non-negative")

        relative_boundary = boundary_time_seconds - absolute_start_time_seconds
        if relative_boundary < 0:
            raise ValueError("boundary_time_seconds precedes absolute_start_time_seconds")

        duration_seconds = pcm.sample_count / pcm.sample_rate
        before_start = relative_boundary - self._guard_seconds - self._side_seconds
        after_start = relative_boundary + self._guard_seconds
        after_end = after_start + self._side_seconds

        if before_start < 0 or after_end > duration_seconds:
            return None

        before = self._extract_blocks(pcm=pcm, start_seconds=before_start)
        after = self._extract_blocks(pcm=pcm, start_seconds=after_start)

        before_center = self._center(before)
        after_center = self._center(after)

        return SpectralDistributionEvidence(
            between=self._profile_distance(before_center, after_center),
            before_sequential_variability=self._sequential_variability(before),
            after_sequential_variability=self._sequential_variability(after),
            before_distribution_spread=self._distribution_spread(before),
            after_distribution_spread=self._distribution_spread(after),
        )

    def _extract_blocks(
        self,
        *,
        pcm: DecodedPcm,
        start_seconds: float,
    ) -> tuple[dict[str, float], ...]:
        block_count = round(self._side_seconds / self._block_seconds)
        return tuple(
            self._extract_profile(
                pcm=pcm,
                start_seconds=start_seconds + index * self._block_seconds,
                duration_seconds=self._block_seconds,
            )
            for index in range(block_count)
        )

    def _extract_profile(
        self,
        *,
        pcm: DecodedPcm,
        start_seconds: float,
        duration_seconds: float,
    ) -> dict[str, float]:
        import subprocess  # nosec B404

        start_sample = round(start_seconds * pcm.sample_rate)
        duration_samples = round(duration_seconds * pcm.sample_rate)
        end_sample = min(pcm.sample_count, start_sample + duration_samples)
        payload = pcm.data[
            start_sample * pcm.sample_width_bytes : end_sample * pcm.sample_width_bytes
        ]
        if not payload:
            raise SpectralAnalysisError("spectral analysis received an empty PCM block")

        measure = "+".join(_SPECTRAL_METRICS)
        filtergraph = (
            "aspectralstats="
            f"win_size=2048:overlap=0.5:measure={measure},"
            "ametadata=mode=print:file=-"
        )
        command = [
            self._ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(pcm.sample_rate),
            "-i",
            "pipe:0",
            "-af",
            filtergraph,
            "-f",
            "null",
            "-",
        ]

        try:
            completed = subprocess.run(  # nosec B603
                command,
                input=payload,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SpectralAnalysisError(f"FFmpeg binary not found: {self._ffmpeg_binary}") from exc

        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise SpectralAnalysisError(
                f"FFmpeg spectral analysis failed with exit code {completed.returncode}: {error}"
            )

        output = completed.stdout.decode("utf-8", errors="replace")
        values: dict[str, list[float]] = {metric: [] for metric in _SPECTRAL_METRICS}
        prefix = "lavfi.aspectralstats.1."

        for line in output.splitlines():
            if not line.startswith(prefix) or "=" not in line:
                continue
            key, raw = line.split("=", 1)
            metric = key[len(prefix) :]
            if metric not in values:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                values[metric].append(value)

        missing = [metric for metric, items in values.items() if not items]
        if missing:
            raise SpectralAnalysisError("missing aspectralstats metadata: " + ", ".join(missing))

        return {metric: statistics.median(items) for metric, items in values.items()}

    @staticmethod
    def _symmetric_change(left: float, right: float) -> float:
        return abs(left - right) / max(abs(left) + abs(right), 1e-12)

    @classmethod
    def _profile_distance(
        cls,
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        return math.sqrt(
            sum(
                cls._symmetric_change(left[metric], right[metric]) ** 2
                for metric in _SPECTRAL_METRICS
            )
        )

    @staticmethod
    def _center(
        blocks: tuple[dict[str, float], ...],
    ) -> dict[str, float]:
        return {
            metric: statistics.median(block[metric] for block in blocks)
            for metric in _SPECTRAL_METRICS
        }

    @classmethod
    def _sequential_variability(
        cls,
        blocks: tuple[dict[str, float], ...],
    ) -> float:
        distances = tuple(
            cls._profile_distance(blocks[index - 1], blocks[index])
            for index in range(1, len(blocks))
        )
        return statistics.median(distances) if distances else 0.0

    @classmethod
    def _distribution_spread(
        cls,
        blocks: tuple[dict[str, float], ...],
    ) -> float:
        center = cls._center(blocks)
        return statistics.median(cls._profile_distance(block, center) for block in blocks)
