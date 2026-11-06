"""Find audio that recurs verbatim inside one continuous stream.

Broadcast streams repeat some audio exactly: station idents, jingles,
sweepers, sponsor tags and ads are played from the same file many times a
day. When a short region of the stream is acoustically identical to an
earlier region, both its existence and its exact edges are known without any
external catalogue: the edges are where the two copies stop matching.

This module is source-agnostic. It reports *recurring regions* and leaves the
meaning (jingle, advert, repeated track) to domain integrations.

Method
------
1. Log-magnitude STFT of mono PCM (512-point frames, 256 hop).
2. Spectral peaks (local maxima in time and frequency, top ``peaks_per_frame``
   per frame) restricted to a mid band that survives low-bitrate codecs.
3. Landmark hashes from peak pairs ``(f1, f2, dt)`` anchored at ``t1``
   (the constellation approach used by audio fingerprinting).
4. Two occurrences of the same hash at times ``ta < tb`` vote for a lag
   ``tb - ta``. Dense runs of votes on one lag are candidate recurrences.
5. Each candidate is aligned at sample precision by cross-correlation and
   its edges are refined on short frames where the two aligned copies stop
   matching, giving ~20 ms edge resolution.
6. Occurrences of the same clip are grouped into ``RecurringClip`` objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from brynse.models import DecodedPcm

FloatArray: TypeAlias = npt.NDArray[np.float32]
IntArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class RecurrenceOccurrence:
    """One place where a recurring clip was heard."""

    start_seconds: float
    end_seconds: float
    similarity: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class RecurringClip:
    """Audio heard at least twice, with every occurrence found."""

    clip_id: int
    duration_seconds: float
    occurrences: tuple[RecurrenceOccurrence, ...]


@dataclass(frozen=True)
class RecurrenceSettings:
    """Tunable detector parameters. Defaults target 8 kHz analysis PCM."""

    n_fft: int = 512
    hop: int = 256
    min_frequency_hz: float = 300.0
    max_frequency_hz: float = 3600.0
    peaks_per_frame: int = 5
    peak_time_radius: int = 2
    peak_frequency_radius: int = 4
    fan_out: int = 4
    target_zone_frames: int = 32
    max_hash_occurrences: int = 40
    min_lag_seconds: float = 60.0
    min_votes: int = 12
    max_vote_gap_seconds: float = 1.5
    min_duration_seconds: float = 0.75
    max_duration_seconds: float = 45.0
    edge_frame_seconds: float = 0.02
    edge_search_seconds: float = 3.0
    edge_similarity: float = 0.5
    min_similarity: float = 0.8


class RecurrenceDetector:
    """Detect verbatim recurring regions in one mono 16-bit PCM signal."""

    def __init__(self, settings: RecurrenceSettings | None = None) -> None:
        self._settings = settings or RecurrenceSettings()
        if self._settings.hop <= 0 or self._settings.n_fft <= 0:
            raise ValueError("n_fft and hop must be greater than zero")
        if not 0 < self._settings.min_duration_seconds < self._settings.max_duration_seconds:
            raise ValueError("duration limits must be positive and increasing")
        if not 0 <= self._settings.min_frequency_hz < self._settings.max_frequency_hz:
            raise ValueError("frequency limits must be nonnegative and increasing")
        if self._settings.peaks_per_frame <= 0 or self._settings.fan_out <= 0:
            raise ValueError("peaks_per_frame and fan_out must be positive")
        if self._settings.peak_time_radius < 0 or self._settings.peak_frequency_radius < 0:
            raise ValueError("peak radii must be nonnegative")
        if not 0 < self._settings.edge_similarity <= 1:
            raise ValueError("edge_similarity must be in (0, 1]")
        if not 0 < self._settings.min_similarity <= 1:
            raise ValueError("min_similarity must be in (0, 1]")
        if self._settings.min_votes <= 0 or self._settings.max_hash_occurrences <= 1:
            raise ValueError("vote thresholds must be positive")
        if self._settings.min_lag_seconds < 0 or self._settings.max_vote_gap_seconds <= 0:
            raise ValueError("lag must be nonnegative and vote gap positive")
        if self._settings.edge_frame_seconds <= 0 or self._settings.edge_search_seconds <= 0:
            raise ValueError("edge frame and search durations must be positive")
        if not 0 < self._settings.target_zone_frames <= 63:
            raise ValueError("target_zone_frames must be between 1 and 63")

    @property
    def settings(self) -> RecurrenceSettings:
        return self._settings

    def detect(self, pcm: DecodedPcm) -> tuple[RecurringClip, ...]:
        if pcm.sample_width_bytes != 2 or pcm.channels != 1:
            raise ValueError("recurrence detection requires mono 16-bit PCM")
        if pcm.sample_rate <= 0:
            raise ValueError("PCM sample rate must be positive")
        if len(pcm.data) % 2:
            raise ValueError("PCM data must contain complete 16-bit samples")

        signal = _pcm_float(pcm)
        rate = pcm.sample_rate
        frames_per_second = rate / self._settings.hop

        peaks_t, peaks_f = self._peaks(signal, rate)
        hashes, anchor_times = self._landmarks(peaks_t, peaks_f)
        lags, votes_t = self._votes(hashes, anchor_times, frames_per_second)
        coarse = self._vote_runs(lags, votes_t, frames_per_second)

        pairs: list[tuple[RecurrenceOccurrence, RecurrenceOccurrence]] = []
        for lag_frames, start_frame, end_frame in coarse:
            refined = self._refine(
                signal,
                rate,
                lag_samples=lag_frames * self._settings.hop,
                start_sample=start_frame * self._settings.hop,
                end_sample=(end_frame + 1) * self._settings.hop + self._settings.n_fft,
            )
            if refined is not None:
                pairs.append(refined)

        return _group_clips(pairs)

    # -- fingerprint ---------------------------------------------------------

    def _peaks(self, signal: FloatArray, rate: int) -> tuple[IntArray, IntArray]:
        s = self._settings
        if len(signal) < s.n_fft:
            empty: IntArray = np.empty(0, dtype=np.int64)
            return empty, empty

        low = max(1, int(s.min_frequency_hz * s.n_fft / rate))
        high = min(s.n_fft // 2, int(s.max_frequency_hz * s.n_fft / rate))
        if high <= low:
            raise ValueError("frequency band has no FFT bins at this sample rate")
        window: FloatArray = np.hanning(s.n_fft).astype(np.float32)

        total_frames = 1 + (len(signal) - s.n_fft) // s.hop
        block = 4096
        margin = s.peak_time_radius
        times: list[IntArray] = []
        freqs: list[IntArray] = []

        for block_start in range(0, total_frames, block):
            first = max(0, block_start - margin)
            last = min(total_frames, block_start + block + margin)
            frames = np.lib.stride_tricks.sliding_window_view(
                signal[first * s.hop : (last - 1) * s.hop + s.n_fft], s.n_fft
            )[:: s.hop]
            spectrum = np.abs(np.fft.rfft(frames * window, axis=1))[:, low:high]
            log_spec = (20.0 * np.log10(spectrum + 1e-6)).astype(np.float32)

            neighbourhood = _max_filter(log_spec, s.peak_time_radius, s.peak_frequency_radius)
            floor = np.median(log_spec, axis=1, keepdims=True) + 6.0
            candidate = (log_spec == neighbourhood) & (log_spec > floor)

            keep_from = block_start - first
            keep_to = keep_from + min(block, total_frames - block_start)
            candidate = candidate[keep_from:keep_to]
            values = np.where(candidate, log_spec[keep_from:keep_to], -np.inf)

            k = min(s.peaks_per_frame, values.shape[1])
            top = np.argpartition(-values, k - 1, axis=1)[:, :k]
            top_values = np.take_along_axis(values, top, axis=1)
            frame_index, slot = np.nonzero(np.isfinite(top_values))
            times.append((frame_index + block_start).astype(np.int64))
            freqs.append(top[frame_index, slot].astype(np.int64))

        peak_times = np.concatenate(times)
        peak_freqs = np.concatenate(freqs)
        order = np.lexsort((peak_freqs, peak_times))
        return peak_times[order], peak_freqs[order]

    def _landmarks(self, times: IntArray, freqs: IntArray) -> tuple[IntArray, IntArray]:
        s = self._settings
        if len(times) < 2:
            empty: IntArray = np.empty(0, dtype=np.int64)
            return empty, empty

        first_target = np.searchsorted(times, times + 1, side="left")
        hashes: list[IntArray] = []
        anchors: list[IntArray] = []
        for offset in range(s.fan_out):
            target = first_target + offset
            valid = target < len(times)
            target = np.where(valid, target, 0)
            dt = times[target] - times
            valid &= (dt > 0) & (dt <= s.target_zone_frames)
            anchor = np.nonzero(valid)[0]
            target = target[anchor]
            hashes.append((freqs[anchor] << 14) | (freqs[target] << 6) | (dt[anchor] & 0x3F))
            anchors.append(times[anchor])

        return np.concatenate(hashes), np.concatenate(anchors)

    def _votes(
        self,
        hashes: IntArray,
        anchor_times: IntArray,
        frames_per_second: float,
    ) -> tuple[IntArray, IntArray]:
        s = self._settings
        if len(hashes) < 2:
            empty: IntArray = np.empty(0, dtype=np.int64)
            return empty, empty

        order = np.lexsort((anchor_times, hashes))
        h = hashes[order]
        t = anchor_times[order]

        _, bucket_start, bucket_size = np.unique(h, return_index=True, return_counts=True)
        size_of: IntArray = np.repeat(bucket_size, bucket_size)
        usable = size_of <= s.max_hash_occurrences
        min_lag = int(s.min_lag_seconds * frames_per_second)

        lags: list[IntArray] = []
        starts: list[IntArray] = []
        for k in range(1, s.max_hash_occurrences):
            same = (h[:-k] == h[k:]) & usable[:-k]
            if not same.any():
                break
            lag = t[k:][same] - t[:-k][same]
            far = lag >= min_lag
            lags.append(lag[far])
            starts.append(t[:-k][same][far])

        if not lags:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        return np.concatenate(lags), np.concatenate(starts)

    def _vote_runs(
        self,
        lags: IntArray,
        starts: IntArray,
        frames_per_second: float,
    ) -> list[tuple[int, int, int]]:
        """Return (lag, first_frame, last_frame) runs of consistent votes."""
        s = self._settings
        if len(lags) == 0:
            return []

        max_gap = max(1, int(s.max_vote_gap_seconds * frames_per_second))
        unique_lags, counts = np.unique(lags, return_counts=True)
        strong = unique_lags[counts >= max(2, s.min_votes // 3)]

        # Index votes by lag once. Searching adjacent lag buckets avoids
        # rescanning every vote for every candidate on long recordings.
        lag_order: IntArray = np.argsort(lags, kind="stable")
        ordered_lags = lags[lag_order]
        ordered_starts = starts[lag_order]

        runs: list[tuple[int, int, int]] = []
        for lag in strong:
            # Preserve the original +/-1 frame tolerance exactly.
            left = int(np.searchsorted(ordered_lags, lag - 1, side="left"))
            right = int(np.searchsorted(ordered_lags, lag + 1, side="right"))
            times = np.sort(ordered_starts[left:right])
            if len(times) < s.min_votes:
                continue
            breaks = np.nonzero(np.diff(times) > max_gap)[0]
            segment_starts = np.concatenate(([0], breaks + 1))
            segment_ends = np.concatenate((breaks, [len(times) - 1]))
            for first, last in zip(segment_starts, segment_ends, strict=True):
                if last - first + 1 >= s.min_votes:
                    runs.append((int(lag), int(times[first]), int(times[last])))

        # Collapse duplicates produced by neighbouring lags.
        runs.sort()
        merged: list[tuple[int, int, int]] = []
        for lag, first, last in runs:
            if merged:
                previous_lag, previous_first, previous_last = merged[-1]
                if abs(lag - previous_lag) <= 1 and first <= previous_last:
                    merged[-1] = (previous_lag, previous_first, max(previous_last, last))
                    continue
            merged.append((lag, first, last))
        return merged

    # -- alignment and edges ------------------------------------------------

    def _refine(
        self,
        signal: FloatArray,
        rate: int,
        *,
        lag_samples: int,
        start_sample: int,
        end_sample: int,
    ) -> tuple[RecurrenceOccurrence, RecurrenceOccurrence] | None:
        s = self._settings
        hop = s.hop

        # Sample-accurate lag from cross-correlation of the matched core.
        core_a = signal[start_sample:end_sample]
        b_start = start_sample + lag_samples - hop
        b_end = end_sample + lag_samples + hop
        if b_start < 0 or b_end > len(signal) or len(core_a) < hop:
            return None
        core_b = signal[b_start:b_end]
        correlation = _xcorr_valid(core_b, core_a)
        lag = lag_samples - hop + int(np.argmax(correlation))

        frame = max(1, int(s.edge_frame_seconds * rate))
        search = int(s.edge_search_seconds * rate)
        left = max(0, start_sample - search, -lag)
        right = min(len(signal) - lag, end_sample + search, len(signal))
        if right - left < frame * 4:
            return None

        a = signal[left:right]
        b = signal[left + lag : right + lag]
        count = len(a) // frame
        a_frames = a[: count * frame].reshape(count, frame).astype(np.float64)
        b_frames = b[: count * frame].reshape(count, frame).astype(np.float64)
        energy = (a_frames**2).sum(axis=1) + (b_frames**2).sum(axis=1)
        residual = ((a_frames - b_frames) ** 2).sum(axis=1)
        # 1 = identical, 0 = unrelated (or both silent).
        similarity = np.where(energy > 1e-3, 1.0 - residual / np.maximum(energy, 1e-12), 0.0)
        matched = _smooth_bool(similarity >= s.edge_similarity, radius=2)

        core_frame = min(count - 1, max(0, (start_sample + end_sample) // 2 - left) // frame)
        if not matched[core_frame]:
            hits = np.nonzero(matched)[0]
            if len(hits) == 0:
                return None
            core_frame = int(hits[np.argmin(np.abs(hits - core_frame))])

        first = core_frame
        while first > 0 and matched[first - 1]:
            first -= 1
        last = core_frame
        while last + 1 < count and matched[last + 1]:
            last += 1

        start = (left + first * frame) / rate
        end = (left + (last + 1) * frame) / rate
        duration = end - start
        if not s.min_duration_seconds <= duration <= s.max_duration_seconds:
            return None
        if first == 0 or last == count - 1:
            # Not bounded by non-matching audio inside the search span:
            # a long repeat (e.g. the same song twice), not a short clip.
            return None

        score = float(np.mean(similarity[first : last + 1]))
        if score < s.min_similarity:
            return None

        offset = lag / rate
        return (
            RecurrenceOccurrence(start, end, score),
            RecurrenceOccurrence(start + offset, end + offset, score),
        )


def _pcm_float(pcm: DecodedPcm) -> FloatArray:
    usable = len(pcm.data) - (len(pcm.data) % 2)
    samples: npt.NDArray[np.int16] = np.frombuffer(pcm.data[:usable], dtype="<i2")
    return (samples.astype(np.float32) / 32768.0).astype(np.float32)


def _max_filter(values: FloatArray, time_radius: int, frequency_radius: int) -> FloatArray:
    result = values.copy()
    padded_t: FloatArray = np.pad(
        values, ((time_radius, time_radius), (0, 0)), constant_values=-np.inf
    )
    for shift in range(2 * time_radius + 1):
        np.maximum(result, padded_t[shift : shift + len(values)], out=result)
    padded_f = np.pad(
        result, ((0, 0), (frequency_radius, frequency_radius)), constant_values=-np.inf
    )
    output = result.copy()
    width = values.shape[1]
    for shift in range(2 * frequency_radius + 1):
        np.maximum(output, padded_f[:, shift : shift + width], out=output)
    return output


def _xcorr_valid(longer: FloatArray, shorter: FloatArray) -> npt.NDArray[np.float64]:
    size = len(longer) + len(shorter)
    n = 1 << (size - 1).bit_length()
    spectrum = np.fft.rfft(longer, n) * np.conj(np.fft.rfft(shorter, n))
    full = np.fft.irfft(spectrum, n)
    return np.asarray(full[: len(longer) - len(shorter) + 1], dtype=np.float64)


def _smooth_bool(mask: npt.NDArray[np.bool_], radius: int) -> npt.NDArray[np.bool_]:
    """Close gaps shorter than ``radius`` frames inside matching runs."""
    if radius <= 0 or len(mask) == 0:
        return mask
    kernel = np.ones(2 * radius + 1)
    dilated = np.convolve(mask.astype(float), kernel, mode="same") > 0
    eroded = np.convolve((~dilated).astype(float), kernel, mode="same") > 0
    return np.asarray(~eroded & dilated | mask, dtype=bool)


def _group_clips(
    pairs: Sequence[tuple[RecurrenceOccurrence, RecurrenceOccurrence]],
) -> tuple[RecurringClip, ...]:
    """Union occurrences that overlap into clips heard N times."""
    occurrences: list[RecurrenceOccurrence] = []
    parent: list[int] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[find(left)] = find(right)

    def index_of(item: RecurrenceOccurrence) -> int:
        for position, existing in enumerate(occurrences):
            overlap = min(existing.end_seconds, item.end_seconds) - max(
                existing.start_seconds, item.start_seconds
            )
            shorter = min(existing.duration_seconds, item.duration_seconds)
            if shorter > 0 and overlap / shorter >= 0.5:
                return position
        occurrences.append(item)
        parent.append(len(parent))
        return len(occurrences) - 1

    for first, second in pairs:
        union(index_of(first), index_of(second))

    groups: dict[int, list[RecurrenceOccurrence]] = {}
    for position, item in enumerate(occurrences):
        groups.setdefault(find(position), []).append(item)

    clips = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda item: item.start_seconds)
        clips.append(
            (
                float(np.median([item.duration_seconds for item in members])),
                tuple(members),
            )
        )
    clips.sort(key=lambda clip: clip[1][0].start_seconds)
    return tuple(
        RecurringClip(clip_id=index + 1, duration_seconds=duration, occurrences=members)
        for index, (duration, members) in enumerate(clips)
    )
