"""Conservative PCM verification of a bounded silent region near ICY metadata."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from brynse.models import DecodedPcm, TemporalSplitDecision, TemporalSplitKind


class PcmSilenceGapVerifier:
    """Recognize a unique sustained silence bracketed by audible audio.

    This verifier does not estimate fade boundaries or infer crossfades. If
    the decoded window is missing either audible flank, it abstains.
    """

    def verify(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float,
        reference_time_seconds: float,
    ) -> TemporalSplitDecision | None:
        rate = pcm.sample_rate
        samples: NDArray[np.int16] = np.frombuffer(pcm.data, dtype="<i2")
        bin_samples = max(1, round(rate * 0.025))
        count = len(samples) // bin_samples
        if count < 60:
            return None

        windows = samples[: count * bin_samples].reshape(count, bin_samples)
        levels = np.sqrt(np.mean((windows.astype(np.float64) / 32768.0) ** 2, axis=1))
        quiet = levels < 0.002
        edges = np.diff(np.pad(quiet.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        supported: list[tuple[float, float]] = []
        for first, last in zip(starts, ends, strict=True):
            duration = (last - first) * bin_samples / rate
            if not 0.5 <= duration <= 3.0:
                continue
            left = absolute_start_time_seconds + first * bin_samples / rate
            right = absolute_start_time_seconds + last * bin_samples / rate
            if max(abs(left - reference_time_seconds), abs(right - reference_time_seconds)) > 9.0:
                continue

            flank = round(0.4 * rate / bin_samples)
            if first < flank or last + flank > count:
                continue
            if (
                float(np.sqrt(np.mean(levels[first - flank : first] ** 2))) < 0.01
                or float(np.sqrt(np.mean(levels[last : last + flank] ** 2))) < 0.01
            ):
                continue
            supported.append((left, right))

        if len(supported) != 1:
            return None

        verified_left, verified_right = supported[0]
        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=round(verified_left, 6),
            incoming_start_seconds=round(verified_right, 6),
        )
