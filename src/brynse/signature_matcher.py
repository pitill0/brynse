"""Recognize stored acoustic references in offline radio PCM.

Matching is source-agnostic and does not decide audio cut boundaries.
The first 2-second probe proposes positions; a second overlapping
2-second probe confirms each candidate independently at its offset.

Full sessions are searched in bounded blocks to limit FFT memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from brynse.models import DecodedPcm
from brynse.signature_catalog import (
    PROBE_OFFSET_SECONDS,
    PROBE_SECONDS,
    SAMPLE_RATE,
    SignatureCatalog,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]

_BLOCK_SAMPLES = SAMPLE_RATE * 120
_PROBE_SAMPLES = int(PROBE_SECONDS * SAMPLE_RATE)
_SHIFT_SAMPLES = int(PROBE_OFFSET_SECONDS * SAMPLE_RATE)
_REFERENCE_SAMPLES = _PROBE_SAMPLES + _SHIFT_SAMPLES
_GROUP_GAP_SAMPLES = SAMPLE_RATE // 2
_MERGE_SAMPLES = SAMPLE_RATE
_MAX_CANDIDATES = 5000


@dataclass(frozen=True)
class SignatureMatch:
    """A recognized core position, not a semantic cut interval."""

    signature_id: str
    start_seconds: float
    correlation_first: float
    correlation_second: float


def _normalized_score(probe: FloatArray, candidate: FloatArray) -> float:
    x = probe - probe.mean()
    y = candidate - candidate.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-10:
        return 0.0
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def _first_probe_candidates(
    audio: npt.NDArray[np.int16],
    probe: FloatArray,
    threshold: float,
) -> list[tuple[int, float]]:
    """Find the strongest position in every threshold-crossing region."""
    centered = probe - probe.mean()
    energy = float(np.dot(centered, centered))
    if energy <= 1e-8:
        return []

    last_start = len(audio) - _REFERENCE_SAMPLES
    if last_start < 0:
        return []

    candidates: list[tuple[int, float]] = []

    for block_start in range(0, last_start + 1, _BLOCK_SAMPLES):
        positions = min(_BLOCK_SAMPLES, last_start - block_start + 1)
        block_end = block_start + positions + _PROBE_SAMPLES - 1
        block = audio[block_start:block_end].astype(np.float64) / 32768.0

        fft_size = 1 << (len(block) + _PROBE_SAMPLES - 2).bit_length()

        convolution = np.fft.irfft(
            np.fft.rfft(block, fft_size) * np.fft.rfft(centered[::-1], fft_size),
            fft_size,
        )
        numerator = convolution[_PROBE_SAMPLES - 1 : _PROBE_SAMPLES - 1 + positions]

        sums = np.concatenate(([0.0], np.cumsum(block)))
        squares = np.concatenate(([0.0], np.cumsum(block * block)))
        totals = sums[_PROBE_SAMPLES:] - sums[:-_PROBE_SAMPLES]
        powers = squares[_PROBE_SAMPLES:] - squares[:-_PROBE_SAMPLES]
        variance = np.maximum(
            powers - totals * totals / _PROBE_SAMPLES,
            0.0,
        )

        denominator = np.sqrt(energy * variance[:positions])
        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros(positions, dtype=np.float64),
            where=denominator > 1e-8,
        )

        hits = np.flatnonzero(scores >= threshold)
        if len(hits) == 0:
            continue

        boundaries: npt.NDArray[np.int64] = np.flatnonzero(np.diff(hits) > _GROUP_GAP_SAMPLES) + 1

        for group in np.split(hits, boundaries):
            strongest = int(group[np.argmax(scores[group])])
            candidates.append((block_start + strongest, float(scores[strongest])))
            if len(candidates) > _MAX_CANDIDATES:
                raise ValueError("too many acoustic match candidates")

    return candidates


def scan_catalog(
    pcm: DecodedPcm,
    catalog: SignatureCatalog,
    *,
    threshold: float = 0.90,
) -> tuple[SignatureMatch, ...]:
    """Search every stored signature without using timestamps or metadata.

    The returned seconds identify the first probe position. They must not
    be interpreted as the jingle start or as safe track-cut boundaries.
    """
    if (
        pcm.sample_rate != SAMPLE_RATE
        or pcm.channels != 1
        or pcm.sample_width_bytes != 2
        or len(pcm.data) % 2
    ):
        raise ValueError("matching requires valid 8 kHz mono 16-bit PCM")

    if (
        not isinstance(threshold, (float, int))
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise ValueError("threshold must be finite and in (0, 1]")

    if not isinstance(catalog, SignatureCatalog):
        raise ValueError("invalid signature catalog")

    audio: npt.NDArray[np.int16] = np.frombuffer(pcm.data, dtype="<i2")
    if len(audio) < _REFERENCE_SAMPLES or not catalog.signatures:
        return ()

    results: list[SignatureMatch] = []

    for signature in catalog.signatures:
        first_bytes, second_bytes = signature.probes()
        first = np.frombuffer(first_bytes, dtype="<i2").astype(np.float64) / 32768.0
        second = np.frombuffer(second_bytes, dtype="<i2").astype(np.float64) / 32768.0

        candidates = _first_probe_candidates(audio, first, threshold)
        verified: list[tuple[int, float, float]] = []

        for position, initial in candidates:
            offset = position + _SHIFT_SAMPLES
            region = audio[offset : offset + _PROBE_SAMPLES].astype(np.float64) / 32768.0
            confirmation = _normalized_score(second, region)

            if confirmation >= threshold:
                verified.append((position, initial, confirmation))

        verified.sort(key=lambda match: match[0])

        # Consolidate locally adjacent hypotheses for the same signature.
        groups: list[list[tuple[int, float, float]]] = []
        for match in verified:
            if groups and match[0] - groups[-1][-1][0] <= _MERGE_SAMPLES:
                groups[-1].append(match)
            else:
                groups.append([match])

        for group in groups:
            position, first_score, second_score = max(
                group, key=lambda match: min(match[1], match[2])
            )
            results.append(
                SignatureMatch(
                    signature_id=signature.signature_id,
                    start_seconds=position / SAMPLE_RATE,
                    correlation_first=first_score,
                    correlation_second=second_score,
                )
            )

    return tuple(sorted(results, key=lambda match: (match.start_seconds, match.signature_id)))
