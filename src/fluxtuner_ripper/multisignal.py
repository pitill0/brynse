from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultiSignalCandidate:
    time_seconds: float
    family_rrf: float
    forward_novelty: float


NOVELTY_SAMPLE_RATE = 8000
NOVELTY_FRAME_SAMPLES = 512
NOVELTY_HOP_SAMPLES = 128
NOVELTY_WINDOW_SECONDS = 0.750
NOVELTY_MIN_FREQUENCY_HZ = 80.0
NOVELTY_MAX_FREQUENCY_HZ = 3800.0


def _spectral_frames(
    samples: np.ndarray,
) -> np.ndarray:
    window = np.hanning(NOVELTY_FRAME_SAMPLES)

    frequencies = np.fft.rfftfreq(
        NOVELTY_FRAME_SAMPLES,
        d=1.0 / NOVELTY_SAMPLE_RATE,
    )

    frequency_mask = (frequencies >= NOVELTY_MIN_FREQUENCY_HZ) & (
        frequencies <= NOVELTY_MAX_FREQUENCY_HZ
    )

    frames: list[np.ndarray] = []

    for start in range(
        0,
        (len(samples) - NOVELTY_FRAME_SAMPLES + 1),
        NOVELTY_HOP_SAMPLES,
    ):
        frame = samples[start : start + NOVELTY_FRAME_SAMPLES] * window

        spectrum = np.abs(np.fft.rfft(frame)) ** 2

        vector = np.log1p(spectrum[frequency_mask])

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        frames.append(vector)

    return np.asarray(
        frames,
        dtype=np.float64,
    )


def _directed_novelty(
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    similarity = target @ source.T

    best_explanation = np.max(
        similarity,
        axis=1,
    )

    return float(np.mean(1.0 - best_explanation))


def forward_novelty(
    audio: np.ndarray,
    *,
    center_seconds: float,
) -> float:
    center = round(center_seconds * NOVELTY_SAMPLE_RATE)

    window_samples = round(NOVELTY_WINDOW_SECONDS * NOVELTY_SAMPLE_RATE)

    pre = audio[center - window_samples : center]

    post = audio[center : center + window_samples]

    source = _spectral_frames(pre)

    target = _spectral_frames(post)

    return _directed_novelty(
        source,
        target,
    )


SEMANTIC_MIN_OFFSET_SECONDS = 1.0
SEMANTIC_MAX_OFFSET_SECONDS = 7.0


@dataclass(frozen=True)
class UnionCandidate:
    time_seconds: float
    sources: tuple[str, ...]


def build_union_candidates(
    *,
    semantic_time: float,
    basin_times: Sequence[float],
    d2_points: Sequence[tuple[float, float]],
) -> tuple[UnionCandidate, ...]:
    pool: dict[float, list[str]] = {}

    def add_candidate(
        *,
        time: float,
        source: str,
    ) -> None:
        key = round(
            float(time),
            3,
        )

        sources = pool.setdefault(
            key,
            [],
        )

        if source not in sources:
            sources.append(source)

    for time in basin_times:
        offset = semantic_time - float(time)

        if SEMANTIC_MIN_OFFSET_SECONDS <= offset <= SEMANTIC_MAX_OFFSET_SECONDS:
            add_candidate(
                time=time,
                source="BASIN",
            )

    points = [
        (
            round(float(time), 6),
            float(strength),
        )
        for time, strength in d2_points
    ]

    for index, (
        time,
        strength,
    ) in enumerate(points):
        previous = points[index - 1][1] if index > 0 else float("-inf")

        following = points[index + 1][1] if index + 1 < len(points) else float("-inf")

        if strength >= previous and strength >= following:
            add_candidate(
                time=time,
                source="D2",
            )

    return tuple(
        UnionCandidate(
            time_seconds=time,
            sources=tuple(sources),
        )
        for time, sources in sorted(pool.items())
    )


@dataclass(frozen=True)
class FamilyRankCandidate:
    time_seconds: float
    d250: float
    d500: float
    d1: float
    d2: float
    mfcc_geo: float
    family_short_rank: float = 0.0
    family_long_rank: float = 0.0
    family_regime_rank: float = 0.0
    family_rrf: float = 0.0


def assign_family_rrf(
    candidates: Sequence[FamilyRankCandidate],
) -> tuple[FamilyRankCandidate, ...]:
    if not candidates:
        return ()

    metric_ranks: dict[str, dict[int, int]] = {}

    for metric in (
        "d250",
        "d500",
        "d1",
        "d2",
        "mfcc_geo",
    ):
        ordered = sorted(
            range(len(candidates)),
            key=lambda index: (
                -getattr(candidates[index], metric),
                candidates[index].time_seconds,
            ),
        )

        metric_ranks[metric] = {
            candidate_index: rank
            for rank, candidate_index in enumerate(
                ordered,
                start=1,
            )
        }

    ranked: list[FamilyRankCandidate] = []

    for index, candidate in enumerate(candidates):
        short_rank = statistics.median(
            (
                metric_ranks["d250"][index],
                metric_ranks["d500"][index],
                metric_ranks["d1"][index],
            )
        )

        long_rank = float(metric_ranks["d2"][index])

        regime_rank = float(metric_ranks["mfcc_geo"][index])

        family_rrf = 1.0 / short_rank + 1.0 / long_rank + 1.0 / regime_rank

        ranked.append(
            FamilyRankCandidate(
                time_seconds=candidate.time_seconds,
                d250=candidate.d250,
                d500=candidate.d500,
                d1=candidate.d1,
                d2=candidate.d2,
                mfcc_geo=candidate.mfcc_geo,
                family_short_rank=float(short_rank),
                family_long_rank=long_rank,
                family_regime_rank=regime_rank,
                family_rrf=family_rrf,
            )
        )

    return tuple(ranked)


class MinimaxBoundarySelector:
    def select(
        self,
        candidates: Sequence[MultiSignalCandidate],
    ) -> MultiSignalCandidate:
        if not candidates:
            raise ValueError("candidates must not be empty")

        indexed = list(enumerate(candidates))

        rrf_order = sorted(
            indexed,
            key=lambda item: (
                -item[1].family_rrf,
                item[1].time_seconds,
            ),
        )

        novelty_order = sorted(
            indexed,
            key=lambda item: (
                -item[1].forward_novelty,
                item[1].time_seconds,
            ),
        )

        rrf_rank = {
            candidate_index: rank
            for rank, (
                candidate_index,
                _,
            ) in enumerate(
                rrf_order,
                start=1,
            )
        }

        novelty_rank = {
            candidate_index: rank
            for rank, (
                candidate_index,
                _,
            ) in enumerate(
                novelty_order,
                start=1,
            )
        }

        selected_index, selected = min(
            indexed,
            key=lambda item: (
                max(
                    rrf_rank[item[0]],
                    novelty_rank[item[0]],
                ),
                (rrf_rank[item[0]] + novelty_rank[item[0]]),
                rrf_rank[item[0]],
                novelty_rank[item[0]],
                item[1].time_seconds,
            ),
        )

        _ = selected_index
        return selected
