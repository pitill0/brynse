from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from brynse.acoustic import RmsAcousticAnalyzer
from brynse.hybrid import HybridAcousticSplitResolver
from brynse.local_discontinuity import LocalDiscontinuityAnalyzer
from brynse.models import (
    AcousticWindow,
    DecodedPcm,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from brynse.temporal_variability import TemporalVariabilityAnalyzer


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


MULTISIGNAL_SAMPLE_RATE = 16000
MULTISIGNAL_D2_STEP_SECONDS = 0.25
MULTISIGNAL_REGIME_SCALES = (
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
)


class MultiSignalTemporalResolver:
    def __init__(
        self,
        *,
        hybrid: Any | None = None,
    ) -> None:
        self._hybrid = hybrid if hybrid is not None else HybridAcousticSplitResolver()

    def resolve(
        self,
        *,
        selected: MultiSignalCandidate,
        profile: Any,
        window: Any,
        semantic_time_seconds: float,
    ) -> TemporalSplitDecision | None:
        outgoing = self._hybrid._select_crossfade_outgoing(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time_seconds,
        )

        boundary = selected.time_seconds

        if outgoing is None:
            return TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )

        if boundary >= outgoing.time_seconds:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.CROSSFADE,
            incoming_start_seconds=boundary,
            outgoing_end_seconds=outgoing.time_seconds,
        )


class AcousticMultiSignalCandidateBuilder:
    def __init__(
        self,
        *,
        rms_analyzer: Any | None = None,
        hybrid: Any | None = None,
        local_analyzer: Any | None = None,
        regime_ratio: Callable[..., float] | None = None,
        novelty: Callable[..., float] | None = None,
        d2_step_seconds: float = MULTISIGNAL_D2_STEP_SECONDS,
    ) -> None:
        if d2_step_seconds <= 0:
            raise ValueError("d2_step_seconds must be greater than zero")

        self._rms = (
            rms_analyzer
            if rms_analyzer is not None
            else RmsAcousticAnalyzer(
                window_seconds=0.05,
            )
        )

        self._hybrid = (
            hybrid
            if hybrid is not None
            else HybridAcousticSplitResolver(
                boundary_distance_penalty=0.0,
            )
        )

        self._local = local_analyzer if local_analyzer is not None else LocalDiscontinuityAnalyzer()

        self._regime_ratio_fn = regime_ratio
        self._novelty_fn = novelty
        self._d2_step_seconds = d2_step_seconds

    @staticmethod
    def _cosine_distance(
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        denominator = np.linalg.norm(left) * np.linalg.norm(right)

        if denominator <= 1e-12:
            return 0.0

        similarity = float(np.dot(left, right) / denominator)

        similarity = max(
            -1.0,
            min(
                1.0,
                similarity,
            ),
        )

        return 1.0 - similarity

    @staticmethod
    def _geomean(
        values: Sequence[float],
    ) -> float:
        return math.exp(
            sum(
                math.log(
                    max(
                        value,
                        1e-12,
                    )
                )
                for value in values
            )
            / len(values)
        )

    @staticmethod
    def _pcm_samples(
        pcm: DecodedPcm,
    ) -> np.ndarray:
        return (
            np.frombuffer(
                pcm.data,
                dtype="<i2",
            ).astype(np.float64)
            / 32768.0
        )

    @staticmethod
    def _mfcc_block(
        *,
        samples: np.ndarray,
        window: AcousticWindow,
        start: float,
        end: float,
    ) -> np.ndarray:
        first = round((start - window.start_time_seconds) * MULTISIGNAL_SAMPLE_RATE)

        last = round((end - window.start_time_seconds) * MULTISIGNAL_SAMPLE_RATE)

        if first < 0 or last > len(samples) or last <= first:
            raise RuntimeError("MFCC block outside decoded acoustic window")

        segment = samples[first:last]

        features = np.asarray(
            TemporalVariabilityAnalyzer._block_features(
                segment,
                MULTISIGNAL_SAMPLE_RATE,
                np,
            ),
            dtype=np.float64,
        )

        # Frozen experiment used only the 12 MFCC values:
        # exclude log RMS at index 0 and chroma after index 12.
        return features[1:13]

    def _regime_ratio(
        self,
        *,
        candidate_time: float,
        scale: float,
        samples: np.ndarray,
        window: AcousticWindow,
    ) -> float:
        if self._regime_ratio_fn is not None:
            return float(
                self._regime_ratio_fn(
                    candidate_time=candidate_time,
                    scale=scale,
                    samples=samples,
                    window=window,
                )
            )

        a = self._mfcc_block(
            samples=samples,
            window=window,
            start=(candidate_time - 2.0 * scale),
            end=(candidate_time - scale),
        )

        b = self._mfcc_block(
            samples=samples,
            window=window,
            start=(candidate_time - scale),
            end=candidate_time,
        )

        c = self._mfcc_block(
            samples=samples,
            window=window,
            start=candidate_time,
            end=(candidate_time + scale),
        )

        d = self._mfcc_block(
            samples=samples,
            window=window,
            start=(candidate_time + scale),
            end=(candidate_time + 2.0 * scale),
        )

        pre = self._cosine_distance(
            a,
            b,
        )

        post = self._cosine_distance(
            c,
            d,
        )

        cross = self._cosine_distance(
            b,
            c,
        )

        internal = (pre + post) / 2.0

        return cross / max(
            internal,
            1e-9,
        )

    def _novelty(
        self,
        *,
        candidate_time: float,
        samples: np.ndarray,
        window: AcousticWindow,
    ) -> float:
        if self._novelty_fn is not None:
            return float(
                self._novelty_fn(
                    candidate_time=candidate_time,
                    samples=samples,
                    window=window,
                )
            )

        relative_time = candidate_time - window.start_time_seconds

        return forward_novelty(
            samples,
            center_seconds=relative_time,
        )

    def build(
        self,
        *,
        semantic_time: float,
        window: AcousticWindow,
        pcm8: DecodedPcm,
        pcm16: DecodedPcm,
    ) -> tuple[MultiSignalCandidate, ...]:
        if pcm8.sample_rate != NOVELTY_SAMPLE_RATE:
            raise ValueError("multisignal novelty requires 8000 Hz PCM")

        if pcm16.sample_rate != MULTISIGNAL_SAMPLE_RATE:
            raise ValueError("multisignal evidence requires 16000 Hz PCM")

        profile = self._rms.analyze(pcm8)

        basins = self._hybrid._basins(
            profile=profile,
            window=window,
            semantic_time_seconds=semantic_time,
        )

        basin_times = tuple(float(basin.center_time_seconds) for basin in basins)

        d2_points: list[tuple[float, float]] = []

        time = semantic_time - SEMANTIC_MAX_OFFSET_SECONDS

        end = semantic_time - SEMANTIC_MIN_OFFSET_SECONDS

        while time <= end + 1e-9:
            evidence = self._local.analyze(
                pcm=pcm16,
                boundary_time_seconds=time,
                absolute_start_time_seconds=(window.start_time_seconds),
            )

            if evidence is not None:
                d2_points.append(
                    (
                        round(
                            time,
                            6,
                        ),
                        float(evidence.change_2s),
                    )
                )

            time += self._d2_step_seconds

        union = build_union_candidates(
            semantic_time=semantic_time,
            basin_times=basin_times,
            d2_points=d2_points,
        )

        if not union:
            return ()

        samples8 = self._pcm_samples(pcm8)

        samples16 = self._pcm_samples(pcm16)

        family_inputs: list[FamilyRankCandidate] = []

        novelty_by_time: dict[float, float] = {}

        for candidate in union:
            candidate_time = candidate.time_seconds

            evidence = self._local.analyze(
                pcm=pcm16,
                boundary_time_seconds=(candidate_time),
                absolute_start_time_seconds=(window.start_time_seconds),
            )

            if evidence is None:
                continue

            ratios = tuple(
                self._regime_ratio(
                    candidate_time=(candidate_time),
                    scale=scale,
                    samples=samples16,
                    window=window,
                )
                for scale in MULTISIGNAL_REGIME_SCALES
            )

            family_inputs.append(
                FamilyRankCandidate(
                    time_seconds=(candidate_time),
                    d250=float(evidence.change_250ms),
                    d500=float(evidence.change_500ms),
                    d1=float(evidence.change_1s),
                    d2=float(evidence.change_2s),
                    mfcc_geo=(self._geomean(ratios)),
                )
            )

            novelty_by_time[candidate_time] = self._novelty(
                candidate_time=(candidate_time),
                samples=samples8,
                window=window,
            )

        ranked = assign_family_rrf(family_inputs)

        return tuple(
            MultiSignalCandidate(
                time_seconds=(candidate.time_seconds),
                family_rrf=(candidate.family_rrf),
                forward_novelty=(novelty_by_time[candidate.time_seconds]),
            )
            for candidate in ranked
        )


CandidateBuilder = Callable[
    ...,
    Sequence[MultiSignalCandidate],
]


@runtime_checkable
class AcousticCandidateBuilder(Protocol):
    def build(
        self,
        *,
        semantic_time: float,
        window: AcousticWindow,
        pcm8: DecodedPcm,
        pcm16: DecodedPcm,
    ) -> Sequence[MultiSignalCandidate]: ...


class MultiSignalCandidateResolver:
    def __init__(
        self,
        *,
        candidate_builder: CandidateBuilder | AcousticCandidateBuilder,
        selector: MinimaxBoundarySelector | None = None,
    ) -> None:
        self._candidate_builder = candidate_builder
        self._selector = selector if selector is not None else MinimaxBoundarySelector()

    def resolve_acoustic_boundary(
        self,
        *,
        semantic_time_seconds: float,
        window: AcousticWindow,
        pcm8: DecodedPcm,
        pcm16: DecodedPcm,
    ) -> MultiSignalCandidate | None:
        builder = self._candidate_builder

        if not hasattr(builder, "build"):
            raise TypeError("candidate_builder does not support acoustic build()")

        candidates = tuple(
            builder.build(
                semantic_time=semantic_time_seconds,
                window=window,
                pcm8=pcm8,
                pcm16=pcm16,
            )
        )

        if not candidates:
            return None

        return self._selector.select(candidates)

    def resolve_selected_boundary(
        self,
        *,
        semantic_time_seconds: float,
    ) -> MultiSignalCandidate | None:
        builder = self._candidate_builder

        if not callable(builder):
            raise TypeError("candidate_builder is not callable")

        callable_builder = cast(
            CandidateBuilder,
            builder,
        )

        candidates = tuple(
            callable_builder(
                semantic_time_seconds=semantic_time_seconds,
            )
        )

        if not candidates:
            return None

        return self._selector.select(candidates)


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
