"""Refinement of the incoming edge of an existing temporal boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryProposalSource,
)
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.basin import AdaptiveBasinBoundaryDetector
from brynse.boundaries import AcousticBoundaryHypothesisBuilder, BoundaryReconciler
from brynse.shadow import ShadowBoundaryAnalyzer
from brynse.spectral import FfmpegSpectralDistributionAnalyzer
from brynse.structural import StructuralBoundaryDetector


@dataclass(frozen=True)
class IncomingBoundaryObservation:
    """Read-only diagnostic observation of incoming refinement."""

    outgoing_end_seconds: float
    hypotheses: tuple
    analyses: tuple
    selected_time_seconds: float | None
    decision: TemporalSplitDecision


IncomingObservationCallback = Callable[[IncomingBoundaryObservation], None]


class IncomingBoundaryRefiner:
    """Refine an incoming edge without moving the resolved outgoing edge."""

    def refine(
        self,
        *,
        decision: TemporalSplitDecision,
        incoming_time_seconds: float | None,
    ) -> TemporalSplitDecision:
        if incoming_time_seconds is None:
            return decision

        if decision.kind is not TemporalSplitKind.HARD_CUT:
            return decision

        outgoing_end = decision.outgoing_end_seconds

        if incoming_time_seconds <= outgoing_end:
            return decision

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing_end,
            incoming_start_seconds=incoming_time_seconds,
        )


class IncomingBoundaryDiscovery:
    """Select a useful incoming edge after a frozen outgoing edge."""

    def discover(
        self,
        *,
        outgoing_end_seconds: float,
        analyses,
    ) -> float | None:
        high_confidence = tuple(
            analysis.hypothesis.time_seconds
            for analysis in analyses
            if (
                analysis.assessment.candidate_confidence
                is BoundaryConfidence.HIGH
                and analysis.hypothesis.time_seconds > outgoing_end_seconds
            )
        )

        if high_confidence:
            return min(high_confidence)

        # Several reconciled basin+structural events after the outgoing
        # edge usually describe an interstitial region:
        #
        #   track -> quiet/jingle -> attack
        #
        # IncomingBoundaryAnalyzer already restricts analyses to
        # BASIN+STRUCTURAL hypotheses, so use the earliest preserved
        # basin recovery as the clean start of the incoming segment.
        recoveries = tuple(
            geometry.recovery_seconds
            for analysis in analyses
            if (
                (
                    geometry := getattr(
                        analysis.hypothesis,
                        "basin_geometry",
                        None,
                    )
                )
                is not None
                and geometry.recovery_seconds > outgoing_end_seconds
            )
        )

        if len(recoveries) < 2:
            return None

        return min(recoveries)


class IncomingBoundaryAnalyzer:
    """Analyze plausible acoustic hypotheses for a refined incoming edge."""

    def __init__(self, *, analyzer) -> None:
        self._analyzer = analyzer

    def analyze(
        self,
        *,
        hypotheses,
        pcm,
        absolute_start_time_seconds: float = 0.0,
        outgoing_end_seconds: float | None = None,
    ) -> tuple:
        analyses = []

        for hypothesis in hypotheses:
            if (
                outgoing_end_seconds is not None
                and hypothesis.time_seconds <= outgoing_end_seconds
            ):
                continue

            sources = {proposal.source for proposal in hypothesis.proposals}
            if sources != {
                BoundaryProposalSource.BASIN,
                BoundaryProposalSource.STRUCTURAL,
            }:
                continue

            analyses.append(
                self._analyzer.analyze(
                    hypothesis=hypothesis,
                    pcm=pcm,
                    absolute_start_time_seconds=absolute_start_time_seconds,
                    measure_spectral=True,
                    measure_temporal=False,
                    measure_local_discontinuity=False,
                )
            )

        return tuple(analyses)


class IncomingBoundaryPipeline:
    """Compose acoustic evidence into optional incoming-edge refinement."""

    def __init__(
        self,
        *,
        hypothesis_builder,
        incoming_analyzer,
        discovery: IncomingBoundaryDiscovery,
        refiner: IncomingBoundaryRefiner,
        on_observation: IncomingObservationCallback | None = None,
    ) -> None:
        self._hypothesis_builder = hypothesis_builder
        self._incoming_analyzer = incoming_analyzer
        self._discovery = discovery
        self._refiner = refiner
        self._on_observation = on_observation

    def refine(
        self,
        *,
        decision: TemporalSplitDecision,
        window,
        pcm8,
        pcm16,
    ) -> TemporalSplitDecision:
        absolute_start = window.start_time_seconds
        outgoing_end = decision.outgoing_end_seconds

        hypotheses = self._hypothesis_builder.build(
            pcm=pcm8,
            absolute_start_time_seconds=absolute_start,
        )

        analyses = self._incoming_analyzer.analyze(
            hypotheses=hypotheses,
            pcm=pcm16,
            absolute_start_time_seconds=absolute_start,
            outgoing_end_seconds=outgoing_end,
        )

        incoming_time = self._discovery.discover(
            outgoing_end_seconds=outgoing_end,
            analyses=analyses,
        )

        result = self._refiner.refine(
            decision=decision,
            incoming_time_seconds=incoming_time,
        )

        if self._on_observation is not None:
            self._on_observation(
                IncomingBoundaryObservation(
                    outgoing_end_seconds=outgoing_end,
                    hypotheses=hypotheses,
                    analyses=analyses,
                    selected_time_seconds=incoming_time,
                    decision=result,
                )
            )

        return result


def build_incoming_boundary_pipeline(
    *,
    ffmpeg_binary: str = "ffmpeg",
    on_observation: IncomingObservationCallback | None = None,
) -> IncomingBoundaryPipeline:
    """Build incoming refinement from the canonical acoustic components."""

    hypothesis_builder = AcousticBoundaryHypothesisBuilder(
        basin_detector=AdaptiveBasinBoundaryDetector(),
        structural_detector=StructuralBoundaryDetector(),
        reconciler=BoundaryReconciler(),
    )

    shadow_analyzer = ShadowBoundaryAnalyzer(
        spectral_analyzer=FfmpegSpectralDistributionAnalyzer(
            ffmpeg_binary=ffmpeg_binary,
        )
    )

    return IncomingBoundaryPipeline(
        hypothesis_builder=hypothesis_builder,
        incoming_analyzer=IncomingBoundaryAnalyzer(
            analyzer=shadow_analyzer,
        ),
        discovery=IncomingBoundaryDiscovery(),
        refiner=IncomingBoundaryRefiner(),
        on_observation=on_observation,
    )
