"""Promotion of autonomous shadow observations into actionable boundaries."""

from __future__ import annotations

from typing import Protocol

from brynse.boundaries import BoundaryConfidence
from brynse.frames import IncrementalFrameTimeline
from brynse.matching import TemporalSplitAligner
from brynse.models import (
    BoundaryCandidate,
    SplitDecision,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from brynse.orchestrator import CandidateResolution
from brynse.shadow import ShadowBoundaryAnalysis


class SplitAligner(Protocol):
    """Align temporal boundary decisions to encoded stream offsets."""

    def align(
        self,
        *,
        decision: TemporalSplitDecision,
        timeline: IncrementalFrameTimeline,
    ) -> SplitDecision: ...


class AutonomousBoundaryPromoter:
    """Promote accepted autonomous observations into generic boundaries.

    Promotion deliberately introduces no new acoustic scoring or confidence
    policy. The initial policy accepts only observations that the existing
    shadow confidence evaluator has already classified as HIGH.

    The result remains source-agnostic: radio metadata, track identity, and
    output naming are responsibilities of downstream integrations.
    """

    def __init__(
        self,
        *,
        split_aligner: SplitAligner | None = None,
    ) -> None:
        self._split_aligner = split_aligner or TemporalSplitAligner()

    def promote(
        self,
        *,
        analysis: ShadowBoundaryAnalysis,
        timeline: IncrementalFrameTimeline,
    ) -> CandidateResolution | None:
        """Promote one HIGH autonomous observation to a hard boundary."""
        if analysis.assessment.candidate_confidence is not BoundaryConfidence.HIGH:
            return None

        time_seconds = analysis.hypothesis.time_seconds

        candidate = BoundaryCandidate(
            time_seconds=time_seconds,
            source="autonomous",
        )
        temporal = TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=time_seconds,
            outgoing_end_seconds=time_seconds,
        )
        split = self._split_aligner.align(
            decision=temporal,
            timeline=timeline,
        )

        return CandidateResolution(
            candidate=candidate,
            temporal=temporal,
            split=split,
        )
