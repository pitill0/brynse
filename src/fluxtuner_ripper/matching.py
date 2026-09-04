from __future__ import annotations

from fluxtuner_ripper.frames import IncrementalFrameTimeline
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    BoundaryMatch,
    BoundaryRelation,
    BoundaryRelationResult,
    EncodedAudioFrame,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
    TrackCandidate,
)


class NearestBoundaryMatcher:
    """Match a semantic track candidate to the nearest acoustic minimum."""

    def __init__(self, search_radius_seconds: float = 8.0) -> None:
        if search_radius_seconds <= 0:
            raise ValueError("search_radius_seconds must be greater than zero")
        self._search_radius = search_radius_seconds

    @property
    def search_radius_seconds(self) -> float:
        return self._search_radius

    def match(
        self,
        *,
        track: TrackCandidate,
        acoustic_candidates: tuple[AcousticBoundaryCandidate, ...],
    ) -> BoundaryMatch | None:
        """Return the nearest acoustic candidate inside the configured radius."""

        eligible = [
            candidate
            for candidate in acoustic_candidates
            if abs(candidate.time_seconds - track.start_time_seconds) <= self._search_radius
        ]
        if not eligible:
            return None

        selected = min(
            eligible,
            key=lambda candidate: (
                abs(candidate.time_seconds - track.start_time_seconds),
                candidate.rms,
                candidate.time_seconds,
            ),
        )

        return BoundaryMatch(
            track=track,
            acoustic=selected,
            delta_seconds=abs(selected.time_seconds - track.start_time_seconds),
        )


class BoundaryRelationClassifier:
    """Classify semantic-vs-acoustic timing before final split decisions."""

    def __init__(self, divergence_threshold_seconds: float = 3.0) -> None:
        if divergence_threshold_seconds <= 0:
            raise ValueError("divergence_threshold_seconds must be greater than zero")
        self._threshold = divergence_threshold_seconds

    @property
    def divergence_threshold_seconds(self) -> float:
        return self._threshold

    def classify(
        self,
        *,
        semantic_time_seconds: float,
        acoustic_time_seconds: float,
    ) -> BoundaryRelationResult:
        if semantic_time_seconds < 0:
            raise ValueError("semantic_time_seconds must be non-negative")
        if acoustic_time_seconds < 0:
            raise ValueError("acoustic_time_seconds must be non-negative")

        signed_delta = semantic_time_seconds - acoustic_time_seconds

        if signed_delta > self._threshold:
            relation = BoundaryRelation.ACOUSTIC_EARLIER
        elif signed_delta < -self._threshold:
            relation = BoundaryRelation.SEMANTIC_EARLIER
        else:
            relation = BoundaryRelation.AGREEMENT

        return BoundaryRelationResult(
            relation=relation,
            semantic_time_seconds=semantic_time_seconds,
            acoustic_time_seconds=acoustic_time_seconds,
            signed_delta_seconds=signed_delta,
        )


class TemporalSplitPolicy:
    """Convert semantic/acoustic relation into an intermediate temporal split."""

    def decide(self, relation: BoundaryRelationResult) -> TemporalSplitDecision:
        if relation.relation is BoundaryRelation.AGREEMENT:
            boundary = relation.semantic_time_seconds
            return TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )

        if relation.relation is BoundaryRelation.ACOUSTIC_EARLIER:
            return TemporalSplitDecision(
                kind=TemporalSplitKind.CROSSFADE,
                incoming_start_seconds=relation.acoustic_time_seconds,
                outgoing_end_seconds=relation.semantic_time_seconds,
            )

        if relation.relation is BoundaryRelation.SEMANTIC_EARLIER:
            boundary = relation.acoustic_time_seconds
            return TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )

        raise RuntimeError(f"unsupported boundary relation: {relation.relation}")


class SplitAlignmentError(RuntimeError):
    """Raised when a temporal split cannot be aligned to retained codec frames."""


class TemporalSplitAligner:
    """Convert temporal split decisions into frame-aligned byte offsets."""

    def align(
        self,
        *,
        decision: TemporalSplitDecision,
        timeline: IncrementalFrameTimeline,
    ) -> SplitDecision:
        frames = timeline.frames
        if not frames:
            raise SplitAlignmentError("cannot align split without retained frames")

        incoming_frame = self._nearest_frame(
            frames,
            decision.incoming_start_seconds,
        )
        outgoing_frame = self._nearest_frame(
            frames,
            decision.outgoing_end_seconds,
        )

        if incoming_frame is None or outgoing_frame is None:
            raise SplitAlignmentError("temporal split lies outside the retained frame timeline")

        if decision.kind is TemporalSplitKind.HARD_CUT:
            return SplitDecision(
                kind=SplitKind.HARD_CUT,
                incoming_start=incoming_frame.offset,
                outgoing_end=incoming_frame.offset,
            )

        if decision.kind is TemporalSplitKind.CROSSFADE:
            if incoming_frame.offset >= outgoing_frame.offset:
                raise SplitAlignmentError(
                    "aligned crossfade boundaries must preserve incoming < outgoing"
                )

            return SplitDecision(
                kind=SplitKind.CROSSFADE,
                incoming_start=incoming_frame.offset,
                outgoing_end=outgoing_frame.offset,
            )

        raise RuntimeError(f"unsupported temporal split kind: {decision.kind}")

    @staticmethod
    def _nearest_frame(
        frames: tuple[EncodedAudioFrame, ...],
        time_seconds: float,
    ) -> EncodedAudioFrame | None:
        first = frames[0]
        last = frames[-1]
        last_end_time = last.time_seconds + last.samples / last.sample_rate

        if time_seconds < first.time_seconds or time_seconds > last_end_time:
            return None

        return min(
            frames,
            key=lambda frame: (
                abs(frame.time_seconds - time_seconds),
                frame.time_seconds,
            ),
        )
