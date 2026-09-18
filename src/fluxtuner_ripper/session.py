"""Stateful ripping session orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fluxtuner_ripper.matching import TemporalSplitAligner
from fluxtuner_ripper.metadata import MetadataSemanticTracker
from fluxtuner_ripper.models import (
    MetadataSemanticDecision,
    RippingIngestResult,
    Segment,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
    TrackCandidate,
)
from fluxtuner_ripper.radio_orchestrator import (
    BoundaryResolution,
    BoundaryResolver,
    RippingOrchestrator,
)
from fluxtuner_ripper.ripping import RippingStreamIngestor


@dataclass(frozen=True)
class SegmentTransition:
    """A resolved transition between two source-agnostic logical segments."""

    outgoing: Segment
    incoming: Segment
    boundary: BoundaryResolution


@dataclass(frozen=True)
class TrackTransition:
    """A resolved transition from the current track to an incoming track."""

    outgoing: TrackCandidate
    incoming: TrackCandidate
    boundary: BoundaryResolution

    def as_segment_transition(self) -> SegmentTransition:
        """Project this radio-specific transition onto the generic segment contract."""
        return SegmentTransition(
            outgoing=self.outgoing.as_segment(),
            incoming=self.incoming.as_segment(),
            boundary=self.boundary,
        )


@dataclass(frozen=True)
class SessionFeedResult:
    """Observable result of feeding one encoded stream chunk."""

    ingest: RippingIngestResult
    semantic_decisions: tuple[MetadataSemanticDecision, ...]
    confirmed_tracks: tuple[TrackCandidate, ...]
    transitions: tuple[TrackTransition, ...]


class RippingSession:
    """Maintain semantic track state while resolving confirmed boundaries."""

    def __init__(
        self,
        *,
        ingestor: RippingStreamIngestor,
        metadata_tracker: MetadataSemanticTracker | None = None,
        orchestrator: BoundaryResolver | None = None,
        transient_exclusion_policy: Callable[[MetadataSemanticDecision], bool] | None = None,
        split_aligner: TemporalSplitAligner | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._metadata_tracker = metadata_tracker or MetadataSemanticTracker()
        self._orchestrator = orchestrator or RippingOrchestrator()
        self._transient_exclusion_policy = transient_exclusion_policy
        self._split_aligner = split_aligner or TemporalSplitAligner()
        self._current_track: TrackCandidate | None = None
        self._pending_exclusion: MetadataSemanticDecision | None = None

    @property
    def current_track(self) -> TrackCandidate | None:
        return self._current_track

    def _handle_confirmed_track(
        self,
        track: TrackCandidate,
    ) -> TrackTransition | None:
        previous = self._current_track
        if previous is None:
            self._current_track = track
            return None

        boundary = self._resolve_pending_exclusion(track)
        if boundary is None:
            boundary = self._orchestrator.resolve_boundary(
                track=track,
                timeline=self._ingestor.timeline,
                ring_buffer=self._ingestor.ring_buffer,
            )
        if boundary is None:
            return None

        self._current_track = track
        return TrackTransition(
            outgoing=previous,
            incoming=track,
            boundary=boundary,
        )

    def _resolve_pending_exclusion(
        self,
        track: TrackCandidate,
    ) -> BoundaryResolution | None:
        pending = self._pending_exclusion
        if pending is None:
            return None

        self._pending_exclusion = None

        exclusion_end = pending.start_time_seconds + pending.lifetime_seconds
        if abs(exclusion_end - track.start_time_seconds) > 1e-6:
            return None

        temporal = TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=pending.start_time_seconds,
            incoming_start_seconds=track.start_time_seconds,
        )
        split = self._split_aligner.align(
            decision=temporal,
            timeline=self._ingestor.timeline,
        )

        return BoundaryResolution(
            track=track,
            match=None,
            relation=None,
            temporal=temporal,
            split=split,
        )

    def _remember_exclusion_candidates(
        self,
        decisions: tuple[MetadataSemanticDecision, ...],
    ) -> None:
        policy = self._transient_exclusion_policy
        if policy is None:
            return

        for decision in decisions:
            if decision.kind is SplitKind.NO_BOUNDARY and policy(decision):
                self._pending_exclusion = decision

    def _confirm_current_tracks(
        self,
    ) -> tuple[tuple[TrackCandidate, ...], tuple[TrackTransition, ...]]:
        timeline = self._ingestor.timeline
        frames = timeline.frames
        if not frames:
            return (), ()

        frame = frames[-1]
        tracks = self._metadata_tracker.confirm_current(
            audio_offset=frame.offset + frame.length,
            audio_time_seconds=(frame.time_seconds + frame.samples / frame.sample_rate),
        )

        transitions: list[TrackTransition] = []
        for track in tracks:
            transition = self._handle_confirmed_track(track)
            if transition is not None:
                transitions.append(transition)

        return tracks, tuple(transitions)

    def feed(self, chunk: bytes) -> SessionFeedResult:
        ingest = self._ingestor.feed(chunk)

        semantic_decisions: list[MetadataSemanticDecision] = []
        confirmed_tracks: list[TrackCandidate] = []
        transitions: list[TrackTransition] = []

        for event in ingest.timed_metadata_events:
            decisions, tracks = self._metadata_tracker.feed(event)
            semantic_decisions.extend(decisions)
            confirmed_tracks.extend(tracks)
            self._remember_exclusion_candidates(decisions)

            for track in tracks:
                transition = self._handle_confirmed_track(track)
                if transition is not None:
                    transitions.append(transition)

        durable_tracks, durable_transitions = self._confirm_current_tracks()
        confirmed_tracks.extend(durable_tracks)
        transitions.extend(durable_transitions)

        return SessionFeedResult(
            ingest=ingest,
            semantic_decisions=tuple(semantic_decisions),
            confirmed_tracks=tuple(confirmed_tracks),
            transitions=tuple(transitions),
        )

    def confirm_current(self) -> tuple[TrackCandidate, ...]:
        tracks, _transitions = self._confirm_current_tracks()
        return tracks
