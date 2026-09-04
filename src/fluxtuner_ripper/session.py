"""Stateful ripping session orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from fluxtuner_ripper.metadata import MetadataSemanticTracker
from fluxtuner_ripper.models import (
    MetadataSemanticDecision,
    RippingIngestResult,
    TrackCandidate,
)
from fluxtuner_ripper.orchestrator import BoundaryResolution, RippingOrchestrator
from fluxtuner_ripper.ripping import RippingStreamIngestor


@dataclass(frozen=True)
class TrackTransition:
    """A resolved transition from the current track to an incoming track."""

    outgoing: TrackCandidate
    incoming: TrackCandidate
    boundary: BoundaryResolution


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
        orchestrator: RippingOrchestrator | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._metadata_tracker = metadata_tracker or MetadataSemanticTracker()
        self._orchestrator = orchestrator or RippingOrchestrator()
        self._current_track: TrackCandidate | None = None

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

    def feed(self, chunk: bytes) -> SessionFeedResult:
        ingest = self._ingestor.feed(chunk)

        semantic_decisions: list[MetadataSemanticDecision] = []
        confirmed_tracks: list[TrackCandidate] = []
        transitions: list[TrackTransition] = []

        for event in ingest.timed_metadata_events:
            decisions, tracks = self._metadata_tracker.feed(event)
            semantic_decisions.extend(decisions)
            confirmed_tracks.extend(tracks)

            for track in tracks:
                transition = self._handle_confirmed_track(track)
                if transition is not None:
                    transitions.append(transition)

        return SessionFeedResult(
            ingest=ingest,
            semantic_decisions=tuple(semantic_decisions),
            confirmed_tracks=tuple(confirmed_tracks),
            transitions=tuple(transitions),
        )

    def confirm_current(self) -> tuple[TrackCandidate, ...]:
        timeline = self._ingestor.timeline
        frames = timeline.frames
        if not frames:
            return ()

        frame = frames[-1]
        tracks = self._metadata_tracker.confirm_current(
            audio_offset=frame.offset + frame.length,
            audio_time_seconds=(frame.time_seconds + frame.samples / frame.sample_rate),
        )

        for track in tracks:
            self._handle_confirmed_track(track)

        return tracks
