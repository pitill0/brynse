"""Radio-domain models built on the source-agnostic core models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from brynse.models import (
    AcousticBoundaryCandidate,
    BoundaryCandidate,
    Segment,
    SplitKind,
)


class ContentKind(StrEnum):
    """Semantic content classification for one stream interval."""

    MUSIC = "music"
    ADVERTISEMENT = "advertisement"
    JINGLE = "jingle"
    STATION_ID = "station_id"
    TALK = "talk"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MetadataEvent:
    """One metadata observation anchored to an absolute encoded-byte offset."""

    title: str
    audio_offset: int

    def __post_init__(self) -> None:
        if self.audio_offset < 0:
            raise ValueError("audio_offset must be non-negative")


@dataclass(frozen=True)
class TimedMetadataEvent:
    """ICY metadata event resolved onto the encoded-audio frame timeline."""

    title: str
    audio_offset: int
    audio_time_seconds: float


@dataclass(frozen=True)
class TrackCandidate:
    """Durable metadata title eligible for later acoustic boundary matching."""

    title: str
    start_offset: int
    start_time_seconds: float
    confirmed_at_offset: int
    confirmed_at_time_seconds: float

    def as_boundary_candidate(self) -> BoundaryCandidate:
        """Project this radio-specific track candidate onto the generic boundary contract."""
        return BoundaryCandidate(
            time_seconds=self.start_time_seconds,
            source="metadata",
            reference_offset=self.start_offset,
        )

    def as_segment(self) -> Segment:
        """Project this radio-specific track candidate onto the generic segment contract."""
        return Segment(
            start_offset=self.start_offset,
            start_time_seconds=self.start_time_seconds,
            label=self.title,
        )


@dataclass(frozen=True)
class MetadataSemanticDecision:
    """Semantic interpretation of one metadata title transition."""

    title: str
    kind: SplitKind
    start_offset: int
    start_time_seconds: float
    lifetime_seconds: float


@dataclass(frozen=True)
class BoundaryMatch:
    """One acoustic candidate selected for a semantic track transition."""

    track: TrackCandidate
    acoustic: AcousticBoundaryCandidate
    delta_seconds: float

    def __post_init__(self) -> None:
        if self.delta_seconds < 0:
            raise ValueError("delta_seconds must be non-negative")


@dataclass(frozen=True)
class IcyParseResult:
    """Audio bytes and metadata events produced from one incremental feed."""

    audio: bytes
    events: tuple[MetadataEvent, ...]


@dataclass(frozen=True)
class RippingIngestResult:
    """Result of feeding one raw ICY chunk into the ripping ingest pipeline."""

    audio: bytes
    metadata_events: tuple[MetadataEvent, ...]
    timed_metadata_events: tuple[TimedMetadataEvent, ...]
