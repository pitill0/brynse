from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SplitKind(StrEnum):
    """Relationship between consecutive track candidates."""

    NO_BOUNDARY = "no_boundary"
    HARD_CUT = "hard_cut"
    CROSSFADE = "crossfade"
    EXCLUSION = "exclusion"


@dataclass(frozen=True)
class SplitDecision:
    """Resolved relationship between consecutive content items.

    ``incoming_start`` and ``outgoing_end`` are absolute encoded-byte offsets.

    A hard cut uses the same offset for both boundaries. A crossfade preserves
    the shared audio interval by allowing the incoming item to start before the
    outgoing item ends. An exclusion deliberately leaves an encoded interval
    unassigned by ending the outgoing item before the incoming item starts.
    NO_BOUNDARY intentionally carries no offsets.
    """

    kind: SplitKind
    incoming_start: int | None = None
    outgoing_end: int | None = None

    def __post_init__(self) -> None:
        if self.kind is SplitKind.NO_BOUNDARY:
            if self.incoming_start is not None or self.outgoing_end is not None:
                raise ValueError("NO_BOUNDARY cannot carry boundary offsets")
            return

        if self.incoming_start is None or self.outgoing_end is None:
            raise ValueError("split decisions require both boundary offsets")
        if self.incoming_start < 0 or self.outgoing_end < 0:
            raise ValueError("boundary offsets must be non-negative")

        if self.kind is SplitKind.HARD_CUT:
            if self.incoming_start != self.outgoing_end:
                raise ValueError("HARD_CUT boundaries must be identical")
            return

        if self.kind is SplitKind.CROSSFADE:
            if self.incoming_start >= self.outgoing_end:
                raise ValueError("CROSSFADE requires incoming_start < outgoing_end")
            return

        if self.kind is SplitKind.EXCLUSION:
            if self.outgoing_end >= self.incoming_start:
                raise ValueError("EXCLUSION requires outgoing_end < incoming_start")
            return

        raise ValueError(f"unsupported split kind: {self.kind}")


@dataclass(frozen=True)
class EncodedAudioFrame:
    """One encoded audio frame placed on an exact accumulated audio timeline."""

    codec: str
    offset: int
    length: int
    sample_rate: int
    samples: int
    time_seconds: float

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("frame offset must be non-negative")
        if self.length <= 0:
            raise ValueError("frame length must be greater than zero")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.samples <= 0:
            raise ValueError("samples must be greater than zero")
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")


@dataclass(frozen=True)
class BoundaryCandidate:
    """Source-agnostic proposal for a logical boundary on the stream timeline."""

    time_seconds: float
    source: str
    reference_offset: int | None = None

    def __post_init__(self) -> None:
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.reference_offset is not None and self.reference_offset < 0:
            raise ValueError("reference_offset must be non-negative")


@dataclass(frozen=True)
class Segment:
    """Source-agnostic logical segment anchored to the retained stream."""

    start_offset: int
    start_time_seconds: float
    label: str | None = None

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if self.start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")


@dataclass(frozen=True)
class AcousticWindow:
    """Frame-aligned encoded audio window prepared for acoustic analysis."""

    start_offset: int
    end_offset: int
    start_time_seconds: float
    end_time_seconds: float
    data: bytes

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        if len(self.data) != self.end_offset - self.start_offset:
            raise ValueError("data length must match offset span")


@dataclass(frozen=True)
class AcousticLevel:
    """RMS energy measured over one PCM analysis window."""

    start_time_seconds: float
    end_time_seconds: float
    rms: float

    def __post_init__(self) -> None:
        if self.start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        if self.rms < 0:
            raise ValueError("rms must be non-negative")


@dataclass(frozen=True)
class AcousticProfile:
    """Ordered RMS measurements derived from decoded PCM."""

    levels: tuple[AcousticLevel, ...]

    def __post_init__(self) -> None:
        previous_end = -1.0
        for level in self.levels:
            if level.start_time_seconds < previous_end:
                raise ValueError("levels must be ordered and non-overlapping")
            previous_end = level.end_time_seconds

    def minimum_level(self) -> AcousticLevel | None:
        if not self.levels:
            return None
        return min(self.levels, key=lambda level: level.rms)


@dataclass(frozen=True)
class AcousticBoundaryCandidate:
    """One local RMS minimum expressed on the absolute stream timeline."""

    time_seconds: float
    rms: float
    relative_time_seconds: float

    def __post_init__(self) -> None:
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")
        if self.relative_time_seconds < 0:
            raise ValueError("relative_time_seconds must be non-negative")
        if self.rms < 0:
            raise ValueError("rms must be non-negative")


class BoundaryRelation(StrEnum):
    """Relative placement of semantic and broad acoustic boundary candidates."""

    AGREEMENT = "agreement"
    ACOUSTIC_EARLIER = "acoustic_earlier"
    SEMANTIC_EARLIER = "semantic_earlier"


@dataclass(frozen=True)
class BoundaryRelationResult:
    """Intermediate comparison between semantic and acoustic candidates."""

    relation: BoundaryRelation
    semantic_time_seconds: float
    acoustic_time_seconds: float
    signed_delta_seconds: float

    def __post_init__(self) -> None:
        if self.semantic_time_seconds < 0:
            raise ValueError("semantic_time_seconds must be non-negative")
        if self.acoustic_time_seconds < 0:
            raise ValueError("acoustic_time_seconds must be non-negative")


class TemporalSplitKind(StrEnum):
    """Split policy expressed in absolute stream time before frame alignment."""

    HARD_CUT = "hard_cut"
    CROSSFADE = "crossfade"
    EXCLUSION = "exclusion"


@dataclass(frozen=True)
class TemporalSplitDecision:
    """Intermediate split decision expressed only in absolute stream time."""

    kind: TemporalSplitKind
    incoming_start_seconds: float
    outgoing_end_seconds: float

    def __post_init__(self) -> None:
        if self.incoming_start_seconds < 0:
            raise ValueError("incoming_start_seconds must be non-negative")
        if self.outgoing_end_seconds < 0:
            raise ValueError("outgoing_end_seconds must be non-negative")

        if self.kind is TemporalSplitKind.HARD_CUT:
            if self.incoming_start_seconds != self.outgoing_end_seconds:
                raise ValueError("HARD_CUT requires identical temporal boundaries")
        elif (
            self.kind is TemporalSplitKind.CROSSFADE
            and self.incoming_start_seconds >= self.outgoing_end_seconds
        ):
            raise ValueError("CROSSFADE requires incoming_start_seconds < outgoing_end_seconds")
        elif (
            self.kind is TemporalSplitKind.EXCLUSION
            and self.outgoing_end_seconds >= self.incoming_start_seconds
        ):
            raise ValueError("EXCLUSION requires outgoing_end_seconds < incoming_start_seconds")


@dataclass(frozen=True)
class SegmentByteRange:
    """Absolute encoded-byte range to materialize for one segment."""

    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")


@dataclass(frozen=True)
class SegmentWritePlan:
    """Encoded-byte ranges for the outgoing and incoming segments."""

    outgoing: SegmentByteRange
    incoming: SegmentByteRange


@dataclass(frozen=True)
class DecodedPcm:
    """Mono signed 16-bit little-endian PCM prepared for acoustic analysis."""

    sample_rate: int
    channels: int
    sample_width_bytes: int
    data: bytes

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if self.channels != 1:
            raise ValueError("DecodedPcm must be mono")
        if self.sample_width_bytes != 2:
            raise ValueError("DecodedPcm must use 16-bit samples")
        if len(self.data) % self.sample_width_bytes != 0:
            raise ValueError("PCM byte length must align to sample width")

    @property
    def sample_count(self) -> int:
        return len(self.data) // self.sample_width_bytes
