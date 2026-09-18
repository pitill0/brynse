from __future__ import annotations

from fluxtuner_ripper.acoustic import (
    AcousticCandidateFinder as AcousticCandidateFinder,
)
from fluxtuner_ripper.acoustic import (
    AcousticDecodeError as AcousticDecodeError,
)
from fluxtuner_ripper.acoustic import (
    AcousticWindowExtractor as AcousticWindowExtractor,
)
from fluxtuner_ripper.acoustic import (
    FfmpegAcousticDecoder as FfmpegAcousticDecoder,
)
from fluxtuner_ripper.acoustic import (
    RmsAcousticAnalyzer as RmsAcousticAnalyzer,
)

# Compatibility re-exports preserved during staged extraction.
from fluxtuner_ripper.buffer import EncodedAudioRingBuffer as EncodedAudioRingBuffer
from fluxtuner_ripper.frames import (
    IncrementalFrameTimeline as IncrementalFrameTimeline,
)
from fluxtuner_ripper.frames import (
    frame_at_or_before_offset as frame_at_or_before_offset,
)
from fluxtuner_ripper.frames import (
    frame_nearest_time as frame_nearest_time,
)
from fluxtuner_ripper.frames import (
    parse_adts_frames as parse_adts_frames,
)
from fluxtuner_ripper.frames import (
    parse_mp3_frames as parse_mp3_frames,
)
from fluxtuner_ripper.icy import IcyStreamParser as IcyStreamParser
from fluxtuner_ripper.matching import (
    BoundaryRelationClassifier as BoundaryRelationClassifier,
)
from fluxtuner_ripper.matching import (
    NearestBoundaryMatcher as NearestBoundaryMatcher,
)
from fluxtuner_ripper.matching import (
    SplitAlignmentError as SplitAlignmentError,
)
from fluxtuner_ripper.matching import (
    TemporalSplitAligner as TemporalSplitAligner,
)
from fluxtuner_ripper.matching import (
    TemporalSplitPolicy as TemporalSplitPolicy,
)
from fluxtuner_ripper.metadata import (
    MetadataSemanticTracker as MetadataSemanticTracker,
)
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate as AcousticBoundaryCandidate,
)
from fluxtuner_ripper.models import (
    AcousticLevel as AcousticLevel,
)
from fluxtuner_ripper.models import (
    AcousticProfile as AcousticProfile,
)
from fluxtuner_ripper.models import (
    AcousticWindow as AcousticWindow,
)
from fluxtuner_ripper.models import (
    BoundaryRelation as BoundaryRelation,
)
from fluxtuner_ripper.models import (
    BoundaryRelationResult as BoundaryRelationResult,
)
from fluxtuner_ripper.models import (
    DecodedPcm as DecodedPcm,
)
from fluxtuner_ripper.models import (
    EncodedAudioFrame as EncodedAudioFrame,
)
from fluxtuner_ripper.models import (
    SplitDecision as SplitDecision,
)
from fluxtuner_ripper.models import (
    SplitKind as SplitKind,
)
from fluxtuner_ripper.models import (
    TemporalSplitDecision as TemporalSplitDecision,
)
from fluxtuner_ripper.models import (
    TemporalSplitKind as TemporalSplitKind,
)
from fluxtuner_ripper.output import (
    AacSegmentFinalizer as AacSegmentFinalizer,
)
from fluxtuner_ripper.output import (
    EncodedSegmentWriter as EncodedSegmentWriter,
)
from fluxtuner_ripper.output import (
    Mp3SegmentFinalizer as Mp3SegmentFinalizer,
)
from fluxtuner_ripper.output import (
    SegmentFileWriteError as SegmentFileWriteError,
)
from fluxtuner_ripper.output import (
    SegmentFileWriter as SegmentFileWriter,
)
from fluxtuner_ripper.output import (
    SegmentFinalizeError as SegmentFinalizeError,
)
from fluxtuner_ripper.output import (
    SegmentOutputService as SegmentOutputService,
)
from fluxtuner_ripper.output import (
    SegmentRangePlanner as SegmentRangePlanner,
)
from fluxtuner_ripper.radio_models import BoundaryMatch as BoundaryMatch
from fluxtuner_ripper.radio_models import ContentKind as ContentKind
from fluxtuner_ripper.radio_models import IcyParseResult as IcyParseResult
from fluxtuner_ripper.radio_models import MetadataEvent as MetadataEvent
from fluxtuner_ripper.radio_models import MetadataSemanticDecision as MetadataSemanticDecision
from fluxtuner_ripper.radio_models import RippingIngestResult as RippingIngestResult
from fluxtuner_ripper.radio_models import TimedMetadataEvent as TimedMetadataEvent
from fluxtuner_ripper.radio_models import TrackCandidate as TrackCandidate


class RippingStreamIngestor:
    """Join ICY parsing, encoded buffering, and exact frame-timeline resolution."""

    def __init__(self, *, metaint: int, codec: str, ring_max_bytes: int) -> None:
        self._icy = IcyStreamParser(metaint)
        self._ring = EncodedAudioRingBuffer(ring_max_bytes)
        self._timeline = IncrementalFrameTimeline(codec)
        self._pending_metadata: list[MetadataEvent] = []

    @property
    def ring_buffer(self) -> EncodedAudioRingBuffer:
        return self._ring

    @property
    def timeline(self) -> IncrementalFrameTimeline:
        return self._timeline

    def feed(self, chunk: bytes) -> RippingIngestResult:
        """Consume one raw ICY chunk and resolve new metadata onto audio time."""

        parsed = self._icy.feed(chunk)

        if parsed.audio:
            self._ring.append(parsed.audio)
            self._timeline.feed(parsed.audio)

        self._pending_metadata.extend(parsed.events)

        timed: list[TimedMetadataEvent] = []
        still_pending: list[MetadataEvent] = []
        for event in self._pending_metadata:
            frame = self._timeline.frame_for_audio_offset(event.audio_offset)
            if frame is None:
                still_pending.append(event)
                continue

            timed.append(
                TimedMetadataEvent(
                    title=event.title,
                    audio_offset=event.audio_offset,
                    audio_time_seconds=frame.time_seconds,
                )
            )

        self._pending_metadata = still_pending
        self._timeline.discard_before(self._ring.start_offset)

        return RippingIngestResult(
            audio=parsed.audio,
            metadata_events=parsed.events,
            timed_metadata_events=tuple(timed),
        )
