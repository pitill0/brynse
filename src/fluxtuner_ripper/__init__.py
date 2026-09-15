"""Public API for FluxTuner Ripper."""

from fluxtuner_ripper.acoustic import (
    AcousticCandidateFinder,
    AcousticDecodeError,
    AcousticWindowExtractor,
    FfmpegAcousticDecoder,
    RmsAcousticAnalyzer,
)
from fluxtuner_ripper.basin import AdaptiveBasinBoundaryDetector, Basin
from fluxtuner_ripper.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
    BoundaryReconciler,
)
from fluxtuner_ripper.buffer import EncodedAudioRingBuffer
from fluxtuner_ripper.frames import (
    IncrementalFrameTimeline,
    frame_at_or_before_offset,
    frame_nearest_time,
    parse_adts_frames,
    parse_mp3_frames,
)
from fluxtuner_ripper.generic_runner import (
    GenericRunner,
    GenericRunResult,
)
from fluxtuner_ripper.icy import IcyStreamParser
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.matching import (
    BoundaryRelationClassifier,
    NearestBoundaryMatcher,
    SplitAlignmentError,
    TemporalSplitAligner,
    TemporalSplitPolicy,
)
from fluxtuner_ripper.metadata import MetadataSemanticTracker
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    AcousticLevel,
    AcousticProfile,
    AcousticWindow,
    BoundaryCandidate,
    BoundaryMatch,
    BoundaryRelation,
    BoundaryRelationResult,
    ContentKind,
    DecodedPcm,
    EncodedAudioFrame,
    IcyParseResult,
    MetadataEvent,
    MetadataSemanticDecision,
    RippingIngestResult,
    Segment,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
    TimedMetadataEvent,
    TrackByteRange,
    TrackCandidate,
    TrackWritePlan,
)
from fluxtuner_ripper.orchestrator import (
    BoundaryResolution,
    CandidateResolution,
    CandidateResolver,
    RippingOrchestrator,
)
from fluxtuner_ripper.output import (
    AacTrackFinalizer,
    EncodedTrackWriter,
    Mp3TrackFinalizer,
    TrackFileWriteError,
    TrackFileWriter,
    TrackFinalizeError,
    TrackOutputService,
    TrackRangePlanner,
)
from fluxtuner_ripper.providers import (
    BoundaryProvider,
    FixedIntervalBoundaryProvider,
)
from fluxtuner_ripper.ripping import RippingStreamIngestor
from fluxtuner_ripper.runner import (
    RippingRunConfig,
    RippingRunError,
    RippingRunner,
    RippingRunResult,
)
from fluxtuner_ripper.session import (
    RippingSession,
    SegmentTransition,
    SessionFeedResult,
    TrackTransition,
)
from fluxtuner_ripper.session_output import (
    SessionOutputWriter,
    WrittenSegment,
    WrittenTrack,
    safe_track_stem,
)
from fluxtuner_ripper.structural import (
    StructuralBoundaryDetector,
    StructuralFeatureFrame,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "SessionOutputWriter",
    "WrittenSegment",
    "WrittenTrack",
    "safe_track_stem",
    "RippingSession",
    "SegmentTransition",
    "SessionFeedResult",
    "TrackTransition",
    "BoundaryResolution",
    "CandidateResolution",
    "CandidateResolver",
    "RippingOrchestrator",
    "AacTrackFinalizer",
    "AcousticBoundaryCandidate",
    "AcousticCandidateFinder",
    "AcousticDecodeError",
    "AcousticLevel",
    "AcousticProfile",
    "AcousticWindow",
    "AcousticWindowExtractor",
    "AdaptiveBasinBoundaryDetector",
    "Basin",
    "BoundaryCandidate",
    "BoundaryConfidence",
    "BoundaryEvidence",
    "BoundaryHypothesis",
    "BoundaryProposal",
    "BoundaryProposalSource",
    "BoundaryReconciler",
    "BoundaryMatch",
    "BoundaryRelation",
    "BoundaryRelationClassifier",
    "BoundaryRelationResult",
    "ContentKind",
    "DecodedPcm",
    "EncodedAudioFrame",
    "EncodedAudioRingBuffer",
    "EncodedStreamIngestor",
    "EncodedTrackWriter",
    "FfmpegAcousticDecoder",
    "GenericRunner",
    "GenericRunResult",
    "IcyParseResult",
    "IcyStreamParser",
    "IncrementalFrameTimeline",
    "MetadataEvent",
    "MetadataSemanticDecision",
    "MetadataSemanticTracker",
    "Mp3TrackFinalizer",
    "NearestBoundaryMatcher",
    "BoundaryProvider",
    "FixedIntervalBoundaryProvider",
    "RippingIngestResult",
    "RippingStreamIngestor",
    "Segment",
    "RippingRunConfig",
    "RippingRunError",
    "RippingRunResult",
    "RippingRunner",
    "RmsAcousticAnalyzer",
    "SplitAlignmentError",
    "SplitDecision",
    "SplitKind",
    "StructuralBoundaryDetector",
    "StructuralFeatureFrame",
    "TemporalSplitAligner",
    "TemporalSplitDecision",
    "TemporalSplitKind",
    "TemporalSplitPolicy",
    "TimedMetadataEvent",
    "TrackByteRange",
    "TrackCandidate",
    "TrackFileWriteError",
    "TrackFileWriter",
    "TrackFinalizeError",
    "TrackOutputService",
    "TrackRangePlanner",
    "TrackWritePlan",
    "frame_at_or_before_offset",
    "frame_nearest_time",
    "parse_adts_frames",
    "parse_mp3_frames",
]
