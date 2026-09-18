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
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.integrations.radio.icy import IcyStreamParser
from fluxtuner_ripper.integrations.radio.metadata import MetadataSemanticTracker
from fluxtuner_ripper.integrations.radio.orchestrator import (
    BoundaryResolution,
    RippingOrchestrator,
)
from fluxtuner_ripper.integrations.radio.ripping import RippingStreamIngestor
from fluxtuner_ripper.integrations.radio.runner import (
    RippingRunConfig,
    RippingRunError,
    RippingRunner,
    RippingRunResult,
)
from fluxtuner_ripper.integrations.radio.session import (
    RippingSession,
    SegmentTransition,
    SessionFeedResult,
    TrackTransition,
)
from fluxtuner_ripper.integrations.radio.session_output import (
    SessionOutputWriter,
    WrittenSegment,
    WrittenTrack,
    safe_track_stem,
)
from fluxtuner_ripper.matching import (
    BoundaryRelationClassifier,
    NearestBoundaryMatcher,
    SplitAlignmentError,
    TemporalSplitAligner,
    TemporalSplitPolicy,
)
from fluxtuner_ripper.models import (
    AcousticBoundaryCandidate,
    AcousticLevel,
    AcousticProfile,
    AcousticWindow,
    BoundaryCandidate,
    BoundaryRelation,
    BoundaryRelationResult,
    DecodedPcm,
    EncodedAudioFrame,
    Segment,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from fluxtuner_ripper.orchestrator import (
    CandidateResolution,
    CandidateResolver,
)
from fluxtuner_ripper.output import (
    AacSegmentFinalizer,
    EncodedSegmentWriter,
    Mp3SegmentFinalizer,
    SegmentFileWriteError,
    SegmentFileWriter,
    SegmentFinalizeError,
    SegmentOutputService,
    SegmentRangePlanner,
)
from fluxtuner_ripper.providers import (
    BoundaryProvider,
    FixedIntervalBoundaryProvider,
)
from fluxtuner_ripper.streaming_runtime import (
    SafeStreamingPipeline,
    create_safe_streaming_pipeline,
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
    "AacSegmentFinalizer",
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
    "BoundaryRelation",
    "BoundaryRelationClassifier",
    "BoundaryRelationResult",
    "DecodedPcm",
    "EncodedAudioFrame",
    "EncodedAudioRingBuffer",
    "EncodedStreamIngestor",
    "EncodedSegmentWriter",
    "FfmpegAcousticDecoder",
    "GenericRunner",
    "GenericRunResult",
    "IcyStreamParser",
    "IncrementalFrameTimeline",
    "MetadataSemanticTracker",
    "Mp3SegmentFinalizer",
    "NearestBoundaryMatcher",
    "BoundaryProvider",
    "FixedIntervalBoundaryProvider",
    "RippingStreamIngestor",
    "Segment",
    "RippingRunConfig",
    "RippingRunError",
    "RippingRunResult",
    "RippingRunner",
    "SafeStreamingPipeline",
    "create_safe_streaming_pipeline",
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
    "SegmentFileWriteError",
    "SegmentFileWriter",
    "SegmentFinalizeError",
    "SegmentOutputService",
    "SegmentRangePlanner",
    "frame_at_or_before_offset",
    "frame_nearest_time",
    "parse_adts_frames",
    "parse_mp3_frames",
]
