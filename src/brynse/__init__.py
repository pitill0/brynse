"""Public API for Brynse."""

from brynse.acoustic import (
    AcousticCandidateFinder,
    AcousticDecodeError,
    AcousticWindowExtractor,
    FfmpegAcousticDecoder,
    RmsAcousticAnalyzer,
)
from brynse.basin import AdaptiveBasinBoundaryDetector, Basin
from brynse.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
    BoundaryReconciler,
)
from brynse.buffer import EncodedAudioRingBuffer
from brynse.frames import (
    IncrementalFrameTimeline,
    frame_at_or_before_offset,
    frame_nearest_time,
    parse_adts_frames,
    parse_mp3_frames,
)
from brynse.generic_runner import (
    GenericRunner,
    GenericRunResult,
)
from brynse.ingest import EncodedStreamIngestor
from brynse.matching import (
    BoundaryRelationClassifier,
    NearestBoundaryMatcher,
    SplitAlignmentError,
    TemporalSplitAligner,
    TemporalSplitPolicy,
)
from brynse.models import (
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
from brynse.orchestrator import (
    CandidateResolution,
    CandidateResolver,
)
from brynse.output import (
    AacSegmentFinalizer,
    EncodedSegmentWriter,
    Mp3SegmentFinalizer,
    SegmentFileWriteError,
    SegmentFileWriter,
    SegmentFinalizeError,
    SegmentOutputService,
    SegmentRangePlanner,
)
from brynse.providers import (
    BoundaryProvider,
    FixedIntervalBoundaryProvider,
)
from brynse.streaming_runtime import (
    SafeStreamingPipeline,
    create_safe_streaming_pipeline,
)
from brynse.structural import (
    StructuralBoundaryDetector,
    StructuralFeatureFrame,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "CandidateResolution",
    "CandidateResolver",
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
    "IncrementalFrameTimeline",
    "Mp3SegmentFinalizer",
    "NearestBoundaryMatcher",
    "BoundaryProvider",
    "FixedIntervalBoundaryProvider",
    "Segment",
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
