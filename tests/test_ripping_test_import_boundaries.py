import ast
from pathlib import Path

FORBIDDEN_FROM_RIPPING = {
    "EncodedAudioRingBuffer",
    "IncrementalFrameTimeline",
    "frame_at_or_before_offset",
    "frame_nearest_time",
    "parse_adts_frames",
    "parse_mp3_frames",
    "IcyStreamParser",
    "AcousticWindowExtractor",
    "AcousticDecodeError",
    "FfmpegAcousticDecoder",
    "RmsAcousticAnalyzer",
    "AcousticCandidateFinder",
    "AcousticWindow",
    "DecodedPcm",
    "AcousticProfile",
    "AcousticLevel",
    "AcousticBoundaryCandidate",
    "ContentKind",
    "MetadataEvent",
    "TimedMetadataEvent",
    "TrackCandidate",
    "MetadataSemanticTracker",
    "SplitKind",
    "NearestBoundaryMatcher",
    "BoundaryRelationClassifier",
    "TemporalSplitPolicy",
    "TemporalSplitAligner",
    "SplitAlignmentError",
    "BoundaryRelation",
    "BoundaryRelationResult",
    "TemporalSplitDecision",
    "TemporalSplitKind",
    "SplitDecision",
    "SegmentRangePlanner",
    "EncodedSegmentWriter",
    "SegmentFileWriter",
    "Mp3SegmentFinalizer",
    "AacSegmentFinalizer",
    "SegmentFinalizeError",
    "SegmentOutputService",
}


def test_ripping_tests_import_low_level_components_from_owner_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "tests/test_ripping.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    violations: list[str] = []

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "fluxtuner_ripper.integrations.radio.ripping"
        ):
            bad = [alias.name for alias in node.names if alias.name in FORBIDDEN_FROM_RIPPING]
            if bad:
                violations.append(f"line {node.lineno}: {', '.join(sorted(bad))}")

    assert not violations, (
        "tests/test_ripping.py still imports low-level symbols "
        "through ripping.py:\n" + "\n".join(violations)
    )
