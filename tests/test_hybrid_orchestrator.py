from __future__ import annotations

from brynse.models import (
    AcousticProfile,
    AcousticWindow,
    BoundaryCandidate,
    DecodedPcm,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)
from brynse.orchestrator import HybridCandidateResolver


class _WindowExtractor:
    def __init__(self, window: AcousticWindow | None) -> None:
        self.window = window

    def extract(self, **kwargs: object) -> AcousticWindow | None:
        return self.window


class _Decoder:
    def decode(self, window: AcousticWindow) -> DecodedPcm:
        return DecodedPcm(
            sample_rate=8000,
            channels=1,
            sample_width_bytes=2,
            data=b"\x00\x00",
        )


class _Analyzer:
    def analyze(self, pcm: DecodedPcm) -> AcousticProfile:
        return AcousticProfile(levels=())


class _Resolver:
    def __init__(self, decision: TemporalSplitDecision | None) -> None:
        self.decision = decision

    def resolve(self, **kwargs: object) -> TemporalSplitDecision | None:
        return self.decision


class _Aligner:
    def __init__(self, decision: SplitDecision) -> None:
        self.decision = decision

    def align(self, **kwargs: object) -> SplitDecision:
        return self.decision


def _candidate() -> BoundaryCandidate:
    return BoundaryCandidate(
        time_seconds=10.0,
        source="test",
    )


def _window() -> AcousticWindow:
    return AcousticWindow(
        start_offset=500,
        end_offset=2500,
        start_time_seconds=8.0,
        end_time_seconds=13.0,
        data=bytes(2000),
    )


def test_hybrid_candidate_resolver_returns_none_without_acoustic_window() -> None:
    resolver = HybridCandidateResolver(
        window_extractor=_WindowExtractor(None),
    )

    assert (
        resolver.resolve_candidate(
            candidate=_candidate(),
            timeline=object(),
            ring_buffer=object(),
        )
        is None
    )


def test_hybrid_candidate_resolver_returns_none_without_temporal_resolution() -> None:
    resolver = HybridCandidateResolver(
        window_extractor=_WindowExtractor(_window()),
        decoder=_Decoder(),
        analyzer=_Analyzer(),
        resolver=_Resolver(None),
    )

    assert (
        resolver.resolve_candidate(
            candidate=_candidate(),
            timeline=object(),
            ring_buffer=object(),
        )
        is None
    )


def test_hybrid_candidate_resolver_composes_independent_crossfade_edges() -> None:
    candidate = _candidate()
    temporal = TemporalSplitDecision(
        kind=TemporalSplitKind.CROSSFADE,
        incoming_start_seconds=9.0,
        outgoing_end_seconds=11.0,
    )
    split = SplitDecision(
        kind=SplitKind.CROSSFADE,
        incoming_start=3750,
        outgoing_end=4580,
    )

    resolver = HybridCandidateResolver(
        window_extractor=_WindowExtractor(_window()),
        decoder=_Decoder(),
        analyzer=_Analyzer(),
        resolver=_Resolver(temporal),
        split_aligner=_Aligner(split),
    )

    resolution = resolver.resolve_candidate(
        candidate=candidate,
        timeline=object(),
        ring_buffer=object(),
    )

    assert resolution is not None
    assert resolution.candidate == candidate
    assert resolution.temporal == temporal
    assert resolution.split == split
