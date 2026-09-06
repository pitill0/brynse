from __future__ import annotations

from fluxtuner_ripper.models import (
    AcousticProfile,
    AcousticWindow,
    DecodedPcm,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
    TrackCandidate,
)
from fluxtuner_ripper.orchestrator import HybridRippingOrchestrator


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


def _track() -> TrackCandidate:
    return TrackCandidate(
        title="Artist - Track",
        start_offset=1000,
        start_time_seconds=10.0,
        confirmed_at_offset=2000,
        confirmed_at_time_seconds=12.0,
    )


def _window() -> AcousticWindow:
    return AcousticWindow(
        start_offset=500,
        end_offset=2500,
        start_time_seconds=8.0,
        end_time_seconds=13.0,
        data=bytes(2000),
    )


def test_hybrid_orchestrator_returns_none_without_acoustic_window() -> None:
    orchestrator = HybridRippingOrchestrator(
        window_extractor=_WindowExtractor(None),
    )

    assert (
        orchestrator.resolve_boundary(
            track=_track(),
            timeline=object(),
            ring_buffer=object(),
        )
        is None
    )


def test_hybrid_orchestrator_returns_none_without_temporal_resolution() -> None:
    orchestrator = HybridRippingOrchestrator(
        window_extractor=_WindowExtractor(_window()),
        decoder=_Decoder(),
        analyzer=_Analyzer(),
        resolver=_Resolver(None),
    )

    assert (
        orchestrator.resolve_boundary(
            track=_track(),
            timeline=object(),
            ring_buffer=object(),
        )
        is None
    )


def test_hybrid_orchestrator_composes_independent_crossfade_edges() -> None:
    track = _track()
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

    orchestrator = HybridRippingOrchestrator(
        window_extractor=_WindowExtractor(_window()),
        decoder=_Decoder(),
        analyzer=_Analyzer(),
        resolver=_Resolver(temporal),
        split_aligner=_Aligner(split),
    )

    resolution = orchestrator.resolve_boundary(
        track=track,
        timeline=object(),
        ring_buffer=object(),
    )

    assert resolution is not None
    assert resolution.track == track
    assert resolution.match is None
    assert resolution.relation is None
    assert resolution.temporal == temporal
    assert resolution.split == split
