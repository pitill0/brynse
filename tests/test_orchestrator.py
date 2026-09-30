from __future__ import annotations

from brynse.integrations.radio.models import (
    BoundaryMatch,
    TrackCandidate,
)
from brynse.integrations.radio.orchestrator import RippingOrchestrator
from brynse.models import (
    AcousticBoundaryCandidate,
    AcousticProfile,
    AcousticWindow,
    BoundaryRelation,
    BoundaryRelationResult,
    DecodedPcm,
    SplitDecision,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)


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


class _Finder:
    def __init__(self, candidate: AcousticBoundaryCandidate) -> None:
        self.candidate = candidate

    def find(self, **kwargs: object) -> tuple[AcousticBoundaryCandidate, ...]:
        return (self.candidate,)


class _Matcher:
    def __init__(self, match: BoundaryMatch | None) -> None:
        self.match_result = match

    def match(self, **kwargs: object) -> BoundaryMatch | None:
        return self.match_result

    def match_candidate(self, **kwargs: object) -> AcousticBoundaryCandidate | None:
        if self.match_result is None:
            return None
        return self.match_result.acoustic


class _Classifier:
    def __init__(self, relation: BoundaryRelationResult) -> None:
        self.relation = relation

    def classify(self, **kwargs: object) -> BoundaryRelationResult:
        return self.relation


class _Policy:
    def __init__(self, decision: TemporalSplitDecision) -> None:
        self.decision = decision

    def decide(self, relation: BoundaryRelationResult) -> TemporalSplitDecision:
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


def test_orchestrator_returns_none_without_acoustic_window() -> None:
    orchestrator = RippingOrchestrator(
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


def test_orchestrator_returns_none_without_boundary_match() -> None:
    acoustic = AcousticBoundaryCandidate(
        time_seconds=9.5,
        rms=0.1,
        relative_time_seconds=1.5,
    )
    orchestrator = RippingOrchestrator(
        window_extractor=_WindowExtractor(_window()),
        decoder=_Decoder(),
        analyzer=_Analyzer(),
        candidate_finder=_Finder(acoustic),
        matcher=_Matcher(None),
    )

    assert (
        orchestrator.resolve_boundary(
            track=_track(),
            timeline=object(),
            ring_buffer=object(),
        )
        is None
    )


def test_orchestrator_composes_full_boundary_resolution() -> None:
    track = _track()
    acoustic = AcousticBoundaryCandidate(
        time_seconds=9.0,
        rms=0.05,
        relative_time_seconds=1.0,
    )
    match = BoundaryMatch(
        track=track,
        acoustic=acoustic,
        delta_seconds=1.0,
    )
    relation = BoundaryRelationResult(
        relation=BoundaryRelation.AGREEMENT,
        semantic_time_seconds=10.0,
        acoustic_time_seconds=9.0,
        signed_delta_seconds=1.0,
    )
    temporal = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=10.0,
        outgoing_end_seconds=10.0,
    )
    split = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=4170,
        outgoing_end=4170,
    )

    orchestrator = RippingOrchestrator(
        window_extractor=_WindowExtractor(_window()),
        decoder=_Decoder(),
        analyzer=_Analyzer(),
        candidate_finder=_Finder(acoustic),
        matcher=_Matcher(match),
        relation_classifier=_Classifier(relation),
        split_policy=_Policy(temporal),
        split_aligner=_Aligner(split),
    )

    resolution = orchestrator.resolve_boundary(
        track=track,
        timeline=object(),
        ring_buffer=object(),
    )

    assert resolution is not None
    assert resolution.track == track
    assert resolution.match == match
    assert resolution.relation == relation
    assert resolution.temporal == temporal
    assert resolution.split == split


def test_multisignal_resolver_applies_incoming_refinement_before_alignment() -> None:
    from brynse.orchestrator import MultiSignalAcousticCandidateResolver

    original = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=100.0,
        outgoing_end_seconds=100.0,
    )
    refined = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        incoming_start_seconds=102.0,
        outgoing_end_seconds=100.0,
    )
    aligned = SplitDecision(
        kind=SplitKind.EXCLUSION,
        incoming_start=4200,
        outgoing_end=4000,
    )

    class WindowExtractor:
        def extract(self, **kwargs):
            return _window()

    class Decoder:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

        def decode(self, window):
            return DecodedPcm(
                sample_rate=self.sample_rate,
                channels=1,
                sample_width_bytes=2,
                data=b"\x00\x00",
            )

    class Analyzer:
        def analyze(self, pcm):
            return AcousticProfile(levels=())

    selected = object()

    class CandidateResolver:
        def resolve_acoustic_boundary(self, **kwargs):
            return selected

    class TemporalResolver:
        def resolve(self, **kwargs):
            assert kwargs["selected"] is selected
            return original

    class IncomingRefiner:
        def __init__(self):
            self.calls = []

        def refine(self, **kwargs):
            self.calls.append(kwargs)
            return refined

    class Aligner:
        def __init__(self):
            self.calls = []

        def align(self, **kwargs):
            self.calls.append(kwargs)
            return aligned

    incoming_refiner = IncomingRefiner()
    aligner = Aligner()

    resolver = MultiSignalAcousticCandidateResolver(
        window_extractor=WindowExtractor(),
        decoder8=Decoder(8000),
        decoder16=Decoder(16000),
        analyzer=Analyzer(),
        candidate_resolver=CandidateResolver(),
        temporal_resolver=TemporalResolver(),
        split_aligner=aligner,
        incoming_refiner=incoming_refiner,
    )

    candidate = _track().as_boundary_candidate()

    resolution = resolver.resolve_candidate(
        candidate=candidate,
        timeline=object(),
        ring_buffer=object(),
    )

    assert resolution is not None
    assert resolution.temporal == refined
    assert resolution.split == aligned

    assert len(incoming_refiner.calls) == 1
    call = incoming_refiner.calls[0]
    assert call["decision"] == original
    assert call["window"] == _window()
    assert call["pcm8"].sample_rate == 8000
    assert call["pcm16"].sample_rate == 16000

    assert len(aligner.calls) == 1
    assert aligner.calls[0]["decision"] == refined


def test_multisignal_resolver_preserves_hard_cut_without_incoming_refiner() -> None:
    from brynse.orchestrator import MultiSignalAcousticCandidateResolver

    original = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=100.0,
        outgoing_end_seconds=100.0,
    )
    aligned = SplitDecision(
        kind=SplitKind.HARD_CUT,
        incoming_start=4000,
        outgoing_end=4000,
    )

    class WindowExtractor:
        def extract(self, **kwargs):
            return _window()

    class Decoder:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

        def decode(self, window):
            return DecodedPcm(
                sample_rate=self.sample_rate,
                channels=1,
                sample_width_bytes=2,
                data=b"\x00\x00",
            )

    class Analyzer:
        def analyze(self, pcm):
            return AcousticProfile(levels=())

    class CandidateResolver:
        def resolve_acoustic_boundary(self, **kwargs):
            return object()

    class TemporalResolver:
        def resolve(self, **kwargs):
            return original

    class Aligner:
        def __init__(self):
            self.decisions = []

        def align(self, **kwargs):
            self.decisions.append(kwargs["decision"])
            return aligned

    aligner = Aligner()

    resolver = MultiSignalAcousticCandidateResolver(
        window_extractor=WindowExtractor(),
        decoder8=Decoder(8000),
        decoder16=Decoder(16000),
        analyzer=Analyzer(),
        candidate_resolver=CandidateResolver(),
        temporal_resolver=TemporalResolver(),
        split_aligner=aligner,
    )

    resolution = resolver.resolve_candidate(
        candidate=_track().as_boundary_candidate(),
        timeline=object(),
        ring_buffer=object(),
    )

    assert resolution is not None
    assert resolution.temporal == original
    assert resolution.split == aligned
    assert aligner.decisions == [original]
