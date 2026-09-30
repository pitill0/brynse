from __future__ import annotations

from brynse.incoming_refinement import IncomingBoundaryRefiner
from brynse.models import TemporalSplitDecision, TemporalSplitKind


def _hard_cut(time_seconds: float = 100.0) -> TemporalSplitDecision:
    return TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        incoming_start_seconds=time_seconds,
        outgoing_end_seconds=time_seconds,
    )


def test_incoming_refiner_preserves_hard_cut_without_later_boundary() -> None:
    decision = _hard_cut()

    refined = IncomingBoundaryRefiner().refine(
        decision=decision,
        incoming_time_seconds=None,
    )

    assert refined == decision


def test_incoming_refiner_promotes_later_incoming_to_exclusion() -> None:
    decision = _hard_cut()

    refined = IncomingBoundaryRefiner().refine(
        decision=decision,
        incoming_time_seconds=102.25,
    )

    assert refined.kind is TemporalSplitKind.EXCLUSION
    assert refined.outgoing_end_seconds == 100.0
    assert refined.incoming_start_seconds == 102.25


def test_incoming_refiner_does_not_move_outgoing_boundary() -> None:
    decision = _hard_cut(345.217)

    refined = IncomingBoundaryRefiner().refine(
        decision=decision,
        incoming_time_seconds=347.019,
    )

    assert refined.outgoing_end_seconds == decision.outgoing_end_seconds


def test_incoming_refiner_ignores_boundary_at_or_before_outgoing() -> None:
    decision = _hard_cut()

    for incoming_time_seconds in (100.0, 99.5):
        refined = IncomingBoundaryRefiner().refine(
            decision=decision,
            incoming_time_seconds=incoming_time_seconds,
        )

        assert refined == decision


def test_incoming_refiner_does_not_modify_non_hard_cut_decision() -> None:
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=102.0,
    )

    refined = IncomingBoundaryRefiner().refine(
        decision=decision,
        incoming_time_seconds=104.0,
    )

    assert refined == decision


from types import SimpleNamespace

from brynse.boundaries import BoundaryConfidence
from brynse.incoming_refinement import IncomingBoundaryDiscovery


def _analysis(
    time_seconds: float,
    confidence: BoundaryConfidence,
):
    return SimpleNamespace(
        hypothesis=SimpleNamespace(time_seconds=time_seconds),
        assessment=SimpleNamespace(candidate_confidence=confidence),
    )


def test_incoming_discovery_returns_none_without_later_high_boundary() -> None:
    analyses = (
        _analysis(99.0, BoundaryConfidence.HIGH),
        _analysis(101.0, BoundaryConfidence.UNRESOLVED),
    )

    discovered = IncomingBoundaryDiscovery().discover(
        outgoing_end_seconds=100.0,
        analyses=analyses,
    )

    assert discovered is None


def test_incoming_discovery_selects_first_later_high_boundary() -> None:
    analyses = (
        _analysis(104.0, BoundaryConfidence.HIGH),
        _analysis(101.5, BoundaryConfidence.HIGH),
        _analysis(102.0, BoundaryConfidence.UNRESOLVED),
    )

    discovered = IncomingBoundaryDiscovery().discover(
        outgoing_end_seconds=100.0,
        analyses=analyses,
    )

    assert discovered == 101.5


def test_incoming_discovery_never_selects_outgoing_boundary_itself() -> None:
    analyses = (
        _analysis(100.0, BoundaryConfidence.HIGH),
        _analysis(102.0, BoundaryConfidence.HIGH),
    )

    discovered = IncomingBoundaryDiscovery().discover(
        outgoing_end_seconds=100.0,
        analyses=analyses,
    )

    assert discovered == 102.0


def test_incoming_analyzer_only_analyzes_both_source_hypotheses() -> None:
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.incoming_refinement import IncomingBoundaryAnalyzer

    basin = BoundaryProposal(
        time_seconds=101.0,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=101.5,
        source=BoundaryProposalSource.STRUCTURAL,
    )
    structural_only = BoundaryProposal(
        time_seconds=103.0,
        source=BoundaryProposalSource.STRUCTURAL,
    )

    both = BoundaryHypothesis(
        time_seconds=101.25,
        proposals=(basin, structural),
    )
    single = BoundaryHypothesis(
        time_seconds=103.0,
        proposals=(structural_only,),
    )

    class Analyzer:
        def __init__(self) -> None:
            self.calls = []

        def analyze(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs["hypothesis"]

    analyzer = Analyzer()
    pcm = object()

    result = IncomingBoundaryAnalyzer(
        analyzer=analyzer,
    ).analyze(
        hypotheses=(single, both),
        pcm=pcm,
        absolute_start_time_seconds=90.0,
    )

    assert result == (both,)
    assert analyzer.calls == [
        {
            "hypothesis": both,
            "pcm": pcm,
            "absolute_start_time_seconds": 90.0,
            "measure_spectral": True,
            "measure_temporal": False,
            "measure_local_discontinuity": False,
        }
    ]


def test_incoming_analyzer_ignores_hypotheses_at_or_before_outgoing_end() -> None:
    from brynse.boundaries import (
        BoundaryHypothesis,
        BoundaryProposal,
        BoundaryProposalSource,
    )
    from brynse.incoming_refinement import IncomingBoundaryAnalyzer

    def both_hypothesis(time_seconds: float) -> BoundaryHypothesis:
        return BoundaryHypothesis(
            time_seconds=time_seconds,
            proposals=(
                BoundaryProposal(
                    time_seconds=time_seconds,
                    source=BoundaryProposalSource.BASIN,
                ),
                BoundaryProposal(
                    time_seconds=time_seconds,
                    source=BoundaryProposalSource.STRUCTURAL,
                ),
            ),
        )

    before = both_hypothesis(99.0)
    equal = both_hypothesis(100.0)
    after = both_hypothesis(102.0)

    class Analyzer:
        def __init__(self) -> None:
            self.calls = []

        def analyze(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs["hypothesis"]

    analyzer = Analyzer()

    result = IncomingBoundaryAnalyzer(
        analyzer=analyzer,
    ).analyze(
        hypotheses=(before, equal, after),
        pcm=object(),
        absolute_start_time_seconds=90.0,
        outgoing_end_seconds=100.0,
    )

    assert result == (after,)
    assert [call["hypothesis"] for call in analyzer.calls] == [after]


def test_incoming_pipeline_refines_later_high_boundary_without_moving_outgoing() -> None:
    from types import SimpleNamespace

    from brynse.incoming_refinement import IncomingBoundaryPipeline
    from brynse.models import TemporalSplitDecision, TemporalSplitKind

    original = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )

    hypothesis = SimpleNamespace(time_seconds=102.0)
    analysis = SimpleNamespace(
        hypothesis=hypothesis,
        assessment=SimpleNamespace(
            candidate_confidence=BoundaryConfidence.HIGH,
        ),
    )

    class HypothesisBuilder:
        def __init__(self):
            self.calls = []

        def build(self, **kwargs):
            self.calls.append(kwargs)
            return (hypothesis,)

    class Analyzer:
        def __init__(self):
            self.calls = []

        def analyze(self, **kwargs):
            self.calls.append(kwargs)
            return (analysis,)

    builder = HypothesisBuilder()
    analyzer = Analyzer()

    pipeline = IncomingBoundaryPipeline(
        hypothesis_builder=builder,
        incoming_analyzer=analyzer,
        discovery=IncomingBoundaryDiscovery(),
        refiner=IncomingBoundaryRefiner(),
    )

    window = SimpleNamespace(start_time_seconds=90.0)
    pcm8 = object()
    pcm16 = object()

    refined = pipeline.refine(
        decision=original,
        window=window,
        pcm8=pcm8,
        pcm16=pcm16,
    )

    assert refined == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=102.0,
    )

    # Frozen-final invariant.
    assert refined.outgoing_end_seconds == original.outgoing_end_seconds

    assert builder.calls == [
        {
            "pcm": pcm8,
            "absolute_start_time_seconds": 90.0,
        }
    ]

    assert analyzer.calls == [
        {
            "hypotheses": (hypothesis,),
            "pcm": pcm16,
            "absolute_start_time_seconds": 90.0,
            "outgoing_end_seconds": 100.0,
        }
    ]


def test_incoming_pipeline_preserves_hard_cut_without_later_high_boundary() -> None:
    from types import SimpleNamespace

    from brynse.incoming_refinement import IncomingBoundaryPipeline
    from brynse.models import TemporalSplitDecision, TemporalSplitKind

    original = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )

    hypothesis = SimpleNamespace(time_seconds=102.0)
    analysis = SimpleNamespace(
        hypothesis=hypothesis,
        assessment=SimpleNamespace(
            candidate_confidence=BoundaryConfidence.UNRESOLVED,
        ),
    )

    class HypothesisBuilder:
        def build(self, **kwargs):
            return (hypothesis,)

    class Analyzer:
        def analyze(self, **kwargs):
            return (analysis,)

    pipeline = IncomingBoundaryPipeline(
        hypothesis_builder=HypothesisBuilder(),
        incoming_analyzer=Analyzer(),
        discovery=IncomingBoundaryDiscovery(),
        refiner=IncomingBoundaryRefiner(),
    )

    result = pipeline.refine(
        decision=original,
        window=SimpleNamespace(start_time_seconds=90.0),
        pcm8=object(),
        pcm16=object(),
    )

    assert result is original


def test_build_incoming_boundary_pipeline_uses_canonical_shadow_components(
    monkeypatch,
) -> None:
    import brynse.incoming_refinement as incoming_module

    created = {}

    class Basin:
        pass

    class Structural:
        pass

    class Reconciler:
        pass

    class Spectral:
        def __init__(self, *, ffmpeg_binary):
            created["spectral_ffmpeg"] = ffmpeg_binary

    class Shadow:
        def __init__(self, *, spectral_analyzer):
            created["shadow_spectral"] = spectral_analyzer

    class Builder:
        def __init__(self, *, basin_detector, structural_detector, reconciler):
            created["basin"] = basin_detector
            created["structural"] = structural_detector
            created["reconciler"] = reconciler

    class Analyzer:
        def __init__(self, *, analyzer):
            created["shadow"] = analyzer

    monkeypatch.setattr(
        incoming_module,
        "AdaptiveBasinBoundaryDetector",
        Basin,
        raising=False,
    )
    monkeypatch.setattr(
        incoming_module,
        "StructuralBoundaryDetector",
        Structural,
        raising=False,
    )
    monkeypatch.setattr(
        incoming_module,
        "BoundaryReconciler",
        Reconciler,
        raising=False,
    )
    monkeypatch.setattr(
        incoming_module,
        "FfmpegSpectralDistributionAnalyzer",
        Spectral,
        raising=False,
    )
    monkeypatch.setattr(
        incoming_module,
        "ShadowBoundaryAnalyzer",
        Shadow,
        raising=False,
    )
    monkeypatch.setattr(
        incoming_module,
        "AcousticBoundaryHypothesisBuilder",
        Builder,
        raising=False,
    )
    monkeypatch.setattr(
        incoming_module,
        "IncomingBoundaryAnalyzer",
        Analyzer,
    )

    pipeline = incoming_module.build_incoming_boundary_pipeline(
        ffmpeg_binary="/custom/ffmpeg",
    )

    assert isinstance(pipeline, incoming_module.IncomingBoundaryPipeline)

    assert isinstance(created["basin"], Basin)
    assert isinstance(created["structural"], Structural)
    assert isinstance(created["reconciler"], Reconciler)

    assert created["spectral_ffmpeg"] == "/custom/ffmpeg"
    assert isinstance(created["shadow_spectral"], Spectral)
    assert isinstance(created["shadow"], Shadow)


def test_incoming_pipeline_reports_observation_without_changing_decision() -> None:
    from types import SimpleNamespace

    from brynse.incoming_refinement import (
        IncomingBoundaryDiscovery,
        IncomingBoundaryPipeline,
        IncomingBoundaryRefiner,
    )
    from brynse.models import TemporalSplitDecision, TemporalSplitKind

    original = TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=100.0,
    )

    hypothesis = SimpleNamespace(time_seconds=102.0)
    analysis = SimpleNamespace(
        hypothesis=hypothesis,
        assessment=SimpleNamespace(
            candidate_confidence=BoundaryConfidence.HIGH,
        ),
    )

    class Builder:
        def build(self, **kwargs):
            return (hypothesis,)

    class Analyzer:
        def analyze(self, **kwargs):
            return (analysis,)

    observations = []

    pipeline = IncomingBoundaryPipeline(
        hypothesis_builder=Builder(),
        incoming_analyzer=Analyzer(),
        discovery=IncomingBoundaryDiscovery(),
        refiner=IncomingBoundaryRefiner(),
        on_observation=observations.append,
    )

    result = pipeline.refine(
        decision=original,
        window=SimpleNamespace(start_time_seconds=90.0),
        pcm8=object(),
        pcm16=object(),
    )

    assert result == TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=100.0,
        incoming_start_seconds=102.0,
    )

    assert len(observations) == 1

    observation = observations[0]
    assert observation.outgoing_end_seconds == 100.0
    assert observation.hypotheses == (hypothesis,)
    assert observation.analyses == (analysis,)
    assert observation.selected_time_seconds == 102.0
    assert observation.decision == result
