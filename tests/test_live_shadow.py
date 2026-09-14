from __future__ import annotations

from dataclasses import dataclass

import pytest

from fluxtuner_ripper.boundaries import (
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
    BoundaryReconciler,
)
from fluxtuner_ripper.confidence import ShadowBoundaryAssessment
from fluxtuner_ripper.live_shadow import (
    LiveShadowBoundaryObserver,
    LiveShadowConfig,
)
from fluxtuner_ripper.models import DecodedPcm
from fluxtuner_ripper.shadow import ShadowBoundaryAnalysis


@dataclass(frozen=True)
class _Frame:
    offset: int
    length: int
    time_seconds: float
    samples: int = 8000
    sample_rate: int = 8000


class _Timeline:
    def __init__(self, count: int = 40) -> None:
        self.frames = tuple(
            _Frame(
                offset=index * 10,
                length=10,
                time_seconds=float(index),
            )
            for index in range(count)
        )


class _Ring:
    start_offset = 0
    end_offset = 400

    def read(self, start_offset: int, end_offset: int) -> bytes:
        return b"x" * (end_offset - start_offset)


class _Decoder:
    def __init__(self, *, sample_rate: int = 8000, seconds: int = 120) -> None:
        self.calls = 0
        self.sample_rate = sample_rate
        self.seconds = seconds

    def decode(self, window: object) -> DecodedPcm:
        self.calls += 1
        return DecodedPcm(
            sample_rate=self.sample_rate,
            channels=1,
            sample_width_bytes=2,
            data=b"\x00\x00" * (self.sample_rate * self.seconds),
        )


class _Detector:
    def __init__(self, evidence: tuple[BoundaryEvidence, ...]) -> None:
        self.evidence = evidence
        self.sample_rates: list[int] = []

    def detect_evidence(self, **kwargs: object) -> tuple[BoundaryEvidence, ...]:
        pcm = kwargs.get("pcm")
        if isinstance(pcm, DecodedPcm):
            self.sample_rates.append(pcm.sample_rate)
        return self.evidence


class _ShadowAnalyzer:
    required_preroll_seconds = 14.0
    required_postroll_seconds = 14.0

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def analyze(self, **kwargs: object) -> ShadowBoundaryAnalysis:
        self.calls.append(kwargs)
        hypothesis = kwargs["hypothesis"]
        assert isinstance(hypothesis, BoundaryHypothesis)
        return ShadowBoundaryAnalysis(
            hypothesis=hypothesis,
            spectral=None,
            assessment=ShadowBoundaryAssessment(
                hypothesis=hypothesis,
                spectral_between=None,
                candidate_confidence=BoundaryConfidence.UNRESOLVED,
            ),
        )


def _both_evidence() -> tuple[BoundaryEvidence, BoundaryEvidence]:
    basin = BoundaryProposal(
        time_seconds=20.0,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=20.5,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.5,
    )
    return (
        BoundaryEvidence(proposal=basin, basin_depth=0.04),
        BoundaryEvidence(
            proposal=structural,
            structural_novelty=0.5,
        ),
    )


def test_live_shadow_observer_reports_both_hypothesis_once() -> None:
    evidence = _both_evidence()
    decoder = _Decoder()
    analyzer = _ShadowAnalyzer()
    observer = LiveShadowBoundaryObserver(
        config=LiveShadowConfig(
            analysis_interval_seconds=1.0,
            history_seconds=90.0,
            report_dedup_seconds=2.0,
        ),
        decoder=decoder,
        temporal_decoder=_Decoder(sample_rate=16000),
        basin_detector=_Detector((evidence[0],)),
        structural_detector=_Detector((evidence[1],)),
        reconciler=BoundaryReconciler(),
        shadow_analyzer=analyzer,
    )

    first = observer.observe(timeline=_Timeline(), ring_buffer=_Ring())
    second = observer.observe(timeline=_Timeline(42), ring_buffer=_Ring())
    third = observer.observe(timeline=_Timeline(44), ring_buffer=_Ring())

    assert first == ()
    assert len(second) == 1
    assert third == ()
    assert decoder.calls == 3
    assert analyzer.calls[0]["measure_spectral"] is True
    assert analyzer.calls[0]["measure_local_discontinuity"] is True


def test_live_shadow_observer_skips_expensive_spectral_for_single_source() -> None:
    structural = BoundaryProposal(
        time_seconds=20.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.7,
    )
    analyzer = _ShadowAnalyzer()
    observer = LiveShadowBoundaryObserver(
        config=LiveShadowConfig(analysis_interval_seconds=1.0),
        decoder=_Decoder(),
        basin_detector=_Detector(()),
        structural_detector=_Detector(
            (
                BoundaryEvidence(
                    proposal=structural,
                    structural_novelty=0.7,
                ),
            )
        ),
        reconciler=BoundaryReconciler(),
        shadow_analyzer=analyzer,
    )

    result = observer.observe(timeline=_Timeline(), ring_buffer=_Ring())

    assert len(result) == 1
    assert analyzer.calls[0]["measure_spectral"] is False
    assert analyzer.calls[0]["measure_local_discontinuity"] is False


def test_live_shadow_observer_reuses_16khz_shadow_pcm_for_local_discontinuity() -> None:
    evidence = _both_evidence()
    primary_decoder = _Decoder(sample_rate=8000)
    shadow_decoder = _Decoder(sample_rate=16000)
    analyzer = _ShadowAnalyzer()

    observer = LiveShadowBoundaryObserver(
        config=LiveShadowConfig(
            analysis_interval_seconds=1.0,
            history_seconds=90.0,
            report_dedup_seconds=2.0,
        ),
        decoder=primary_decoder,
        temporal_decoder=shadow_decoder,
        basin_detector=_Detector((evidence[0],)),
        structural_detector=_Detector((evidence[1],)),
        reconciler=BoundaryReconciler(),
        shadow_analyzer=analyzer,
    )

    result = observer.observe(timeline=_Timeline(42), ring_buffer=_Ring())

    assert len(result) == 1
    assert primary_decoder.calls == 1
    assert shadow_decoder.calls == 1

    call = analyzer.calls[0]
    local_pcm = call["local_discontinuity_pcm"]
    assert isinstance(local_pcm, DecodedPcm)
    assert local_pcm.sample_rate == 16000
    assert call["measure_local_discontinuity"] is True
    assert call["measure_temporal"] is False


def test_live_shadow_observer_respects_analysis_interval() -> None:
    evidence = _both_evidence()
    decoder = _Decoder()
    observer = LiveShadowBoundaryObserver(
        config=LiveShadowConfig(analysis_interval_seconds=15.0),
        decoder=decoder,
        basin_detector=_Detector((evidence[0],)),
        structural_detector=_Detector((evidence[1],)),
        reconciler=BoundaryReconciler(),
        shadow_analyzer=_ShadowAnalyzer(),
    )

    observer.observe(timeline=_Timeline(40), ring_buffer=_Ring())
    observer.observe(timeline=_Timeline(45), ring_buffer=_Ring())

    assert decoder.calls == 1


def test_live_shadow_observer_forwards_custom_ffmpeg_to_spectral_analyzer(
    monkeypatch,
) -> None:
    import fluxtuner_ripper.live_shadow as live_shadow

    captured: list[str] = []

    class FakeSpectralAnalyzer:
        required_preroll_seconds = 14.0
        required_postroll_seconds = 14.0

        def __init__(self, *, ffmpeg_binary: str) -> None:
            captured.append(ffmpeg_binary)

        def analyze(self, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        live_shadow,
        "FfmpegSpectralDistributionAnalyzer",
        FakeSpectralAnalyzer,
    )

    LiveShadowBoundaryObserver(ffmpeg_binary="/custom/ffmpeg")

    assert captured == ["/custom/ffmpeg"]


def test_live_shadow_config_defaults_to_safe_history_window() -> None:
    assert LiveShadowConfig().history_seconds == 120.0


def test_live_shadow_observer_rejects_history_smaller_than_required_context() -> None:
    class LongContextAnalyzer(_ShadowAnalyzer):
        required_preroll_seconds = 42.0
        required_postroll_seconds = 42.0

    with pytest.raises(ValueError, match="history_seconds"):
        LiveShadowBoundaryObserver(
            config=LiveShadowConfig(
                analysis_interval_seconds=15.0,
                history_seconds=104.0,
                stability_margin_seconds=6.0,
            ),
            shadow_analyzer=LongContextAnalyzer(),
        )


def test_live_shadow_observer_accepts_history_covering_required_context() -> None:
    class LongContextAnalyzer(_ShadowAnalyzer):
        required_preroll_seconds = 42.0
        required_postroll_seconds = 42.0

    observer = LiveShadowBoundaryObserver(
        config=LiveShadowConfig(
            analysis_interval_seconds=15.0,
            history_seconds=105.0,
            stability_margin_seconds=6.0,
        ),
        shadow_analyzer=LongContextAnalyzer(),
    )

    assert observer._config.history_seconds == 105.0


def test_live_shadow_config_rejects_negative_stability_margin() -> None:
    import pytest

    with pytest.raises(ValueError, match="stability_margin_seconds"):
        LiveShadowConfig(stability_margin_seconds=-0.1)


def test_live_shadow_observer_waits_until_hypothesis_is_behind_finalized_horizon() -> None:
    evidence = _both_evidence()
    analyzer = _ShadowAnalyzer()
    observer = LiveShadowBoundaryObserver(
        config=LiveShadowConfig(
            analysis_interval_seconds=1.0,
            stability_margin_seconds=6.0,
        ),
        decoder=_Decoder(),
        temporal_decoder=_Decoder(sample_rate=16000),
        basin_detector=_Detector((evidence[0],)),
        structural_detector=_Detector((evidence[1],)),
        reconciler=BoundaryReconciler(),
        shadow_analyzer=analyzer,
    )

    early = observer.observe(timeline=_Timeline(40), ring_buffer=_Ring())
    mature = observer.observe(timeline=_Timeline(43), ring_buffer=_Ring())

    assert early == ()
    assert len(mature) == 1
    assert mature[0].hypothesis.time_seconds == 20.25


def test_live_shadow_observer_never_reopens_finalized_past() -> None:
    old_structural = BoundaryProposal(
        time_seconds=19.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.5,
    )
    new_structural = BoundaryProposal(
        time_seconds=23.0,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.5,
    )
    detector = _Detector(
        (
            BoundaryEvidence(
                proposal=old_structural,
                structural_novelty=0.5,
            ),
        )
    )
    observer = LiveShadowBoundaryObserver(
        config=LiveShadowConfig(
            analysis_interval_seconds=1.0,
            stability_margin_seconds=6.0,
        ),
        decoder=_Decoder(),
        basin_detector=_Detector(()),
        structural_detector=detector,
        reconciler=BoundaryReconciler(),
        shadow_analyzer=_ShadowAnalyzer(),
    )

    first = observer.observe(timeline=_Timeline(40), ring_buffer=_Ring())
    assert len(first) == 1
    assert first[0].hypothesis.time_seconds == 19.0

    detector.evidence = (
        BoundaryEvidence(
            proposal=old_structural,
            structural_novelty=0.5,
        ),
        BoundaryEvidence(
            proposal=new_structural,
            structural_novelty=0.5,
        ),
    )
    second = observer.observe(timeline=_Timeline(45), ring_buffer=_Ring())

    assert [item.hypothesis.time_seconds for item in second] == [23.0]


def test_live_shadow_observer_uses_16khz_temporal_decoder_for_default_shadow(
    monkeypatch,
) -> None:
    import fluxtuner_ripper.live_shadow as live_shadow

    rates: list[int] = []

    class FakeDecoder:
        def __init__(
            self,
            *,
            ffmpeg_binary: str,
            output_sample_rate: int = 8000,
        ) -> None:
            rates.append(output_sample_rate)

        def decode(self, window: object) -> DecodedPcm:
            raise AssertionError("decode should not be called by constructor test")

    monkeypatch.setattr(live_shadow, "FfmpegAcousticDecoder", FakeDecoder)

    LiveShadowBoundaryObserver()

    assert rates == [8000, 16000]


class _ValidatedShadowAnalyzer(_ShadowAnalyzer):
    supports_temporal_variability = True
    required_preroll_seconds = 42.0
    required_postroll_seconds = 42.0


@dataclass(frozen=True)
class _LongFrame:
    offset: int
    length: int
    time_seconds: float
    samples: int = 8000
    sample_rate: int = 8000


class _LongTimeline:
    def __init__(self, count: int = 110) -> None:
        self.frames = tuple(
            _LongFrame(
                offset=index * 10,
                length=10,
                time_seconds=float(index),
            )
            for index in range(count)
        )


class _LongRing:
    start_offset = 0
    end_offset = 2000

    def read(self, start_offset: int, end_offset: int) -> bytes:
        return b"x" * (end_offset - start_offset)


def test_live_shadow_observer_keeps_detection_at_8khz_and_shadow_evidence_at_16khz() -> None:
    basin = BoundaryProposal(
        time_seconds=60.0,
        source=BoundaryProposalSource.BASIN,
    )
    structural = BoundaryProposal(
        time_seconds=60.5,
        source=BoundaryProposalSource.STRUCTURAL,
        strength=0.5,
    )
    primary_decoder = _Decoder(sample_rate=8000)
    shadow_decoder = _Decoder(sample_rate=16000)
    basin_detector = _Detector((BoundaryEvidence(proposal=basin, basin_depth=0.04),))
    structural_detector = _Detector(
        (
            BoundaryEvidence(
                proposal=structural,
                structural_novelty=0.5,
            ),
        )
    )
    analyzer = _ValidatedShadowAnalyzer()

    observer = LiveShadowBoundaryObserver(
        config=LiveShadowConfig(
            analysis_interval_seconds=1.0,
            history_seconds=110.0,
            stability_margin_seconds=6.0,
        ),
        decoder=primary_decoder,
        temporal_decoder=shadow_decoder,
        basin_detector=basin_detector,
        structural_detector=structural_detector,
        reconciler=BoundaryReconciler(),
        shadow_analyzer=analyzer,
    )

    result = observer.observe(
        timeline=_LongTimeline(),
        ring_buffer=_LongRing(),
    )

    assert len(result) == 1
    assert primary_decoder.calls == 1
    assert shadow_decoder.calls == 1
    assert basin_detector.sample_rates == [8000]
    assert structural_detector.sample_rates == [8000]

    call = analyzer.calls[0]
    spectral_pcm = call["pcm"]
    temporal_pcm = call["temporal_pcm"]
    local_discontinuity_pcm = call["local_discontinuity_pcm"]

    assert isinstance(spectral_pcm, DecodedPcm)
    assert isinstance(temporal_pcm, DecodedPcm)
    assert isinstance(local_discontinuity_pcm, DecodedPcm)
    assert spectral_pcm.sample_rate == 16000
    assert temporal_pcm.sample_rate == 16000
    assert local_discontinuity_pcm.sample_rate == 16000
    assert spectral_pcm is temporal_pcm
    assert spectral_pcm is local_discontinuity_pcm
    assert call["measure_spectral"] is True
    assert call["measure_temporal"] is True
    assert call["measure_local_discontinuity"] is True
