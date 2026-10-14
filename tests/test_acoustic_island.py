from types import SimpleNamespace

import pytest

from brynse.acoustic_island import AcousticIslandDetector, AcousticIslandEvidence
from brynse.boundaries import BoundaryProposalSource
from brynse.transition_evidence import FadeRoutingEvidence, IncomingFadeEvidence, OutgoingFadeEvidence


def _proposal(source):
    return SimpleNamespace(source=source)


def _supported_handoff(*, time_seconds: float, basin_start: float, basin_recovery: float):
    return SimpleNamespace(
        time_seconds=time_seconds,
        proposals=(
            _proposal(BoundaryProposalSource.BASIN),
            _proposal(BoundaryProposalSource.STRUCTURAL),
        ),
        basin_geometry=SimpleNamespace(
            start_seconds=basin_start,
            minimum_seconds=time_seconds,
            recovery_seconds=basin_recovery,
        ),
    )


def _region(*, reference: float, left, right, inside_left: bool = False, inside_right: bool = False):
    return SimpleNamespace(
        reference_time_seconds=reference,
        left=left,
        right=right,
        reference_inside_left_basin=inside_left,
        reference_inside_right_basin=inside_right,
    )


def _outgoing(*, start: float, floor: float):
    return OutgoingFadeEvidence(
        start_time_seconds=start,
        floor_time_seconds=floor,
        rise_time_seconds=None,
        start_rms=4000.0,
        floor_rms=100.0,
        descent_ratio=40.0,
        descent_consistency=0.8,
    )


def _incoming(*, start: float, established: float, ascent_ratio: float, persistence_ratio: float):
    return IncomingFadeEvidence(
        start_time_seconds=start,
        established_time_seconds=established,
        start_rms=100.0,
        established_rms=100.0 * ascent_ratio,
        ascent_ratio=ascent_ratio,
        ascent_consistency=0.6,
        persistence_ratio=persistence_ratio,
    )


def test_evidence_never_exposes_temporal_decision_fields() -> None:
    right = _supported_handoff(time_seconds=110.0, basin_start=109.5, basin_recovery=110.5)
    evidence = AcousticIslandEvidence(incoming_handoff=right, observed_region=SimpleNamespace())
    assert not hasattr(evidence, 'outgoing_end_seconds')
    assert not hasattr(evidence, 'incoming_start_seconds')
    assert not hasattr(evidence, 'kind')


def test_evidence_requires_region_or_complete_reset_pair() -> None:
    right = _supported_handoff(time_seconds=110.0, basin_start=109.5, basin_recovery=110.5)
    with pytest.raises(ValueError):
        AcousticIslandEvidence(incoming_handoff=right)
    with pytest.raises(ValueError):
        AcousticIslandEvidence(
            incoming_handoff=right,
            terminal_decay=_outgoing(start=100.0, floor=104.0),
        )


def test_detector_observes_unique_bracketing_region() -> None:
    left = _supported_handoff(time_seconds=100.0, basin_start=99.5, basin_recovery=100.5)
    right = _supported_handoff(time_seconds=108.0, basin_start=107.5, basin_recovery=108.5)
    region = _region(reference=104.0, left=left, right=right)
    result = AcousticIslandDetector().observe(
        reference_time_seconds=104.0,
        fade=FadeRoutingEvidence(),
        hypotheses=(left, right),
        bounded_insert_regions=(region,),
    )
    assert len(result) == 1
    evidence = result[0]
    assert evidence.observed_region is region
    assert evidence.incoming_handoff is right
    assert evidence.terminal_decay is None
    assert evidence.source_reset is None


def test_detector_observes_terminal_reset_without_region() -> None:
    right = _supported_handoff(time_seconds=1105.279, basin_start=1105.0, basin_recovery=1105.8)
    outgoing = _outgoing(start=1093.0, floor=1098.879)
    reset = _incoming(start=1098.879, established=1099.379, ascent_ratio=22.0, persistence_ratio=14.0)
    result = AcousticIslandDetector().observe(
        reference_time_seconds=1104.536,
        fade=FadeRoutingEvidence(outgoing_fades=(outgoing,), incoming_fades=(reset,)),
        hypotheses=(right,),
    )
    assert len(result) == 1
    evidence = result[0]
    assert evidence.terminal_decay is outgoing
    assert evidence.source_reset is reset
    assert evidence.incoming_handoff is right
    assert evidence.observed_region is None


def test_detector_refines_region_with_earlier_terminal_reset() -> None:
    left = _supported_handoff(time_seconds=2154.703, basin_start=2154.2, basin_recovery=2155.1)
    right = _supported_handoff(time_seconds=2171.378, basin_start=2170.8, basin_recovery=2172.0)
    region = _region(reference=2155.860, left=left, right=right)
    outgoing = _outgoing(start=2147.253, floor=2155.303)
    reset = _incoming(start=2151.453, established=2151.953, ascent_ratio=12.798, persistence_ratio=5.411)
    result = AcousticIslandDetector().observe(
        reference_time_seconds=2155.860,
        fade=FadeRoutingEvidence(outgoing_fades=(outgoing,), incoming_fades=(reset,)),
        hypotheses=(left, right),
        bounded_insert_regions=(region,),
    )
    assert len(result) == 1
    evidence = result[0]
    assert evidence.observed_region is region
    assert evidence.terminal_decay is outgoing
    assert evidence.source_reset is reset
    assert evidence.incoming_handoff is right


def test_detector_does_not_promote_weak_reset() -> None:
    right = _supported_handoff(time_seconds=110.0, basin_start=109.5, basin_recovery=110.5)
    result = AcousticIslandDetector().observe(
        reference_time_seconds=108.0,
        fade=FadeRoutingEvidence(
            outgoing_fades=(_outgoing(start=100.0, floor=106.0),),
            incoming_fades=(_incoming(start=105.0, established=105.5, ascent_ratio=3.0, persistence_ratio=8.0),),
        ),
        hypotheses=(right,),
    )
    assert result == ()


def test_detector_rejects_early_reset_when_later_owned_reset_explains_handoff() -> None:
    owner = _supported_handoff(time_seconds=105.0, basin_start=104.5, basin_recovery=105.5)
    right = _supported_handoff(time_seconds=112.0, basin_start=111.5, basin_recovery=112.5)
    outgoing = _outgoing(start=100.0, floor=106.0)
    early_unowned = _incoming(start=103.0, established=103.5, ascent_ratio=5.0, persistence_ratio=5.0)
    later_owned = _incoming(start=105.0, established=105.4, ascent_ratio=6.0, persistence_ratio=6.0)
    result = AcousticIslandDetector().observe(
        reference_time_seconds=108.0,
        fade=FadeRoutingEvidence(outgoing_fades=(outgoing,), incoming_fades=(early_unowned, later_owned)),
        hypotheses=(owner, right),
    )
    assert result == ()


def test_detector_rejects_non_bracketing_region() -> None:
    left = _supported_handoff(time_seconds=105.0, basin_start=104.5, basin_recovery=105.5)
    right = _supported_handoff(time_seconds=110.0, basin_start=109.5, basin_recovery=110.5)
    region = _region(reference=100.0, left=left, right=right)
    result = AcousticIslandDetector().observe(
        reference_time_seconds=100.0,
        fade=FadeRoutingEvidence(),
        hypotheses=(left, right),
        bounded_insert_regions=(region,),
    )
    assert result == ()



def test_detector_observes_long_terminal_reset_chain() -> None:
    handoff = _supported_handoff(
        time_seconds=529.947,
        basin_start=528.972,
        basin_recovery=530.372,
    )
    outgoing = _outgoing(
        start=504.422,
        floor=505.572,
    )
    reset = _incoming(
        start=512.472,
        established=517.522,
        ascent_ratio=50.42,
        persistence_ratio=34.10,
    )

    result = AcousticIslandDetector().observe(
        reference_time_seconds=528.379,
        fade=FadeRoutingEvidence(
            outgoing_fades=(outgoing,),
            incoming_fades=(reset,),
        ),
        hypotheses=(handoff,),
    )

    assert len(result) == 1

    evidence = result[0]

    assert evidence.observed_region is None
    assert evidence.terminal_decay is outgoing
    assert evidence.source_reset is reset
    assert evidence.incoming_handoff is handoff


def test_detector_does_not_promote_short_reset_chain() -> None:
    handoff = _supported_handoff(
        time_seconds=825.995,
        basin_start=825.5,
        basin_recovery=826.5,
    )
    outgoing = _outgoing(
        start=810.0,
        floor=812.920,
    )
    reset = _incoming(
        start=818.270,
        established=819.0,
        ascent_ratio=32.63,
        persistence_ratio=17.33,
    )

    result = AcousticIslandDetector().observe(
        reference_time_seconds=827.376,
        fade=FadeRoutingEvidence(
            outgoing_fades=(outgoing,),
            incoming_fades=(reset,),
        ),
        hypotheses=(handoff,),
    )

    assert result == ()


def test_detector_does_not_promote_long_chain_from_weak_decay() -> None:
    handoff = _supported_handoff(
        time_seconds=1799.799,
        basin_start=1799.2,
        basin_recovery=1800.2,
    )

    outgoing = OutgoingFadeEvidence(
        start_time_seconds=1777.149,
        floor_time_seconds=1777.799,
        rise_time_seconds=None,
        start_rms=6399.92,
        floor_rms=3796.58,
        descent_ratio=1.69,
        descent_consistency=0.92,
    )

    reset = _incoming(
        start=1781.799,
        established=1781.849,
        ascent_ratio=8.04,
        persistence_ratio=5.05,
    )

    result = AcousticIslandDetector().observe(
        reference_time_seconds=1800.856,
        fade=FadeRoutingEvidence(
            outgoing_fades=(outgoing,),
            incoming_fades=(reset,),
        ),
        hypotheses=(handoff,),
    )

    assert result == ()
