from __future__ import annotations

from array import array

import pytest

from fluxtuner_ripper.basin import AdaptiveBasinBoundaryDetector
from fluxtuner_ripper.boundaries import BoundaryProposalSource
from fluxtuner_ripper.models import DecodedPcm


def _pcm_from_levels(
    levels: list[int],
    *,
    sample_rate: int = 100,
    step_seconds: float = 0.1,
) -> DecodedPcm:
    samples_per_level = round(sample_rate * step_seconds)
    samples: list[int] = []
    for value in levels:
        samples.extend([value] * samples_per_level)

    return DecodedPcm(
        sample_rate=sample_rate,
        channels=1,
        sample_width_bytes=2,
        data=array("h", samples).tobytes(),
    )


def test_adaptive_basin_detector_finds_a13_style_basin() -> None:
    levels = [1000] * 30 + [100, 80, 70, 90, 120] + [500] * 8 + [1000] * 20

    basins = AdaptiveBasinBoundaryDetector().detect_basins(pcm=_pcm_from_levels(levels))

    assert len(basins) == 1
    assert basins[0].start_seconds == pytest.approx(3.0)
    assert basins[0].minimum_seconds == pytest.approx(3.25)
    assert basins[0].recovery_seconds >= 3.5
    assert basins[0].depth < 0.10


def test_adaptive_basin_detector_exposes_minimum_as_basin_proposal() -> None:
    levels = [1000] * 30 + [100, 80, 70, 90, 120] + [500] * 8 + [1000] * 20

    proposals = AdaptiveBasinBoundaryDetector().detect_proposals(pcm=_pcm_from_levels(levels))

    assert len(proposals) == 1
    assert proposals[0].time_seconds == pytest.approx(3.25)
    assert proposals[0].source is BoundaryProposalSource.BASIN
    assert proposals[0].strength is None


def test_adaptive_basin_detector_does_not_apply_old_a13_depth_gate() -> None:
    levels = [1000] * 30 + [250, 240, 230, 240, 250] + [500] * 8 + [1000] * 20

    basins = AdaptiveBasinBoundaryDetector().detect_basins(pcm=_pcm_from_levels(levels))

    assert len(basins) == 1
    assert basins[0].depth > 0.10


def test_adaptive_basin_detector_applies_absolute_start_time() -> None:
    levels = [1000] * 30 + [100, 80, 70, 90, 120] + [500] * 8 + [1000] * 20

    proposals = AdaptiveBasinBoundaryDetector().detect_proposals(
        pcm=_pcm_from_levels(levels),
        absolute_start_time_seconds=100.0,
    )

    assert proposals[0].time_seconds == pytest.approx(103.25)


def test_adaptive_basin_detector_returns_empty_without_recovery() -> None:
    levels = [1000] * 30 + [100] * 20

    assert AdaptiveBasinBoundaryDetector().detect_proposals(pcm=_pcm_from_levels(levels)) == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rms_step_seconds": 0.0}, "rms_step_seconds"),
        ({"baseline_seconds": 0.0}, "baseline_seconds"),
        ({"recovery_span_seconds": 0.0}, "recovery_span_seconds"),
        ({"max_recovery_seconds": 0.0}, "max_recovery_seconds"),
        ({"start_quiet_ratio": 1.0}, "start_quiet_ratio"),
        ({"basin_continue_ratio": 1.0}, "basin_continue_ratio"),
        ({"recovery_baseline_ratio": 1.0}, "recovery_baseline_ratio"),
        ({"recovery_jump_ratio": 1.0}, "recovery_jump_ratio"),
    ],
)
def test_adaptive_basin_detector_rejects_invalid_configuration(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveBasinBoundaryDetector(**kwargs)


def test_adaptive_basin_detector_exposes_depth_evidence() -> None:
    levels = [1000] * 30 + [100, 80, 70, 90, 120] + [500] * 8 + [1000] * 20
    detector = AdaptiveBasinBoundaryDetector()
    pcm = _pcm_from_levels(levels)

    evidence = detector.detect_evidence(pcm=pcm)

    assert len(evidence) == 1
    assert evidence[0].proposal.source is BoundaryProposalSource.BASIN
    assert evidence[0].proposal.time_seconds == pytest.approx(3.25)
    assert evidence[0].basin_depth == pytest.approx(0.07)
    assert evidence[0].structural_novelty is None
    assert detector.detect_proposals(pcm=pcm) == tuple(item.proposal for item in evidence)
