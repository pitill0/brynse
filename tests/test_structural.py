from __future__ import annotations

from array import array

import pytest

from brynse.boundaries import BoundaryProposalSource
from brynse.models import DecodedPcm
from brynse.structural import StructuralBoundaryDetector


def _pcm(samples: list[int], *, sample_rate: int = 100) -> DecodedPcm:
    return DecodedPcm(
        sample_rate=sample_rate,
        channels=1,
        sample_width_bytes=2,
        data=array("h", samples).tobytes(),
    )


def _square_wave(
    *,
    sample_rate: int,
    seconds: float,
    half_period_samples: int,
    amplitude: int = 10000,
) -> list[int]:
    count = round(sample_rate * seconds)
    return [
        amplitude if (index // half_period_samples) % 2 == 0 else -amplitude
        for index in range(count)
    ]


def test_structural_detector_finds_same_energy_pattern_change() -> None:
    sample_rate = 100
    slow = _square_wave(sample_rate=sample_rate, seconds=6.0, half_period_samples=10)
    fast = _square_wave(sample_rate=sample_rate, seconds=6.0, half_period_samples=2)

    proposals = StructuralBoundaryDetector(
        frame_seconds=0.5,
        comparison_span_frames=4,
        min_novelty=0.05,
        min_spacing_seconds=2.0,
    ).detect_proposals(pcm=_pcm(slow + fast, sample_rate=sample_rate))

    assert proposals
    closest = min(proposals, key=lambda item: abs(item.time_seconds - 6.0))
    assert closest.time_seconds == pytest.approx(6.0, abs=0.5)
    assert closest.source is BoundaryProposalSource.STRUCTURAL
    assert closest.strength is not None


def test_structural_detector_does_not_require_an_rms_drop() -> None:
    sample_rate = 100
    slow = _square_wave(sample_rate=sample_rate, seconds=5.0, half_period_samples=10)
    fast = _square_wave(sample_rate=sample_rate, seconds=5.0, half_period_samples=2)

    detector = StructuralBoundaryDetector(
        frame_seconds=0.5,
        comparison_span_frames=3,
        min_novelty=0.05,
        min_spacing_seconds=2.0,
    )
    frames = detector._feature_frames(_pcm(slow + fast, sample_rate=sample_rate))

    assert frames[8].rms == pytest.approx(frames[12].rms)


def test_structural_detector_returns_empty_for_constant_structure() -> None:
    samples = _square_wave(sample_rate=100, seconds=12.0, half_period_samples=5)

    proposals = StructuralBoundaryDetector(
        frame_seconds=0.5,
        comparison_span_frames=4,
        min_novelty=0.05,
    ).detect_proposals(pcm=_pcm(samples))

    assert proposals == ()


def test_structural_detector_applies_absolute_start_time() -> None:
    sample_rate = 100
    first = _square_wave(sample_rate=sample_rate, seconds=6.0, half_period_samples=10)
    second = _square_wave(sample_rate=sample_rate, seconds=6.0, half_period_samples=2)

    proposals = StructuralBoundaryDetector(
        frame_seconds=0.5,
        comparison_span_frames=4,
        min_novelty=0.05,
        min_spacing_seconds=2.0,
    ).detect_proposals(
        pcm=_pcm(first + second, sample_rate=sample_rate),
        absolute_start_time_seconds=100.0,
    )

    closest = min(proposals, key=lambda item: abs(item.time_seconds - 106.0))
    assert closest.time_seconds == pytest.approx(106.0, abs=0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_seconds": 0.0}, "frame_seconds"),
        ({"comparison_span_frames": 0}, "comparison_span_frames"),
        ({"min_novelty": 0.0}, "min_novelty"),
        ({"min_spacing_seconds": 0.0}, "min_spacing_seconds"),
    ],
)
def test_structural_detector_rejects_invalid_configuration(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StructuralBoundaryDetector(**kwargs)


def test_structural_detector_exposes_novelty_evidence() -> None:
    sample_rate = 100
    slow = _square_wave(
        sample_rate=sample_rate,
        seconds=6.0,
        half_period_samples=10,
    )
    fast = _square_wave(
        sample_rate=sample_rate,
        seconds=6.0,
        half_period_samples=2,
    )
    pcm = _pcm(slow + fast, sample_rate=sample_rate)
    detector = StructuralBoundaryDetector(
        frame_seconds=0.5,
        comparison_span_frames=4,
        min_novelty=0.05,
        min_spacing_seconds=2.0,
    )

    evidence = detector.detect_evidence(pcm=pcm)

    assert evidence
    assert all(item.proposal.source is BoundaryProposalSource.STRUCTURAL for item in evidence)
    assert all(item.structural_novelty == item.proposal.strength for item in evidence)
    assert all(item.basin_depth is None for item in evidence)
    assert detector.detect_proposals(pcm=pcm) == tuple(item.proposal for item in evidence)


def test_structural_detector_exposes_local_and_persistent_change_evidence() -> None:
    sample_rate = 100
    slow = _square_wave(
        sample_rate=sample_rate,
        seconds=35.0,
        half_period_samples=10,
    )
    fast = _square_wave(
        sample_rate=sample_rate,
        seconds=35.0,
        half_period_samples=2,
    )
    pcm = _pcm(slow + fast, sample_rate=sample_rate)
    detector = StructuralBoundaryDetector(
        frame_seconds=0.5,
        comparison_span_frames=4,
        min_novelty=0.05,
        min_spacing_seconds=2.0,
    )

    evidence = detector.detect_evidence(pcm=pcm)

    closest = min(
        evidence,
        key=lambda item: abs(item.proposal.time_seconds - 35.0),
    )
    assert closest.proposal.time_seconds == pytest.approx(35.0, abs=0.5)
    assert closest.local_change is not None
    assert closest.local_change > 0.0
    assert closest.persistent_change is not None
    assert closest.persistent_change > 0.0


def test_structural_detector_leaves_unavailable_context_evidence_empty() -> None:
    sample_rate = 100
    slow = _square_wave(
        sample_rate=sample_rate,
        seconds=6.0,
        half_period_samples=10,
    )
    fast = _square_wave(
        sample_rate=sample_rate,
        seconds=6.0,
        half_period_samples=2,
    )
    pcm = _pcm(slow + fast, sample_rate=sample_rate)
    detector = StructuralBoundaryDetector(
        frame_seconds=0.5,
        comparison_span_frames=4,
        min_novelty=0.05,
        min_spacing_seconds=2.0,
    )

    evidence = detector.detect_evidence(pcm=pcm)

    closest = min(
        evidence,
        key=lambda item: abs(item.proposal.time_seconds - 6.0),
    )
    assert closest.local_change is None
    assert closest.persistent_change is None
