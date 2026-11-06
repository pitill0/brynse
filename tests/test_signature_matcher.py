"""Tests for offline matching of persisted acoustic references."""

from __future__ import annotations

import numpy as np
import pytest

from brynse.models import DecodedPcm
from brynse.signature_catalog import AcousticSignature, SignatureCatalog
from brynse.signature_matcher import scan_catalog

RATE = 8000


def pcm(samples: np.ndarray) -> DecodedPcm:
    return DecodedPcm(
        sample_rate=RATE,
        channels=1,
        sample_width_bytes=2,
        data=samples.astype("<i2").tobytes(),
    )


def fixture_audio() -> tuple[SignatureCatalog, np.ndarray]:
    rng = np.random.default_rng(12345)
    reference = rng.integers(-14000, 14001, size=20000, dtype=np.int16)
    signature = AcousticSignature(
        station_id="station-one",
        signature_id="ident-one",
        label="Ident one",
        pcm=reference.tobytes(),
    )
    return SignatureCatalog("station-one", (signature,)), reference


def test_finds_two_known_instances_without_position_hints() -> None:
    catalog, reference = fixture_audio()
    rng = np.random.default_rng(5678)
    audio = rng.integers(-12000, 12001, size=RATE * 12, dtype=np.int16)
    positions = (RATE * 2, RATE * 7)

    for position in positions:
        audio[position : position + len(reference)] = reference

    found = scan_catalog(pcm(audio), catalog)

    assert len(found) == 2
    assert [m.signature_id for m in found] == ["ident-one", "ident-one"]
    assert [m.start_seconds for m in found] == pytest.approx([2.0, 7.0], abs=0.001)
    assert all(m.correlation_first > 0.999 for m in found)
    assert all(m.correlation_second > 0.999 for m in found)


def test_unrelated_audio_produces_no_matches() -> None:
    catalog, _ = fixture_audio()
    rng = np.random.default_rng(7654)
    audio = rng.integers(-12000, 12001, size=RATE * 8, dtype=np.int16)
    assert scan_catalog(pcm(audio), catalog) == ()


def test_short_input_and_empty_catalog() -> None:
    catalog, _ = fixture_audio()
    assert scan_catalog(pcm(np.zeros(1000, dtype=np.int16)), catalog) == ()
    empty = SignatureCatalog("station-one", ())
    assert scan_catalog(pcm(np.zeros(RATE * 5, dtype=np.int16)), empty) == ()


@pytest.mark.parametrize("threshold", [0.0, -0.5, 1.01, float("nan"), float("inf")])
def test_invalid_thresholds_are_rejected(threshold: float) -> None:
    catalog, _ = fixture_audio()
    with pytest.raises(ValueError, match="threshold"):
        scan_catalog(pcm(np.zeros(RATE * 5, dtype=np.int16)), catalog, threshold=threshold)


def test_rejects_wrong_pcm_format() -> None:
    catalog, _ = fixture_audio()
    invalid = DecodedPcm(
        sample_rate=44100,
        channels=1,
        sample_width_bytes=2,
        data=bytes(40000),
    )
    with pytest.raises(ValueError, match="8 kHz"):
        scan_catalog(invalid, catalog)


def test_detects_reference_at_end_of_capture() -> None:
    catalog, reference = fixture_audio()
    rng = np.random.default_rng(2468)
    audio = rng.integers(-12000, 12001, size=RATE * 6, dtype=np.int16)
    position = len(audio) - len(reference)
    audio[position:] = reference
    matches = scan_catalog(pcm(audio), catalog)
    assert len(matches) == 1
    assert matches[0].start_seconds == pytest.approx(position / RATE, abs=0.001)


def test_match_spanning_internal_block_boundary() -> None:
    """A stored reference must survive an internal 120-second scan boundary."""
    catalog, reference = fixture_audio()
    rng = np.random.default_rng(4631)

    audio = rng.integers(-12000, 12001, size=RATE * 128, dtype=np.int16)
    positions = (int(119.5 * RATE), int(124.0 * RATE))

    for position in positions:
        audio[position : position + len(reference)] = reference

    found = scan_catalog(pcm(audio), catalog)

    assert len(found) == 2
    assert [match.start_seconds for match in found] == pytest.approx([119.5, 124.0], abs=0.001)
    assert all(match.correlation_second > 0.99 for match in found)


def test_first_probe_alone_does_not_confirm_a_jingle() -> None:
    """A matching first probe must not bypass second-probe confirmation."""
    catalog, reference = fixture_audio()
    rng = np.random.default_rng(4632)

    audio = rng.integers(-12000, 12001, size=RATE * 9, dtype=np.int16)

    start = RATE * 3
    audio[start : start + 16000] = reference[:16000]

    assert scan_catalog(pcm(audio), catalog) == ()


def test_two_distinct_signatures_are_reported_separately() -> None:
    """Multiple references must retain their distinct catalog identities."""
    first_catalog, first_reference = fixture_audio()
    rng = np.random.default_rng(4633)

    second_reference = rng.integers(-14000, 14001, size=20000, dtype=np.int16)
    second = AcousticSignature(
        station_id="station-one",
        signature_id="ident-two",
        label="Ident two",
        pcm=second_reference.tobytes(),
    )

    catalog = SignatureCatalog(
        "station-one",
        (first_catalog.signatures[0], second),
    )

    audio = rng.integers(-12000, 12001, size=RATE * 20, dtype=np.int16)
    audio[RATE * 3 : RATE * 3 + 20000] = first_reference
    audio[RATE * 12 : RATE * 12 + 20000] = second_reference

    found = scan_catalog(pcm(audio), catalog)

    assert [(match.signature_id, match.start_seconds) for match in found] == [
        ("ident-one", 3.0),
        ("ident-two", 12.0),
    ]
