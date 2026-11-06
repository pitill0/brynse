"""Tests for the offline, versioned acoustic reference catalog."""

import copy

import pytest

from brynse.signature_catalog import (
    AcousticSignature,
    SignatureCatalog,
    catalog_from_document,
    catalog_to_document,
    load_catalog,
    save_catalog,
)

PCM = bytes(range(256)) * 156 + bytes(range(64))


def sample_signature(identifier: str = "ident-01") -> AcousticSignature:
    return AcousticSignature(
        station_id="jazz-lounge-bar",
        signature_id=identifier,
        label="Station ident",
        pcm=PCM,
    )


def test_catalog_roundtrip_and_probe_layout(tmp_path) -> None:
    signature = sample_signature()
    catalog = SignatureCatalog("jazz-lounge-bar", (signature,))
    destination = tmp_path / "catalog.json"

    save_catalog(destination, catalog)
    restored = load_catalog(destination)

    assert restored == catalog
    first, second = restored.signatures[0].probes()
    assert len(first) == len(second) == 32000
    assert first == PCM[:32000]
    assert second == PCM[8000:40000]


def test_save_never_overwrites_existing_catalog(tmp_path) -> None:
    destination = tmp_path / "catalog.json"
    catalog = SignatureCatalog("jazz-lounge-bar", (sample_signature(),))
    save_catalog(destination, catalog)

    with pytest.raises(FileExistsError):
        save_catalog(destination, catalog)

    assert load_catalog(destination) == catalog


def test_rejects_corrupted_reference_audio() -> None:
    document = catalog_to_document(SignatureCatalog("jazz-lounge-bar", (sample_signature(),)))
    document["signatures"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="integrity"):
        catalog_from_document(document)


def test_rejects_unsupported_version_and_format() -> None:
    original = catalog_to_document(SignatureCatalog("jazz-lounge-bar", (sample_signature(),)))
    newer = copy.deepcopy(original)
    newer["schema_version"] = 999

    with pytest.raises(ValueError, match="schema"):
        catalog_from_document(newer)

    wrong_rate = copy.deepcopy(original)
    wrong_rate["audio_format"]["sample_rate"] = 44100

    with pytest.raises(ValueError, match="format"):
        catalog_from_document(wrong_rate)


def test_station_isolation_and_duplicate_identifiers() -> None:
    signature = sample_signature()
    with pytest.raises(ValueError, match="different station"):
        SignatureCatalog("another-station", (signature,))

    with pytest.raises(ValueError, match="duplicate"):
        SignatureCatalog("jazz-lounge-bar", (signature, signature))


def test_rejects_short_pcm_and_invalid_base64() -> None:
    with pytest.raises(ValueError, match="2.5 seconds"):
        AcousticSignature("station", "ident", "Ident", b"short")

    document = catalog_to_document(SignatureCatalog("jazz-lounge-bar", (sample_signature(),)))
    document["signatures"][0]["pcm_base64"] = "invalid!"

    with pytest.raises(ValueError, match="base64"):
        catalog_from_document(document)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("id", None),
        ("id", 123),
        ("id", ""),
        ("label", None),
        ("label", 123),
        ("label", ""),
    ],
)
def test_rejects_invalid_signature_metadata(field, bad_value) -> None:
    document = catalog_to_document(SignatureCatalog("jazz-lounge-bar", (sample_signature(),)))
    document["signatures"][0][field] = bad_value

    with pytest.raises(ValueError):
        catalog_from_document(document)


def test_rejects_oversized_catalog_file(tmp_path) -> None:
    destination = tmp_path / "large.json"
    with destination.open("wb") as handle:
        handle.truncate(16 * 1024 * 1024 + 1)

    with pytest.raises(ValueError, match="size limit"):
        load_catalog(destination)
