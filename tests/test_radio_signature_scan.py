"""Contract tests for the offline radio signature scanner."""

from __future__ import annotations

import json

import pytest

from brynse.integrations.radio import signature_scan
from brynse.models import DecodedPcm
from brynse.signature_catalog import (
    AcousticSignature,
    SignatureCatalog,
    save_catalog,
)
from brynse.signature_matcher import SignatureMatch


@pytest.fixture
def scan_environment(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()

    catalog_path = tmp_path / "catalog.json"
    signature = AcousticSignature(
        station_id="test-station",
        signature_id="test-ident",
        label="Test ident",
        pcm=bytes(40000),
    )
    save_catalog(
        catalog_path,
        SignatureCatalog("test-station", (signature,)),
    )

    pcm = DecodedPcm(
        sample_rate=8000,
        channels=1,
        sample_width_bytes=2,
        data=bytes(8000 * 2 * 8),
    )

    calls = []

    def fake_decode(path):
        assert path == session
        calls.append("decode")
        return pcm

    def fake_scan(audio, catalog, *, threshold):
        assert audio is pcm
        assert catalog.station_id == "test-station"
        assert threshold == 0.90
        calls.append("scan")
        return (
            SignatureMatch(
                signature_id="test-ident",
                start_seconds=2.125,
                correlation_first=0.98,
                correlation_second=0.97,
            ),
        )

    monkeypatch.setattr(
        signature_scan,
        "decode_session_audio",
        fake_decode,
    )
    monkeypatch.setattr(
        signature_scan,
        "scan_catalog",
        fake_scan,
    )

    return session, catalog_path, calls


def test_cli_writes_structured_recognition_report(scan_environment, tmp_path, capsys):
    session, catalog, calls = scan_environment
    destination = tmp_path / "matches.json"

    rc = signature_scan.main(
        [
            str(session),
            "--catalog",
            str(catalog),
            "--json",
            str(destination),
        ]
    )

    assert rc == 0
    assert calls == ["decode", "scan"]

    document = json.loads(destination.read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert document["station_id"] == "test-station"
    assert document["duration_seconds"] == 8.0
    assert document["boundary_status"] == "not_assessed"
    assert document["audio_modified"] is False
    assert len(document["matches"]) == 1
    assert document["matches"][0]["time_seconds"] == 2.125
    assert document["matches"][0]["boundary_status"] == "not_assessed"
    assert "Recognition only" in capsys.readouterr().out


def test_cli_refuses_to_overwrite_report(scan_environment, tmp_path, capsys):
    session, catalog, calls = scan_environment
    destination = tmp_path / "matches.json"
    destination.write_text("existing report", encoding="utf-8")

    rc = signature_scan.main(
        [
            str(session),
            "--catalog",
            str(catalog),
            "--json",
            str(destination),
        ]
    )

    assert rc == 2
    assert calls == []
    assert destination.read_text(encoding="utf-8") == "existing report"
    assert "refusing to overwrite" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value",
    ["0", "-0.1", "1.1", "nan", "inf", "invalid"],
)
def test_cli_rejects_invalid_thresholds(scan_environment, value):
    session, catalog, calls = scan_environment

    with pytest.raises(SystemExit) as error:
        signature_scan.main(
            [
                str(session),
                "--catalog",
                str(catalog),
                "--threshold",
                value,
            ]
        )

    assert error.value.code == 2
    assert calls == []


def test_cli_rejects_missing_session(scan_environment, tmp_path, capsys):
    _, catalog, calls = scan_environment

    rc = signature_scan.main(
        [
            str(tmp_path / "missing"),
            "--catalog",
            str(catalog),
        ]
    )

    assert rc == 2
    assert calls == []
    assert "session directory" in capsys.readouterr().err


def test_cli_handles_empty_recognition(scan_environment, tmp_path, monkeypatch, capsys):
    session, catalog, _ = scan_environment

    monkeypatch.setattr(
        signature_scan,
        "scan_catalog",
        lambda _pcm, _catalog, *, threshold: (),
    )

    destination = tmp_path / "empty.json"

    assert (
        signature_scan.main(
            [
                str(session),
                "--catalog",
                str(catalog),
                "--json",
                str(destination),
            ]
        )
        == 0
    )

    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["matches"] == []
    assert document["audio_modified"] is False
    assert "No acoustic signatures matched" in capsys.readouterr().out


def test_cli_handles_invalid_catalog(scan_environment, tmp_path, capsys):
    session, _, calls = scan_environment
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version":999}', encoding="utf-8")

    assert (
        signature_scan.main(
            [
                str(session),
                "--catalog",
                str(invalid),
            ]
        )
        == 2
    )

    assert calls == []
    assert "Error:" in capsys.readouterr().err


def test_cli_rejects_invalid_utf8_catalog(scan_environment, tmp_path, capsys) -> None:
    session, _, calls = scan_environment
    invalid = tmp_path / "invalid-utf8.json"
    invalid.write_bytes(bytes([0xFF, 0xFE]))

    rc = signature_scan.main([str(session), "--catalog", str(invalid)])

    assert rc == 2
    assert calls == []
    assert "Error:" in capsys.readouterr().err


def test_cli_rejects_truncated_catalog(scan_environment, tmp_path, capsys) -> None:
    session, _, calls = scan_environment
    invalid = tmp_path / "truncated.json"
    invalid.write_text('{"schema_version":', encoding="utf-8")

    rc = signature_scan.main([str(session), "--catalog", str(invalid)])

    assert rc == 2
    assert calls == []
    assert "Error:" in capsys.readouterr().err
