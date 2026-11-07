"""Regression contracts for bounded acoustic FFmpeg decoding."""

import subprocess
from types import SimpleNamespace

import pytest

from brynse.acoustic import AcousticDecodeError, FfmpegAcousticDecoder


def test_acoustic_decoder_has_finite_default_timeout(monkeypatch):
    observed = {}

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=b"\x00\x00",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    decoder = FfmpegAcousticDecoder()
    decoder.decode(SimpleNamespace(data=b"encoded-audio"))

    timeout = observed.get("timeout")
    assert isinstance(timeout, (int, float))
    assert 0 < timeout <= 120


def test_acoustic_decoder_wraps_timeout_as_domain_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout", 1),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    decoder = FfmpegAcousticDecoder()

    with pytest.raises(AcousticDecodeError, match="timed out"):
        decoder.decode(SimpleNamespace(data=b"encoded-audio"))


@pytest.mark.parametrize("timeout", [0, -1, 3601, float("inf"), float("nan")])
def test_acoustic_decoder_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        FfmpegAcousticDecoder(timeout_seconds=timeout)


def test_acoustic_decoder_accepts_custom_timeout(monkeypatch):
    observed = {}

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=b"\\x00\\x00",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    decoder = FfmpegAcousticDecoder(timeout_seconds=7.5)
    decoder.decode(SimpleNamespace(data=b"encoded-audio"))

    assert observed["timeout"] == 7.5
