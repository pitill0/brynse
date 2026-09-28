from __future__ import annotations

import shutil
import subprocess
from array import array

import pytest

from brynse.models import DecodedPcm
from brynse.spectral import (
    FfmpegSpectralDistributionAnalyzer,
    SpectralAnalysisError,
    SpectralDistributionEvidence,
)


def _pcm(samples: list[int], *, sample_rate: int = 8000) -> DecodedPcm:
    return DecodedPcm(
        sample_rate=sample_rate,
        channels=1,
        sample_width_bytes=2,
        data=array("h", samples).tobytes(),
    )


def test_spectral_evidence_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="between"):
        SpectralDistributionEvidence(
            between=-0.01,
            before_sequential_variability=0.0,
            after_sequential_variability=0.0,
            before_distribution_spread=0.0,
            after_distribution_spread=0.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"side_seconds": 0.0}, "side_seconds"),
        ({"guard_seconds": -0.1}, "guard_seconds"),
        ({"block_seconds": 0.0}, "block_seconds"),
        ({"side_seconds": 2.0, "block_seconds": 3.0}, "block_seconds"),
        ({"side_seconds": 5.0, "block_seconds": 2.0}, "multiple"),
    ],
)
def test_spectral_analyzer_rejects_invalid_configuration(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FfmpegSpectralDistributionAnalyzer(**kwargs)


def test_spectral_analyzer_reports_required_context() -> None:
    analyzer = FfmpegSpectralDistributionAnalyzer(
        side_seconds=12.0,
        guard_seconds=2.0,
        block_seconds=2.0,
    )

    assert analyzer.required_preroll_seconds == 14.0
    assert analyzer.required_postroll_seconds == 14.0


def test_spectral_analyzer_returns_none_without_complete_context() -> None:
    pcm = _pcm([0] * (8000 * 20))

    analyzer = FfmpegSpectralDistributionAnalyzer(
        side_seconds=4.0,
        guard_seconds=1.0,
        block_seconds=2.0,
    )

    assert analyzer.analyze(pcm=pcm, boundary_time_seconds=4.0) is None
    assert analyzer.analyze(pcm=pcm, boundary_time_seconds=16.0) is None


def test_spectral_analyzer_applies_absolute_start_time() -> None:
    pcm = _pcm([0] * (8000 * 20))
    analyzer = FfmpegSpectralDistributionAnalyzer(
        side_seconds=4.0,
        guard_seconds=1.0,
        block_seconds=2.0,
    )

    assert (
        analyzer.analyze(
            pcm=pcm,
            boundary_time_seconds=104.0,
            absolute_start_time_seconds=100.0,
        )
        is None
    )

    with pytest.raises(ValueError, match="precedes"):
        analyzer.analyze(
            pcm=pcm,
            boundary_time_seconds=99.0,
            absolute_start_time_seconds=100.0,
        )


def test_spectral_analyzer_rejects_missing_binary() -> None:
    pcm = _pcm([1000] * (8000 * 12))

    with pytest.raises(SpectralAnalysisError, match="FFmpeg binary not found"):
        FfmpegSpectralDistributionAnalyzer(
            ffmpeg_binary="/definitely/missing/fluxtuner-ffmpeg",
            side_seconds=4.0,
            guard_seconds=1.0,
            block_seconds=2.0,
        ).analyze(
            pcm=pcm,
            boundary_time_seconds=6.0,
        )


def test_spectral_analyzer_detects_large_frequency_change_when_ffmpeg_available() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:duration=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=2200:duration=5",
            "-filter_complex",
            "[0:a][1:a]concat=n=2:v=0:a=1,aresample=8000,aformat=channel_layouts=mono",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )

    pcm = DecodedPcm(
        sample_rate=8000,
        channels=1,
        sample_width_bytes=2,
        data=generated.stdout,
    )
    analyzer = FfmpegSpectralDistributionAnalyzer(
        ffmpeg_binary=ffmpeg,
        side_seconds=3.0,
        guard_seconds=0.5,
        block_seconds=1.0,
    )

    evidence = analyzer.analyze(
        pcm=pcm,
        boundary_time_seconds=5.0,
    )

    assert evidence is not None
    assert evidence.between > 0.20
    assert evidence.before_sequential_variability >= 0.0
    assert evidence.after_sequential_variability >= 0.0
    assert evidence.before_distribution_spread >= 0.0
    assert evidence.after_distribution_spread >= 0.0


def test_spectral_analyzer_reports_small_change_for_stationary_tone_when_ffmpeg_available() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=700:duration=10",
            "-ar",
            "8000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    pcm = DecodedPcm(
        sample_rate=8000,
        channels=1,
        sample_width_bytes=2,
        data=generated.stdout,
    )
    analyzer = FfmpegSpectralDistributionAnalyzer(
        ffmpeg_binary=ffmpeg,
        side_seconds=3.0,
        guard_seconds=0.5,
        block_seconds=1.0,
    )

    evidence = analyzer.analyze(
        pcm=pcm,
        boundary_time_seconds=5.0,
    )

    assert evidence is not None
    assert evidence.between < 0.10
