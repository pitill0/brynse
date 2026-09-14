"""Validated short-horizon spectral discontinuity evidence."""

from __future__ import annotations

from dataclasses import dataclass

from fluxtuner_ripper.models import DecodedPcm
from fluxtuner_ripper.spectral import FfmpegSpectralDistributionAnalyzer

_VALIDATED_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class LocalDiscontinuityEvidence:
    """Descriptive short-horizon PRE/POST discontinuity around one boundary."""

    change_250ms: float
    change_500ms: float
    change_1s: float
    change_2s: float

    def __post_init__(self) -> None:
        values = (
            ("change_250ms", self.change_250ms),
            ("change_500ms", self.change_500ms),
            ("change_1s", self.change_1s),
            ("change_2s", self.change_2s),
        )
        for name, value in values:
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, float]:
        """Return a flat serializable representation of local evidence."""
        return {
            "local_discontinuity_250ms": self.change_250ms,
            "local_discontinuity_500ms": self.change_500ms,
            "local_discontinuity_1s": self.change_1s,
            "local_discontinuity_2s": self.change_2s,
        }


class LocalDiscontinuityAnalyzer:
    """Measure validated local PRE/POST discontinuity at four short horizons."""

    def __init__(self, *, ffmpeg_binary: str = "ffmpeg") -> None:
        self._analyzers = (
            FfmpegSpectralDistributionAnalyzer(
                ffmpeg_binary=ffmpeg_binary,
                side_seconds=0.25,
                guard_seconds=0.0,
                block_seconds=0.25,
            ),
            FfmpegSpectralDistributionAnalyzer(
                ffmpeg_binary=ffmpeg_binary,
                side_seconds=0.5,
                guard_seconds=0.0,
                block_seconds=0.5,
            ),
            FfmpegSpectralDistributionAnalyzer(
                ffmpeg_binary=ffmpeg_binary,
                side_seconds=1.0,
                guard_seconds=0.0,
                block_seconds=1.0,
            ),
            FfmpegSpectralDistributionAnalyzer(
                ffmpeg_binary=ffmpeg_binary,
                side_seconds=2.0,
                guard_seconds=0.0,
                block_seconds=2.0,
            ),
        )

    @property
    def required_preroll_seconds(self) -> float:
        """Return the largest validated short-horizon PRE requirement."""
        return 2.0

    @property
    def required_postroll_seconds(self) -> float:
        """Return the largest validated short-horizon POST requirement."""
        return 2.0

    def analyze(
        self,
        *,
        pcm: DecodedPcm,
        boundary_time_seconds: float,
        absolute_start_time_seconds: float = 0.0,
    ) -> LocalDiscontinuityEvidence | None:
        """Return local evidence, or ``None`` when any horizon lacks context."""
        if pcm.sample_rate != _VALIDATED_SAMPLE_RATE:
            raise ValueError("local discontinuity analysis requires validated 16000 Hz PCM")

        measured: list[float] = []
        for analyzer in self._analyzers:
            evidence = analyzer.analyze(
                pcm=pcm,
                boundary_time_seconds=boundary_time_seconds,
                absolute_start_time_seconds=absolute_start_time_seconds,
            )
            if evidence is None:
                return None
            measured.append(evidence.between)

        return LocalDiscontinuityEvidence(
            change_250ms=measured[0],
            change_500ms=measured[1],
            change_1s=measured[2],
            change_2s=measured[3],
        )
