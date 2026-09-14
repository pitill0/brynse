"""Read-only live shadow observation over the retained encoded stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fluxtuner_ripper.acoustic import FfmpegAcousticDecoder
from fluxtuner_ripper.basin import AdaptiveBasinBoundaryDetector
from fluxtuner_ripper.boundaries import (
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposalSource,
    BoundaryReconciler,
)
from fluxtuner_ripper.models import AcousticWindow
from fluxtuner_ripper.shadow import ShadowBoundaryAnalysis, ShadowBoundaryAnalyzer
from fluxtuner_ripper.spectral import FfmpegSpectralDistributionAnalyzer
from fluxtuner_ripper.structural import StructuralBoundaryDetector


@dataclass(frozen=True)
class LiveShadowConfig:
    """Runtime bounds for read-only live shadow analysis."""

    analysis_interval_seconds: float = 15.0
    history_seconds: float = 120.0
    report_dedup_seconds: float = 3.0
    stability_margin_seconds: float = 6.0

    def __post_init__(self) -> None:
        if self.analysis_interval_seconds <= 0:
            raise ValueError("analysis_interval_seconds must be greater than zero")
        if self.history_seconds <= 0:
            raise ValueError("history_seconds must be greater than zero")
        if self.report_dedup_seconds <= 0:
            raise ValueError("report_dedup_seconds must be greater than zero")
        if self.stability_margin_seconds < 0:
            raise ValueError("stability_margin_seconds must not be negative")


class LiveShadowBoundaryObserver:
    """Observe autonomous hypotheses without modifying ripping decisions."""

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        config: LiveShadowConfig | None = None,
        decoder: FfmpegAcousticDecoder | None = None,
        temporal_decoder: FfmpegAcousticDecoder | None = None,
        basin_detector: AdaptiveBasinBoundaryDetector | None = None,
        structural_detector: StructuralBoundaryDetector | None = None,
        reconciler: BoundaryReconciler | None = None,
        shadow_analyzer: ShadowBoundaryAnalyzer | None = None,
    ) -> None:
        self._config = config or LiveShadowConfig()
        self._decoder = decoder or FfmpegAcousticDecoder(ffmpeg_binary=ffmpeg_binary)
        self._temporal_decoder = temporal_decoder or FfmpegAcousticDecoder(
            ffmpeg_binary=ffmpeg_binary,
            output_sample_rate=16000,
        )
        self._basin_detector = basin_detector or AdaptiveBasinBoundaryDetector()
        self._structural_detector = structural_detector or StructuralBoundaryDetector()
        self._reconciler = reconciler or BoundaryReconciler()
        self._shadow_analyzer = (
            shadow_analyzer
            if shadow_analyzer is not None
            else ShadowBoundaryAnalyzer(
                spectral_analyzer=FfmpegSpectralDistributionAnalyzer(ffmpeg_binary=ffmpeg_binary)
            )
        )
        minimum_history_seconds = (
            self._shadow_analyzer.required_preroll_seconds
            + self._shadow_analyzer.required_postroll_seconds
            + self._config.stability_margin_seconds
            + self._config.analysis_interval_seconds
        )
        if self._config.history_seconds < minimum_history_seconds:
            raise ValueError(
                "history_seconds must cover shadow preroll + postroll + "
                "stability margin + analysis interval "
                f"(minimum {minimum_history_seconds:.3f}s, "
                f"got {self._config.history_seconds:.3f}s)"
            )

        self._last_analysis_end: float | None = None
        self._reported_times: list[float] = []
        self._finalized_until_seconds = 0.0

    def observe(
        self,
        *,
        timeline: Any,
        ring_buffer: Any,
    ) -> tuple[ShadowBoundaryAnalysis, ...]:
        """Return newly reportable shadow analyses from retained audio."""
        frames = timeline.frames
        if not frames:
            return ()

        latest = frames[-1]
        latest_end = latest.time_seconds + latest.samples / latest.sample_rate

        if (
            self._last_analysis_end is not None
            and latest_end - self._last_analysis_end < self._config.analysis_interval_seconds
        ):
            return ()

        window = self._retained_window(
            frames=frames,
            ring_buffer=ring_buffer,
            latest_end=latest_end,
        )
        if window is None:
            return ()

        pcm = self._decoder.decode(window)
        absolute_start = window.start_time_seconds

        evidence: tuple[BoundaryEvidence, ...] = (
            *self._basin_detector.detect_evidence(
                pcm=pcm,
                absolute_start_time_seconds=absolute_start,
            ),
            *self._structural_detector.detect_evidence(
                pcm=pcm,
                absolute_start_time_seconds=absolute_start,
            ),
        )
        hypotheses = self._reconciler.reconcile(
            tuple(item.proposal for item in evidence),
            evidence=evidence,
        )
        self._last_analysis_end = latest_end

        finalized_horizon = self._finalized_horizon(latest_end)

        analyses: list[ShadowBoundaryAnalysis] = []
        shadow_pcm = None
        for hypothesis in hypotheses:
            if hypothesis.time_seconds <= self._finalized_until_seconds:
                continue
            if hypothesis.time_seconds > finalized_horizon:
                continue
            if self._already_reported(hypothesis.time_seconds):
                continue
            if not self._has_complete_analysis_context(
                hypothesis=hypothesis,
                window=window,
            ):
                continue

            measure_spectral = self._should_measure_spectral(hypothesis)
            measure_temporal = self._should_measure_temporal(hypothesis)
            measure_local_discontinuity = self._should_measure_local_discontinuity(hypothesis)
            if (
                measure_spectral or measure_temporal or measure_local_discontinuity
            ) and shadow_pcm is None:
                shadow_pcm = self._temporal_decoder.decode(window)

            analysis = self._shadow_analyzer.analyze(
                hypothesis=hypothesis,
                pcm=shadow_pcm if measure_spectral and shadow_pcm is not None else pcm,
                temporal_pcm=shadow_pcm if measure_temporal else None,
                local_discontinuity_pcm=(shadow_pcm if measure_local_discontinuity else None),
                absolute_start_time_seconds=absolute_start,
                measure_spectral=measure_spectral,
                measure_temporal=measure_temporal,
                measure_local_discontinuity=measure_local_discontinuity,
            )
            analyses.append(analysis)
            self._reported_times.append(hypothesis.time_seconds)

        self._finalized_until_seconds = max(
            self._finalized_until_seconds,
            finalized_horizon,
        )
        return tuple(analyses)

    def _retained_window(
        self,
        *,
        frames: tuple[Any, ...],
        ring_buffer: Any,
        latest_end: float,
    ) -> AcousticWindow | None:
        retained = [
            frame
            for frame in frames
            if frame.offset + frame.length > ring_buffer.start_offset
            and frame.offset < ring_buffer.end_offset
        ]
        if not retained:
            return None

        target_start = max(0.0, latest_end - self._config.history_seconds)
        eligible = [frame for frame in retained if frame.time_seconds >= target_start]
        start_frame = eligible[0] if eligible else retained[0]
        end_frame = retained[-1]

        start_offset = max(start_frame.offset, ring_buffer.start_offset)
        end_offset = min(
            end_frame.offset + end_frame.length,
            ring_buffer.end_offset,
        )
        if start_offset >= end_offset:
            return None

        return AcousticWindow(
            start_offset=start_offset,
            end_offset=end_offset,
            start_time_seconds=start_frame.time_seconds,
            end_time_seconds=(end_frame.time_seconds + end_frame.samples / end_frame.sample_rate),
            data=ring_buffer.read(start_offset, end_offset),
        )

    def _finalized_horizon(self, latest_end: float) -> float:
        return max(
            0.0,
            latest_end
            - self._shadow_analyzer.required_postroll_seconds
            - self._config.stability_margin_seconds,
        )

    def _already_reported(self, time_seconds: float) -> bool:
        return any(
            abs(previous - time_seconds) <= self._config.report_dedup_seconds
            for previous in self._reported_times
        )

    def _has_complete_analysis_context(
        self,
        *,
        hypothesis: BoundaryHypothesis,
        window: AcousticWindow,
    ) -> bool:
        return (
            hypothesis.time_seconds - window.start_time_seconds
            >= self._shadow_analyzer.required_preroll_seconds
            and window.end_time_seconds - hypothesis.time_seconds
            >= self._shadow_analyzer.required_postroll_seconds
        )

    def _should_measure_temporal(self, hypothesis: BoundaryHypothesis) -> bool:
        if not getattr(self._shadow_analyzer, "supports_temporal_variability", False):
            return False
        return self._should_measure_spectral(hypothesis)

    @classmethod
    def _should_measure_local_discontinuity(
        cls,
        hypothesis: BoundaryHypothesis,
    ) -> bool:
        return cls._should_measure_spectral(hypothesis)

    @staticmethod
    def _should_measure_spectral(hypothesis: BoundaryHypothesis) -> bool:
        sources = {proposal.source for proposal in hypothesis.proposals}
        return sources == {
            BoundaryProposalSource.BASIN,
            BoundaryProposalSource.STRUCTURAL,
        }
