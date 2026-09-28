"""Lightweight structural audio-boundary proposal detection."""

from __future__ import annotations

import math
import statistics
from array import array
from dataclasses import dataclass

from brynse.boundaries import (
    BoundaryEvidence,
    BoundaryProposal,
    BoundaryProposalSource,
)
from brynse.models import DecodedPcm


@dataclass(frozen=True)
class StructuralFeatureFrame:
    """Compact time-domain description of one PCM analysis frame."""

    start_time_seconds: float
    end_time_seconds: float
    rms: float
    zero_crossing_rate: float
    mean_absolute_delta: float
    crest_factor: float


class StructuralBoundaryDetector:
    """Propose boundaries from persistent changes in audio structure.

    This detector deliberately does not look for silence or RMS basins. It uses
    several normalized time-domain features and compares short feature contexts
    on each side of a possible boundary. The result is only a proposal source;
    final acceptance belongs to the boundary reconciliation/confidence layer.
    """

    def __init__(
        self,
        *,
        frame_seconds: float = 0.5,
        comparison_span_frames: int = 6,
        min_novelty: float = 0.12,
        min_spacing_seconds: float = 3.0,
    ) -> None:
        if frame_seconds <= 0:
            raise ValueError("frame_seconds must be greater than zero")
        if comparison_span_frames <= 0:
            raise ValueError("comparison_span_frames must be greater than zero")
        if min_novelty <= 0:
            raise ValueError("min_novelty must be greater than zero")
        if min_spacing_seconds <= 0:
            raise ValueError("min_spacing_seconds must be greater than zero")

        self._frame_seconds = frame_seconds
        self._comparison_span = comparison_span_frames
        self._min_novelty = min_novelty
        self._min_spacing = min_spacing_seconds

    def detect_proposals(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float = 0.0,
    ) -> tuple[BoundaryProposal, ...]:
        """Return structural novelty peaks as unresolved boundary proposals."""
        if absolute_start_time_seconds < 0:
            raise ValueError("absolute_start_time_seconds must be non-negative")

        frames = self._feature_frames(pcm)
        span = self._comparison_span
        if len(frames) < span * 2 + 1:
            return ()

        novelty: list[tuple[int, float]] = []
        for boundary_index in range(span, len(frames) - span + 1):
            before = frames[boundary_index - span : boundary_index]
            after = frames[boundary_index : boundary_index + span]
            score = self._context_distance(before, after)
            novelty.append((boundary_index, score))

        proposals: list[BoundaryProposal] = []
        last_time: float | None = None

        for index, (boundary_index, score) in enumerate(novelty):
            if score < self._min_novelty:
                continue

            left = novelty[index - 1][1] if index > 0 else None
            right = novelty[index + 1][1] if index + 1 < len(novelty) else None
            if left is not None and score < left:
                continue
            if right is not None and score < right:
                continue

            time_seconds = absolute_start_time_seconds + frames[boundary_index].start_time_seconds

            if last_time is not None and time_seconds - last_time < self._min_spacing:
                if proposals and score > (proposals[-1].strength or 0.0):
                    proposals[-1] = BoundaryProposal(
                        time_seconds=time_seconds,
                        source=BoundaryProposalSource.STRUCTURAL,
                        strength=min(1.0, score),
                    )
                    last_time = time_seconds
                continue

            proposals.append(
                BoundaryProposal(
                    time_seconds=time_seconds,
                    source=BoundaryProposalSource.STRUCTURAL,
                    strength=min(1.0, score),
                )
            )
            last_time = time_seconds

        return tuple(proposals)

    def detect_evidence(
        self,
        *,
        pcm: DecodedPcm,
        absolute_start_time_seconds: float = 0.0,
    ) -> tuple[BoundaryEvidence, ...]:
        """Return structural proposals with descriptive context evidence."""
        proposals = self.detect_proposals(
            pcm=pcm,
            absolute_start_time_seconds=absolute_start_time_seconds,
        )
        frames = self._feature_frames(pcm)

        return tuple(
            BoundaryEvidence(
                proposal=proposal,
                local_change=self._context_change_at_time(
                    frames=frames,
                    time_seconds=proposal.time_seconds - absolute_start_time_seconds,
                    side_seconds=8.0,
                    guard_seconds=1.0,
                ),
                persistent_change=self._context_change_at_time(
                    frames=frames,
                    time_seconds=proposal.time_seconds - absolute_start_time_seconds,
                    side_seconds=25.0,
                    guard_seconds=5.0,
                ),
                structural_novelty=proposal.strength,
            )
            for proposal in proposals
        )

    def _context_change_at_time(
        self,
        *,
        frames: tuple[StructuralFeatureFrame, ...],
        time_seconds: float,
        side_seconds: float,
        guard_seconds: float,
    ) -> float | None:
        """Compare equal feature contexts on both sides of one boundary time."""
        if not frames:
            return None

        boundary_index = min(
            range(len(frames)),
            key=lambda index: abs(frames[index].start_time_seconds - time_seconds),
        )
        side_frames = max(1, round(side_seconds / self._frame_seconds))
        guard_frames = max(0, round(guard_seconds / self._frame_seconds))

        left_end = boundary_index - guard_frames
        left_start = left_end - side_frames
        right_start = boundary_index + guard_frames
        right_end = right_start + side_frames

        if left_start < 0 or right_end > len(frames):
            return None

        before = frames[left_start:left_end]
        after = frames[right_start:right_end]
        if not before or not after:
            return None

        return self._context_distance(before, after)

    def _feature_frames(self, pcm: DecodedPcm) -> tuple[StructuralFeatureFrame, ...]:
        samples = array("h")
        samples.frombytes(pcm.data)

        samples_per_frame = max(1, round(pcm.sample_rate * self._frame_seconds))
        frames: list[StructuralFeatureFrame] = []

        for start in range(0, len(samples), samples_per_frame):
            chunk = samples[start : start + samples_per_frame]
            if len(chunk) < max(2, samples_per_frame // 2):
                continue

            square_sum = sum(sample * sample for sample in chunk)
            rms = math.sqrt(square_sum / len(chunk)) / 32768.0

            crossings = sum(
                1
                for left, right in zip(chunk, chunk[1:], strict=False)
                if (left < 0 <= right) or (left >= 0 > right)
            )
            zero_crossing_rate = crossings / max(1, len(chunk) - 1)

            mean_absolute_delta = (
                statistics.fmean(
                    abs(right - left) for left, right in zip(chunk, chunk[1:], strict=False)
                )
                / 65535.0
            )

            peak = max(abs(sample) for sample in chunk) / 32768.0
            crest_factor = peak / max(rms, 1e-9)

            frames.append(
                StructuralFeatureFrame(
                    start_time_seconds=start / pcm.sample_rate,
                    end_time_seconds=(start + len(chunk)) / pcm.sample_rate,
                    rms=rms,
                    zero_crossing_rate=zero_crossing_rate,
                    mean_absolute_delta=mean_absolute_delta,
                    crest_factor=crest_factor,
                )
            )

        return tuple(frames)

    @staticmethod
    def _centroid(
        frames: tuple[StructuralFeatureFrame, ...],
    ) -> tuple[float, float, float, float]:
        return (
            statistics.fmean(frame.rms for frame in frames),
            statistics.fmean(frame.zero_crossing_rate for frame in frames),
            statistics.fmean(frame.mean_absolute_delta for frame in frames),
            statistics.fmean(frame.crest_factor for frame in frames),
        )

    @classmethod
    def _context_distance(
        cls,
        before: tuple[StructuralFeatureFrame, ...],
        after: tuple[StructuralFeatureFrame, ...],
    ) -> float:
        left = cls._centroid(before)
        right = cls._centroid(after)

        scales = (1.0, 1.0, 1.0, 4.0)
        squared = sum(
            ((right_value - left_value) / scale) ** 2
            for left_value, right_value, scale in zip(left, right, scales, strict=False)
        )
        return math.sqrt(squared)
