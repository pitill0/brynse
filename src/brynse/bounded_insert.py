"""Neutral observation of bounded inserted audio regions."""

from __future__ import annotations

from dataclasses import dataclass

from brynse.boundaries import (
    BasinGeometry,
    BoundaryConfidence,
    BoundaryEvidence,
    BoundaryHypothesis,
    BoundaryProposal,
    BoundaryProposalSource,
)


@dataclass(frozen=True)
class BoundedInsertEdgeEvidence:
    """Neutral acoustic description of one bounded-region edge."""

    hypothesis: BoundaryHypothesis

    @property
    def confidence(self) -> BoundaryConfidence:
        """Return reconciled boundary confidence without reinterpreting it."""
        return self.hypothesis.confidence

    @property
    def proposals(self) -> tuple[BoundaryProposal, ...]:
        """Return all proposals supporting the edge."""
        return self.hypothesis.proposals

    @property
    def evidence(self) -> tuple[BoundaryEvidence, ...]:
        """Return all descriptive measurements attached to the edge."""
        return self.hypothesis.evidence

    @property
    def basin_geometry(self) -> BasinGeometry | None:
        """Return preserved basin geometry when available."""
        return self.hypothesis.basin_geometry


@dataclass(frozen=True)
class BoundedInsertRegionEvidence:
    """Descriptive geometry between two independently supported boundaries."""

    reference_time_seconds: float
    left: BoundaryHypothesis
    right: BoundaryHypothesis
    interior_hypotheses: tuple[BoundaryHypothesis, ...] = ()

    def __post_init__(self) -> None:
        if self.reference_time_seconds < 0:
            raise ValueError("reference_time_seconds must be non-negative")
        if self.left.time_seconds > self.right.time_seconds:
            raise ValueError("left boundary must not follow right boundary")
        if self.left_geometry is None or self.right_geometry is None:
            raise ValueError("bounded region edges require basin geometry")

    @property
    def left_edge(self) -> BoundedInsertEdgeEvidence:
        """Return the complete neutral description of the left edge."""
        return BoundedInsertEdgeEvidence(self.left)

    @property
    def right_edge(self) -> BoundedInsertEdgeEvidence:
        """Return the complete neutral description of the right edge."""
        return BoundedInsertEdgeEvidence(self.right)

    @property
    def left_geometry(self) -> BasinGeometry | None:
        """Return complete basin geometry for the left supporting edge."""
        return self.left.basin_geometry

    @property
    def right_geometry(self) -> BasinGeometry | None:
        """Return complete basin geometry for the right supporting edge."""
        return self.right.basin_geometry

    @property
    def support_span_seconds(self) -> float:
        """Return temporal separation between supporting hypotheses."""
        return self.right.time_seconds - self.left.time_seconds

    @property
    def acoustic_gap_seconds(self) -> float:
        """Return gap between left recovery and right basin start.

        Positive values describe acoustic separation. Negative values describe
        overlapping basin envelopes.
        """
        assert self.left_geometry is not None
        assert self.right_geometry is not None

        return (
            self.right_geometry.start_seconds
            - self.left_geometry.recovery_seconds
        )

    @property
    def reference_inside_left_basin(self) -> bool:
        """Return whether the reference lies inside the left basin envelope."""
        assert self.left_geometry is not None

        return (
            self.left_geometry.start_seconds
            <= self.reference_time_seconds
            <= self.left_geometry.recovery_seconds
        )

    @property
    def reference_inside_right_basin(self) -> bool:
        """Return whether the reference lies inside the right basin envelope."""
        assert self.right_geometry is not None

        return (
            self.right_geometry.start_seconds
            <= self.reference_time_seconds
            <= self.right_geometry.recovery_seconds
        )

    @property
    def reference_offset_from_left_support_seconds(self) -> float:
        """Return signed distance from left supporting hypothesis."""
        return self.reference_time_seconds - self.left.time_seconds

    @property
    def reference_offset_to_right_support_seconds(self) -> float:
        """Return signed distance from reference to right supporting hypothesis."""
        return self.right.time_seconds - self.reference_time_seconds

    @property
    def structural_only_interior_count(self) -> int:
        """Count interior hypotheses supported only by structural proposals."""
        count = 0

        for hypothesis in self.interior_hypotheses:
            sources = {
                proposal.source
                for proposal in getattr(hypothesis, "proposals", ())
                if hasattr(proposal, "source")
            }

            if sources == {BoundaryProposalSource.STRUCTURAL}:
                count += 1

        return count


class BoundedInsertRegionDiscovery:
    """Describe bounded regions without classifying or cutting them."""

    @staticmethod
    def _is_supported_edge(hypothesis: BoundaryHypothesis) -> bool:
        proposals = getattr(hypothesis, "proposals", None)
        geometry = getattr(hypothesis, "basin_geometry", None)

        if proposals is None or geometry is None:
            return False

        sources = {
            proposal.source
            for proposal in proposals
            if hasattr(proposal, "source")
        }

        return {
            BoundaryProposalSource.BASIN,
            BoundaryProposalSource.STRUCTURAL,
        }.issubset(sources)

    @staticmethod
    def _reference_relates_to_pair(
        *,
        reference_time_seconds: float,
        left: BoundaryHypothesis,
        right: BoundaryHypothesis,
    ) -> bool:
        left_geometry = left.basin_geometry
        right_geometry = right.basin_geometry

        assert left_geometry is not None
        assert right_geometry is not None

        between_supports = (
            left.time_seconds
            <= reference_time_seconds
            <= right.time_seconds
        )
        inside_left_basin = (
            left_geometry.start_seconds
            <= reference_time_seconds
            <= left_geometry.recovery_seconds
        )
        inside_right_basin = (
            right_geometry.start_seconds
            <= reference_time_seconds
            <= right_geometry.recovery_seconds
        )

        return between_supports or inside_left_basin or inside_right_basin

    def discover(
        self,
        *,
        reference_time_seconds: float,
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> tuple[BoundedInsertRegionEvidence, ...]:
        """Return adjacent supported regions related to the reference time."""
        if reference_time_seconds < 0:
            raise ValueError("reference_time_seconds must be non-negative")

        ordered = tuple(
            sorted(
                hypotheses,
                key=lambda hypothesis: hypothesis.time_seconds,
            )
        )

        supported = tuple(
            hypothesis
            for hypothesis in ordered
            if self._is_supported_edge(hypothesis)
        )

        regions = []

        for left, right in zip(supported, supported[1:], strict=False):
            if not self._reference_relates_to_pair(
                reference_time_seconds=reference_time_seconds,
                left=left,
                right=right,
            ):
                continue

            interior = tuple(
                hypothesis
                for hypothesis in ordered
                if (
                    left.time_seconds
                    < hypothesis.time_seconds
                    < right.time_seconds
                )
            )

            regions.append(
                BoundedInsertRegionEvidence(
                    reference_time_seconds=reference_time_seconds,
                    left=left,
                    right=right,
                    interior_hypotheses=interior,
                )
            )

        return tuple(regions)
