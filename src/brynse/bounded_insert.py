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
class BoundedInsertAsymmetricRegionEvidence:
    """Descriptive asymmetric bounded-region geometry.

    The left edge has reconciled basin and structural support. The right edge
    preserves complementary acoustic evidence without implying ownership or
    choosing temporal boundaries.
    """

    reference_time_seconds: float
    primary: BoundaryHypothesis
    complementary: BoundaryHypothesis
    interior_hypotheses: tuple[BoundaryHypothesis, ...] = ()

    def __post_init__(self) -> None:
        if self.reference_time_seconds < 0:
            raise ValueError("reference_time_seconds must be non-negative")
        if self.primary.time_seconds > self.complementary.time_seconds:
            raise ValueError("primary edge must not follow complementary edge")

    @property
    def primary_edge(self) -> BoundedInsertEdgeEvidence:
        """Return complete neutral evidence for the primary edge."""
        return BoundedInsertEdgeEvidence(self.primary)

    @property
    def complementary_edge(self) -> BoundedInsertEdgeEvidence:
        """Return complete neutral evidence for the complementary edge."""
        return BoundedInsertEdgeEvidence(self.complementary)

    @property
    def primary_distance_from_reference_seconds(self) -> float:
        """Return distance from primary support to reference."""
        return self.reference_time_seconds - self.primary.time_seconds

    @property
    def complementary_distance_from_reference_seconds(self) -> float:
        """Return distance from reference to complementary support."""
        return self.complementary.time_seconds - self.reference_time_seconds


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
    def acoustic_gap_seconds(self) -> float | None:
        """Return basin-envelope gap when both edges preserve basin geometry.

        Positive values describe acoustic separation. Negative values describe
        overlapping basin envelopes. Missing basin geometry remains unknown.
        """
        left_geometry = self.left_geometry
        right_geometry = self.right_geometry

        if left_geometry is None or right_geometry is None:
            return None

        return (
            right_geometry.start_seconds
            - left_geometry.recovery_seconds
        )

    @property
    def reference_inside_left_basin(self) -> bool:
        """Return whether the reference lies inside the left basin envelope."""
        geometry = self.left_geometry

        if geometry is None:
            return False

        return (
            geometry.start_seconds
            <= self.reference_time_seconds
            <= geometry.recovery_seconds
        )

    @property
    def reference_inside_right_basin(self) -> bool:
        """Return whether the reference lies inside the right basin envelope."""
        geometry = self.right_geometry

        if geometry is None:
            return False

        return (
            geometry.start_seconds
            <= self.reference_time_seconds
            <= geometry.recovery_seconds
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

    @staticmethod
    def _proposal_sources(
        hypothesis: BoundaryHypothesis,
    ) -> set[BoundaryProposalSource]:
        return {
            proposal.source
            for proposal in getattr(hypothesis, "proposals", ())
            if hasattr(proposal, "source")
        }

    @classmethod
    def _is_complementary_edge(
        cls,
        hypothesis: BoundaryHypothesis,
    ) -> bool:
        sources = cls._proposal_sources(hypothesis)
        geometry = getattr(hypothesis, "basin_geometry", None)

        if sources == {BoundaryProposalSource.STRUCTURAL}:
            return True

        return (
            sources == {BoundaryProposalSource.BASIN}
            and geometry is not None
        )

    def discover_asymmetric(
        self,
        *,
        reference_time_seconds: float,
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> tuple[BoundedInsertAsymmetricRegionEvidence, ...]:
        """Describe asymmetric primary/complementary regions near reference."""
        if reference_time_seconds < 0:
            raise ValueError("reference_time_seconds must be non-negative")

        ordered = tuple(
            sorted(
                hypotheses,
                key=lambda hypothesis: hypothesis.time_seconds,
            )
        )

        left_primaries = tuple(
            hypothesis
            for hypothesis in ordered
            if (
                hypothesis.time_seconds < reference_time_seconds
                and self._is_supported_edge(hypothesis)
            )
        )

        right_complementaries = tuple(
            hypothesis
            for hypothesis in ordered
            if (
                hypothesis.time_seconds > reference_time_seconds
                and self._is_complementary_edge(hypothesis)
            )
        )

        if not left_primaries or not right_complementaries:
            return ()

        complementary = min(
            right_complementaries,
            key=lambda hypothesis: hypothesis.time_seconds,
        )

        regions = []

        for primary in left_primaries:
            interior = tuple(
                hypothesis
                for hypothesis in ordered
                if (
                    primary.time_seconds
                    < hypothesis.time_seconds
                    < complementary.time_seconds
                )
            )

            if any(
                self._is_supported_edge(hypothesis)
                for hypothesis in interior
            ):
                continue

            primary_distance = (
                reference_time_seconds
                - primary.time_seconds
            )
            complementary_distance = (
                complementary.time_seconds
                - reference_time_seconds
            )

            if complementary_distance >= primary_distance:
                continue

            regions.append(
                BoundedInsertAsymmetricRegionEvidence(
                    reference_time_seconds=reference_time_seconds,
                    primary=primary,
                    complementary=complementary,
                    interior_hypotheses=interior,
                )
            )

        return tuple(regions)

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
