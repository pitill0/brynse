"""Boundary proposal and reconciliation domain model."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import StrEnum


class BoundaryProposalSource(StrEnum):
    """Audio-derived source that proposed a possible track boundary."""

    BASIN = "basin"
    STRUCTURAL = "structural"


class BoundaryConfidence(StrEnum):
    """Reconciled confidence for one possible track boundary."""

    UNRESOLVED = "unresolved"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class BoundaryProposal:
    """One audio-derived proposal for a possible track boundary."""

    time_seconds: float
    source: BoundaryProposalSource
    strength: float | None = None

    def __post_init__(self) -> None:
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")
        if self.strength is not None and not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between zero and one")


@dataclass(frozen=True)
class BoundaryEvidence:
    """Independent descriptive evidence attached to one proposal."""

    proposal: BoundaryProposal
    basin_depth: float | None = None
    local_change: float | None = None
    persistent_change: float | None = None
    structural_novelty: float | None = None
    metadata_support: bool | None = None

    def __post_init__(self) -> None:
        values = (
            ("basin_depth", self.basin_depth),
            ("local_change", self.local_change),
            ("persistent_change", self.persistent_change),
            ("structural_novelty", self.structural_novelty),
        )
        for name, value in values:
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class BoundaryHypothesis:
    """One reconciled temporal boundary hypothesis."""

    time_seconds: float
    proposals: tuple[BoundaryProposal, ...]
    evidence: tuple[BoundaryEvidence, ...] = ()
    confidence: BoundaryConfidence = BoundaryConfidence.UNRESOLVED

    def __post_init__(self) -> None:
        if self.time_seconds < 0:
            raise ValueError("time_seconds must be non-negative")
        if not self.proposals:
            raise ValueError("boundary hypotheses require at least one proposal")
        for item in self.evidence:
            if item.proposal not in self.proposals:
                raise ValueError("evidence proposal must belong to the hypothesis")


class BoundaryReconciler:
    """Cluster nearby audio proposals into single unresolved hypotheses."""

    def __init__(self, *, cluster_radius_seconds: float = 3.0) -> None:
        if cluster_radius_seconds <= 0:
            raise ValueError("cluster_radius_seconds must be greater than zero")
        self._cluster_radius = cluster_radius_seconds

    def reconcile(
        self,
        proposals: tuple[BoundaryProposal, ...],
        *,
        evidence: tuple[BoundaryEvidence, ...] = (),
    ) -> tuple[BoundaryHypothesis, ...]:
        """Return time-clustered hypotheses without assigning confidence yet."""
        if not proposals:
            if evidence:
                raise ValueError("evidence requires at least one proposal")
            return ()

        known = set(proposals)
        for item in evidence:
            if item.proposal not in known:
                raise ValueError("evidence references an unknown proposal")

        ordered = sorted(
            proposals,
            key=lambda item: (item.time_seconds, item.source.value),
        )

        clusters: list[list[BoundaryProposal]] = []
        current: list[BoundaryProposal] = [ordered[0]]

        for proposal in ordered[1:]:
            cluster_span = proposal.time_seconds - current[0].time_seconds

            if cluster_span <= self._cluster_radius:
                current.append(proposal)
                continue

            clusters.append(current)
            current = [proposal]

        clusters.append(current)

        hypotheses: list[BoundaryHypothesis] = []
        for cluster in clusters:
            cluster_tuple = tuple(cluster)
            cluster_evidence = tuple(item for item in evidence if item.proposal in cluster_tuple)
            hypotheses.append(
                BoundaryHypothesis(
                    time_seconds=statistics.median(item.time_seconds for item in cluster_tuple),
                    proposals=cluster_tuple,
                    evidence=cluster_evidence,
                )
            )

        return tuple(hypotheses)
