"""Boundary proposal and reconciliation domain model."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import StrEnum


class BoundaryProposalSource(StrEnum):
    """Audio-derived source that proposed a possible boundary."""

    BASIN = "basin"
    STRUCTURAL = "structural"


class BoundaryConfidence(StrEnum):
    """Reconciled confidence for one possible boundary."""

    UNRESOLVED = "unresolved"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class BoundaryProposal:
    """One audio-derived proposal for a possible boundary."""

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
    basin_start_seconds: float | None = None
    basin_recovery_seconds: float | None = None
    local_change: float | None = None
    persistent_change: float | None = None
    structural_novelty: float | None = None
    metadata_support: bool | None = None

    def __post_init__(self) -> None:
        values = (
            ("basin_depth", self.basin_depth),
            ("basin_start_seconds", self.basin_start_seconds),
            ("basin_recovery_seconds", self.basin_recovery_seconds),
            ("local_change", self.local_change),
            ("persistent_change", self.persistent_change),
            ("structural_novelty", self.structural_novelty),
        )
        for name, value in values:
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class BasinGeometry:
    """Preserved temporal geometry of one acoustic quiet basin."""

    start_seconds: float
    minimum_seconds: float
    recovery_seconds: float


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

    @property
    def basin_geometry(self) -> BasinGeometry | None:
        """Return the complete basin geometry attached to this hypothesis."""
        for item in self.evidence:
            if item.proposal.source is not BoundaryProposalSource.BASIN:
                continue
            if (
                item.basin_start_seconds is None
                or item.basin_recovery_seconds is None
            ):
                continue
            return BasinGeometry(
                start_seconds=item.basin_start_seconds,
                minimum_seconds=item.proposal.time_seconds,
                recovery_seconds=item.basin_recovery_seconds,
            )
        return None


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


class AcousticBoundaryHypothesisBuilder:
    """Build reconciled acoustic hypotheses from independent detectors."""

    def __init__(
        self,
        *,
        basin_detector,
        structural_detector,
        reconciler: BoundaryReconciler | None = None,
    ) -> None:
        self._basin_detector = basin_detector
        self._structural_detector = structural_detector
        self._reconciler = reconciler or BoundaryReconciler()

    def build(
        self,
        *,
        pcm,
        absolute_start_time_seconds: float = 0.0,
    ) -> tuple[BoundaryHypothesis, ...]:
        evidence = (
            *self._basin_detector.detect_evidence(
                pcm=pcm,
                absolute_start_time_seconds=absolute_start_time_seconds,
            ),
            *self._structural_detector.detect_evidence(
                pcm=pcm,
                absolute_start_time_seconds=absolute_start_time_seconds,
            ),
        )

        return self._reconciler.reconcile(
            tuple(item.proposal for item in evidence),
            evidence=evidence,
        )
