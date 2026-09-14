"""Composed descriptive evidence state around a candidate transition."""

from __future__ import annotations

from dataclasses import dataclass

from fluxtuner_ripper.local_discontinuity import LocalDiscontinuityEvidence
from fluxtuner_ripper.spectral import SpectralDistributionEvidence
from fluxtuner_ripper.temporal_variability import TemporalVariabilityEvidence


@dataclass(frozen=True)
class TransitionEvidenceState:
    """Read-only multi-signal evidence state for one candidate transition."""

    spectral_change: float | None = None
    variability_ratio_8s: float | None = None
    variability_ratio_16s: float | None = None
    variability_ratio_24s: float | None = None
    variability_ratio_40s: float | None = None
    local_discontinuity_250ms: float | None = None
    local_discontinuity_500ms: float | None = None
    local_discontinuity_1s: float | None = None
    local_discontinuity_2s: float | None = None

    def __post_init__(self) -> None:
        values = (
            ("spectral_change", self.spectral_change),
            ("variability_ratio_8s", self.variability_ratio_8s),
            ("variability_ratio_16s", self.variability_ratio_16s),
            ("variability_ratio_24s", self.variability_ratio_24s),
            ("variability_ratio_40s", self.variability_ratio_40s),
            ("local_discontinuity_250ms", self.local_discontinuity_250ms),
            ("local_discontinuity_500ms", self.local_discontinuity_500ms),
            ("local_discontinuity_1s", self.local_discontinuity_1s),
            ("local_discontinuity_2s", self.local_discontinuity_2s),
        )
        for name, value in values:
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def has_temporal_variability(self) -> bool:
        """Return whether all validated temporal-variability horizons are present."""
        return all(
            value is not None
            for value in (
                self.variability_ratio_8s,
                self.variability_ratio_16s,
                self.variability_ratio_24s,
                self.variability_ratio_40s,
            )
        )

    def as_dict(self) -> dict[str, float | None]:
        """Return a flat serializable representation of descriptive evidence."""
        return {
            "spectral_change": self.spectral_change,
            "variability_ratio_8s": self.variability_ratio_8s,
            "variability_ratio_16s": self.variability_ratio_16s,
            "variability_ratio_24s": self.variability_ratio_24s,
            "variability_ratio_40s": self.variability_ratio_40s,
            "local_discontinuity_250ms": self.local_discontinuity_250ms,
            "local_discontinuity_500ms": self.local_discontinuity_500ms,
            "local_discontinuity_1s": self.local_discontinuity_1s,
            "local_discontinuity_2s": self.local_discontinuity_2s,
        }


class TransitionEvidenceExtractor:
    """Compose a transition evidence state from already measured signals."""

    def extract(
        self,
        *,
        spectral: SpectralDistributionEvidence | None,
        temporal: TemporalVariabilityEvidence | None = None,
        local_discontinuity: LocalDiscontinuityEvidence | None = None,
        variability_ratio_8s: float | None = None,
        variability_ratio_16s: float | None = None,
        variability_ratio_24s: float | None = None,
        variability_ratio_40s: float | None = None,
    ) -> TransitionEvidenceState:
        """Return evidence state without making a boundary decision."""
        return TransitionEvidenceState(
            spectral_change=spectral.between if spectral is not None else None,
            variability_ratio_8s=(
                temporal.variability_ratio_8s if temporal is not None else variability_ratio_8s
            ),
            variability_ratio_16s=(
                temporal.variability_ratio_16s if temporal is not None else variability_ratio_16s
            ),
            variability_ratio_24s=(
                temporal.variability_ratio_24s if temporal is not None else variability_ratio_24s
            ),
            variability_ratio_40s=(
                temporal.variability_ratio_40s if temporal is not None else variability_ratio_40s
            ),
            local_discontinuity_250ms=(
                local_discontinuity.change_250ms if local_discontinuity is not None else None
            ),
            local_discontinuity_500ms=(
                local_discontinuity.change_500ms if local_discontinuity is not None else None
            ),
            local_discontinuity_1s=(
                local_discontinuity.change_1s if local_discontinuity is not None else None
            ),
            local_discontinuity_2s=(
                local_discontinuity.change_2s if local_discontinuity is not None else None
            ),
        )
