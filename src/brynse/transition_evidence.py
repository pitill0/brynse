"""Composed descriptive evidence state around a candidate transition."""

from __future__ import annotations

from dataclasses import dataclass

from brynse.local_discontinuity import LocalDiscontinuityEvidence
from brynse.spectral import SpectralDistributionEvidence
from brynse.temporal_variability import TemporalVariabilityEvidence


@dataclass(frozen=True)
class CrossfadeRiseEvidence:
    """Descriptive evidence for one persistent post-reference energy rise."""

    time_seconds: float
    jump_ratio: float
    persistence_ratio: float
    after_global_ratio: float


@dataclass(frozen=True)
class OutgoingFadeEvidence:
    """Descriptive RMS geometry for one possible outgoing fade before a rise."""

    start_time_seconds: float
    floor_time_seconds: float
    rise_time_seconds: float
    start_rms: float
    floor_rms: float
    descent_ratio: float
    descent_consistency: float

    def __post_init__(self) -> None:
        if self.start_time_seconds < 0:
            raise ValueError("start_time_seconds must be non-negative")
        if self.floor_time_seconds < self.start_time_seconds:
            raise ValueError("floor_time_seconds must not precede start_time_seconds")
        if self.rise_time_seconds < self.floor_time_seconds:
            raise ValueError("rise_time_seconds must not precede floor_time_seconds")
        if self.start_rms < 0:
            raise ValueError("start_rms must be non-negative")
        if self.floor_rms < 0:
            raise ValueError("floor_rms must be non-negative")
        if self.descent_ratio < 0:
            raise ValueError("descent_ratio must be non-negative")
        if not 0.0 <= self.descent_consistency <= 1.0:
            raise ValueError("descent_consistency must be between zero and one")


@dataclass(frozen=True)
class CrossfadeBasinEvidence:
    """Descriptive evidence for one crossfade-compatible quiet basin."""

    center_time_seconds: float
    lead_seconds: float
    width_seconds: float
    prominence: float
    quietness: float

    def __post_init__(self) -> None:
        if self.center_time_seconds < 0:
            raise ValueError("center_time_seconds must be non-negative")
        if self.lead_seconds < 0:
            raise ValueError("lead_seconds must be non-negative")
        if self.width_seconds <= 0:
            raise ValueError("width_seconds must be greater than zero")
        if self.prominence < 0:
            raise ValueError("prominence must be non-negative")
        if self.quietness < 0:
            raise ValueError("quietness must be non-negative")


@dataclass(frozen=True)
class FadeRoutingEvidence:
    """Descriptive evidence available when considering fade ownership."""

    rise: CrossfadeRiseEvidence | None = None
    basins: tuple[CrossfadeBasinEvidence, ...] = ()
    outgoing_fades: tuple[OutgoingFadeEvidence, ...] = ()


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
