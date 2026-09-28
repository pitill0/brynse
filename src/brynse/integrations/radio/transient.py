"""Conservative semantic policy for excluding short non-track intervals."""

from __future__ import annotations

import re
import unicodedata

from brynse.integrations.radio.models import MetadataSemanticDecision
from brynse.models import SplitKind

_TOKEN_SEPARATOR = re.compile(r"[^a-z0-9]+")

_POSITIVE_MARKERS = (
    "advertisement",
    "advertising",
    "ad break",
    "commercial break",
    "werbepause",
    "werbung",
    "reklame",
    "publicidad",
    "intervalo publicitario",
    "spot publicitaire",
    "coupure publicitaire",
    "pubblicita",
    "jingle",
    "station break",
    "station id",
)

_NEGATIVE_MARKERS = (
    "ad free",
    "advert free",
    "advertisement free",
    "commercial free",
    "werbefrei",
    "ohne werbung",
    "sin publicidad",
    "sans publicite",
    "senza pubblicita",
)


def _normalize_title(title: str) -> str:
    decomposed = unicodedata.normalize("NFKD", title.casefold())
    ascii_text = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return _TOKEN_SEPARATOR.sub(" ", ascii_text).strip()


class ConservativeTransientExclusionPolicy:
    """Authorize only short metadata intervals with explicit non-track markers."""

    def __init__(self, *, max_duration_seconds: float = 8.0) -> None:
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be greater than zero")
        self._max_duration_seconds = max_duration_seconds

    @property
    def max_duration_seconds(self) -> float:
        return self._max_duration_seconds

    def __call__(self, decision: MetadataSemanticDecision) -> bool:
        if decision.kind is not SplitKind.NO_BOUNDARY:
            return False
        if not 0.0 < decision.lifetime_seconds <= self._max_duration_seconds:
            return False

        normalized = _normalize_title(decision.title)
        if not normalized:
            return False

        padded = f" {normalized} "
        if any(f" {marker} " in padded for marker in _NEGATIVE_MARKERS):
            return False

        return any(f" {marker} " in padded for marker in _POSITIVE_MARKERS)
