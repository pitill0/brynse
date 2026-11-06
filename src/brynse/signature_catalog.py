"""Versioned, offline acoustic references for recurring radio audio.

A signature stores 2.5 seconds of 8 kHz mono signed 16-bit PCM.
It identifies repeated audio, not a guaranteed clean cut boundary.

No radio metadata, routing or output decisions are made here.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 8000
PROBE_SECONDS = 2.0
PROBE_OFFSET_SECONDS = 0.5
REFERENCE_SAMPLES = 20000
REFERENCE_BYTES = REFERENCE_SAMPLES * 2
CATALOG_VERSION = 1
MAX_SIGNATURES = 128
MAX_CATALOG_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class AcousticSignature:
    """One exact PCM reference belonging to one radio station."""

    station_id: str
    signature_id: str
    label: str
    pcm: bytes

    def __post_init__(self) -> None:
        for name in ("station_id", "signature_id", "label"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 256
                or any(ord(char) < 32 for char in value)
            ):
                raise ValueError(f"invalid {name}")

        if not isinstance(self.pcm, bytes) or len(self.pcm) != REFERENCE_BYTES:
            raise ValueError("signature must contain exactly 2.5 seconds of PCM")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.pcm).hexdigest()

    def probes(self) -> tuple[bytes, bytes]:
        """Two overlapping 2-second windows, spaced 500 ms apart."""
        count = int(PROBE_SECONDS * SAMPLE_RATE) * 2
        shift = int(PROBE_OFFSET_SECONDS * SAMPLE_RATE) * 2
        return self.pcm[:count], self.pcm[shift : shift + count]


@dataclass(frozen=True)
class SignatureCatalog:
    """An immutable collection of references for a single station."""

    station_id: str
    signatures: tuple[AcousticSignature, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.station_id, str) or not self.station_id.strip():
            raise ValueError("station_id is required")
        if len(self.station_id) > 256:
            raise ValueError("station_id is too long")
        if not isinstance(self.signatures, tuple):
            raise ValueError("signatures must be a tuple")
        if len(self.signatures) > MAX_SIGNATURES:
            raise ValueError("catalog exceeds signature limit")

        identifiers: set[str] = set()
        for signature in self.signatures:
            if not isinstance(signature, AcousticSignature):
                raise ValueError("invalid catalog entry")
            if signature.station_id != self.station_id:
                raise ValueError("signature belongs to a different station")
            if signature.signature_id in identifiers:
                raise ValueError("duplicate signature identifier")
            identifiers.add(signature.signature_id)


def catalog_to_document(catalog: SignatureCatalog) -> dict[str, Any]:
    """Return the portable schema-1 representation."""
    return {
        "schema_version": CATALOG_VERSION,
        "kind": "brynse_acoustic_signature_catalog",
        "station_id": catalog.station_id,
        "audio_format": {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "sample_width_bytes": 2,
            "encoding": "signed_16_bit_little_endian",
            "reference_samples": REFERENCE_SAMPLES,
        },
        "signatures": [
            {
                "id": signature.signature_id,
                "label": signature.label,
                "sha256": signature.sha256,
                "pcm_base64": base64.b64encode(signature.pcm).decode("ascii"),
            }
            for signature in catalog.signatures
        ],
    }


def catalog_from_document(document: dict[str, Any]) -> SignatureCatalog:
    """Validate version, format, payload and integrity before loading."""
    if not isinstance(document, dict):
        raise ValueError("invalid catalog document")
    if document.get("schema_version") != CATALOG_VERSION:
        raise ValueError("unsupported catalog schema")
    if document.get("kind") != "brynse_acoustic_signature_catalog":
        raise ValueError("unsupported catalog kind")

    expected_format = {
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width_bytes": 2,
        "encoding": "signed_16_bit_little_endian",
        "reference_samples": REFERENCE_SAMPLES,
    }
    if document.get("audio_format") != expected_format:
        raise ValueError("unsupported acoustic format")

    entries = document.get("signatures")
    if not isinstance(entries, list) or len(entries) > MAX_SIGNATURES:
        raise ValueError("invalid signature collection")

    station_id = document.get("station_id")
    if not isinstance(station_id, str):
        raise ValueError("invalid station identifier")

    signatures = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid signature entry")

        encoded = entry.get("pcm_base64")
        if not isinstance(encoded, str) or len(encoded) > 60000:
            raise ValueError("invalid encoded PCM")

        try:
            pcm = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid base64 PCM") from exc

        if hashlib.sha256(pcm).hexdigest() != entry.get("sha256"):
            raise ValueError("signature integrity check failed")

        signature_id = entry.get("id")
        label = entry.get("label")
        if not isinstance(signature_id, str):
            raise ValueError("invalid signature identifier")
        if not isinstance(label, str):
            raise ValueError("invalid signature label")

        signatures.append(
            AcousticSignature(
                station_id=station_id,
                signature_id=signature_id,
                label=label,
                pcm=pcm,
            )
        )

    return SignatureCatalog(station_id, tuple(signatures))


def save_catalog(path: Path, catalog: SignatureCatalog) -> None:
    """Create a new catalog exclusively; never overwrite existing data."""
    document = catalog_to_document(catalog)
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(payload)


def load_catalog(path: Path) -> SignatureCatalog:
    """Load a bounded local catalog and validate its complete contents."""
    path = Path(path)
    if path.stat().st_size > MAX_CATALOG_BYTES:
        raise ValueError("catalog file exceeds size limit")

    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    return catalog_from_document(document)
