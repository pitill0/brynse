"""Passive capture of one exact live ICY radio session.

The raw stream file is the canonical replay source.  It contains the exact
bytes returned by StreamSource.read(), including ICY metadata framing.

events.jsonl is intentionally derivative: it records the TimedMetadataEvent
objects observed by the normal pipeline for human inspection and replay
validation, but it is not required to reconstruct the raw session.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from brynse.integrations.radio.models import TimedMetadataEvent


_SCHEMA_VERSION = 1


class RadioSessionCapture:
    """Persist exact stream input plus observable metadata sidecar."""

    def __init__(
        self,
        *,
        directory: Path,
        codec: str,
        metaint: int,
        metadata_threshold_seconds: float,
        transient_exclusion: bool,
        autonomous_boundaries: bool,
        source_url: str,
        source_metadata: Mapping[str, str],
    ) -> None:
        if codec not in {"mp3", "aac"}:
            raise ValueError(
                f"unsupported capture codec: {codec!r}"
            )

        if metaint <= 0:
            raise ValueError(
                "metaint must be greater than zero"
            )

        if metadata_threshold_seconds <= 0:
            raise ValueError(
                "metadata_threshold_seconds must be greater than zero"
            )

        self._directory = directory
        self._stream_path = directory / "stream.icy"
        self._events_path = directory / "events.jsonl"
        self._manifest_path = directory / "manifest.json"

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing = tuple(
            path
            for path in (
                self._stream_path,
                self._events_path,
                self._manifest_path,
            )
            if path.exists()
        )

        if existing:
            raise FileExistsError(
                "session capture already contains: "
                + ", ".join(
                    str(path)
                    for path in existing
                )
            )

        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "codec": codec,
            "metaint": metaint,
            "metadata_threshold_seconds": (
                metadata_threshold_seconds
            ),
            "transient_exclusion": transient_exclusion,
            "autonomous_boundaries": autonomous_boundaries,
            "source_url": source_url,
            "source_metadata": {
                str(key): str(value)
                for key, value in source_metadata.items()
            },
        }

        self._manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            self._stream = self._stream_path.open("xb")
            self._events = self._events_path.open(
                "x",
                encoding="utf-8",
            )
        except Exception:
            self._stream_path.unlink(
                missing_ok=True,
            )
            self._events_path.unlink(
                missing_ok=True,
            )
            self._manifest_path.unlink(
                missing_ok=True,
            )
            raise

        self._closed = False

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def stream_path(self) -> Path:
        return self._stream_path

    @property
    def events_path(self) -> Path:
        return self._events_path

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def write_stream_chunk(
        self,
        chunk: bytes,
    ) -> None:
        """Append bytes exactly as returned by StreamSource.read()."""

        if self._closed:
            raise RuntimeError(
                "session capture is closed"
            )

        if not chunk:
            return

        written = self._stream.write(chunk)

        if written != len(chunk):
            raise OSError(
                "session capture did not write the complete stream chunk"
            )

    def write_metadata_event(
        self,
        event: TimedMetadataEvent,
    ) -> None:
        """Record one normal-pipeline metadata observation."""

        if self._closed:
            raise RuntimeError(
                "session capture is closed"
            )

        self._events.write(
            json.dumps(
                {
                    "title": event.title,
                    "audio_offset": event.audio_offset,
                    "audio_time_seconds": (
                        event.audio_time_seconds
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )
        self._events.flush()

    def close(self) -> None:
        """Flush and close capture files idempotently."""

        if self._closed:
            return

        self._closed = True

        stream_error = None

        try:
            self._stream.flush()
            self._stream.close()
        except OSError as exc:
            stream_error = exc

        try:
            self._events.flush()
            self._events.close()
        except OSError:
            if stream_error is None:
                raise

        if stream_error is not None:
            raise stream_error
