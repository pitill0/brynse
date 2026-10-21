"""Deterministic offline replay of captured radio sessions.

A replay session consumes the exact ``stream.icy`` bytes captured before
RippingSession.feed().  The normal RippingRunner, ICY parser, Observer,
Router, Specialists and PCM verification pipeline are reused unchanged.

``events.jsonl`` is not an input to the pipeline.  It is an oracle used to
verify that replay reconstructed the same metadata timeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from brynse.integrations.radio.cli import _routing_trace_record
from brynse.integrations.radio.models import TimedMetadataEvent
from brynse.integrations.radio.runner import (
    RippingRunConfig,
    RippingRunError,
    RippingRunner,
)
from brynse.source import BinaryIOStreamSource, StreamSource

_SCHEMA_VERSION = 1


class ReplaySessionError(RuntimeError):
    """Raised when a captured session cannot be replayed safely."""


@dataclass(frozen=True)
class ReplaySession:
    """Validated immutable description of one captured session."""

    directory: Path
    stream_path: Path
    events_path: Path
    manifest_path: Path
    codec: str
    metaint: int
    metadata_threshold_seconds: float
    transient_exclusion: bool
    autonomous_boundaries: bool
    source_url: str
    source_metadata: Mapping[str, str]


def load_replay_session(
    directory: Path,
) -> ReplaySession:
    """Load and validate one captured radio session."""

    stream_path = directory / "stream.icy"
    events_path = directory / "events.jsonl"
    manifest_path = directory / "manifest.json"

    for path in (
        stream_path,
        events_path,
        manifest_path,
    ):
        if not path.is_file():
            raise ReplaySessionError(
                f"replay session is missing {path.name}: {path}"
            )

    try:
        raw_manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8",
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplaySessionError(
            f"could not read replay manifest {manifest_path}: {exc}"
        ) from exc

    if not isinstance(raw_manifest, dict):
        raise ReplaySessionError(
            "replay manifest must contain a JSON object"
        )

    schema_version = raw_manifest.get(
        "schema_version"
    )

    if schema_version != _SCHEMA_VERSION:
        raise ReplaySessionError(
            "unsupported replay schema version: "
            f"{schema_version!r}"
        )

    codec = raw_manifest.get("codec")

    if codec not in {"mp3", "aac"}:
        raise ReplaySessionError(
            f"unsupported replay codec: {codec!r}"
        )

    metaint = raw_manifest.get("metaint")

    if (
        not isinstance(metaint, int)
        or isinstance(metaint, bool)
        or metaint <= 0
    ):
        raise ReplaySessionError(
            f"invalid replay metaint: {metaint!r}"
        )

    metadata_threshold_seconds = raw_manifest.get(
        "metadata_threshold_seconds"
    )

    if (
        not isinstance(
            metadata_threshold_seconds,
            (int, float),
        )
        or isinstance(
            metadata_threshold_seconds,
            bool,
        )
        or metadata_threshold_seconds <= 0
    ):
        raise ReplaySessionError(
            "invalid replay metadata threshold: "
            f"{metadata_threshold_seconds!r}"
        )

    transient_exclusion = raw_manifest.get(
        "transient_exclusion"
    )
    autonomous_boundaries = raw_manifest.get(
        "autonomous_boundaries"
    )

    if not isinstance(
        transient_exclusion,
        bool,
    ):
        raise ReplaySessionError(
            "replay transient_exclusion must be boolean"
        )

    if not isinstance(
        autonomous_boundaries,
        bool,
    ):
        raise ReplaySessionError(
            "replay autonomous_boundaries must be boolean"
        )

    source_url = raw_manifest.get("source_url")

    if (
        not isinstance(source_url, str)
        or not source_url
    ):
        raise ReplaySessionError(
            "replay source_url must be a non-empty string"
        )

    raw_metadata = raw_manifest.get(
        "source_metadata"
    )

    if not isinstance(raw_metadata, dict):
        raise ReplaySessionError(
            "replay source_metadata must be a JSON object"
        )

    source_metadata = {}

    for key, value in raw_metadata.items():
        if not isinstance(key, str):
            raise ReplaySessionError(
                "replay source_metadata keys must be strings"
            )

        if not isinstance(value, str):
            raise ReplaySessionError(
                "replay source_metadata values must be strings"
            )

        source_metadata[key] = value

    captured_metaint = source_metadata.get(
        "icy-metaint"
    )

    if captured_metaint != str(metaint):
        raise ReplaySessionError(
            "manifest metaint does not match captured "
            "icy-metaint metadata"
        )

    return ReplaySession(
        directory=directory,
        stream_path=stream_path,
        events_path=events_path,
        manifest_path=manifest_path,
        codec=codec,
        metaint=metaint,
        metadata_threshold_seconds=float(
            metadata_threshold_seconds
        ),
        transient_exclusion=transient_exclusion,
        autonomous_boundaries=autonomous_boundaries,
        source_url=source_url,
        source_metadata=source_metadata,
    )


def open_replay_source(
    session: ReplaySession,
) -> StreamSource:
    """Open the exact captured ICY byte stream."""

    try:
        stream: BinaryIO = session.stream_path.open(
            "rb"
        )
    except OSError as exc:
        raise ReplaySessionError(
            f"could not open replay stream "
            f"{session.stream_path}: {exc}"
        ) from exc

    return BinaryIOStreamSource(
        stream,
        metadata=session.source_metadata,
    )


def load_captured_events(
    session: ReplaySession,
) -> tuple[dict[str, object], ...]:
    """Load the captured metadata sidecar as validation evidence."""

    records = []

    try:
        lines = session.events_path.read_text(
            encoding="utf-8",
        ).splitlines()
    except OSError as exc:
        raise ReplaySessionError(
            f"could not read captured events "
            f"{session.events_path}: {exc}"
        ) from exc

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplaySessionError(
                "invalid captured event JSON at "
                f"line {line_number}: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ReplaySessionError(
                "captured event must be a JSON object "
                f"at line {line_number}"
            )

        records.append(record)

    return tuple(records)


def replay_event_record(
    event: TimedMetadataEvent,
) -> dict[str, object]:
    """Represent a replay event in capture-sidecar form."""

    return {
        "title": event.title,
        "audio_offset": event.audio_offset,
        "audio_time_seconds": (
            event.audio_time_seconds
        ),
    }


def validate_replayed_events(
    *,
    captured: tuple[dict[str, object], ...],
    replayed: tuple[dict[str, object], ...],
) -> None:
    """Require byte-derived replay metadata to match the capture."""

    if replayed == captured:
        return

    mismatch_index = None

    for index, (
        captured_item,
        replayed_item,
    ) in enumerate(
        zip(
            captured,
            replayed,
            strict=False,
        )
    ):
        if captured_item != replayed_item:
            mismatch_index = index
            break

    if mismatch_index is None:
        mismatch_index = min(
            len(captured),
            len(replayed),
        )

    captured_item = (
        captured[mismatch_index]
        if mismatch_index < len(captured)
        else None
    )
    replayed_item = (
        replayed[mismatch_index]
        if mismatch_index < len(replayed)
        else None
    )

    raise ReplaySessionError(
        "replayed metadata timeline does not match capture: "
        f"index={mismatch_index} "
        f"captured={captured_item!r} "
        f"replayed={replayed_item!r}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brynse-radio-replay",
        description=(
            "Replay an exact captured Brynse ICY session "
            "through the normal radio pipeline."
        ),
    )
    parser.add_argument(
        "session",
        type=Path,
        help=(
            "Captured session directory containing "
            "stream.icy, events.jsonl and manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for replayed tracks",
    )
    parser.add_argument(
        "--routing-trace",
        type=Path,
        help="Write replay routing observations as JSONL",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg binary used by the normal acoustic pipeline",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    session = load_replay_session(
        args.session
    )

    captured_events = load_captured_events(
        session
    )
    replayed_events: list[
        dict[str, object]
    ] = []

    routing_trace = None
    write_routing = None

    if args.routing_trace is not None:
        args.routing_trace.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            routing_trace = (
                args.routing_trace.open(
                    "w",
                    encoding="utf-8",
                )
            )
        except OSError as exc:
            raise ReplaySessionError(
                "could not open replay routing trace "
                f"{args.routing_trace}: {exc}"
            ) from exc

        def write_routing(observation) -> None:
            if routing_trace is None:
                raise ReplaySessionError('replay routing trace is not open')

            routing_trace.write(
                json.dumps(
                    _routing_trace_record(
                        observation
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            routing_trace.flush()

    runner = RippingRunner(
        RippingRunConfig(
            url=session.source_url,
            output_directory=args.output,
            codec=session.codec,
            ffmpeg_binary=args.ffmpeg,
            metadata_threshold_seconds=(
                session.metadata_threshold_seconds
            ),
            transient_exclusion=(
                session.transient_exclusion
            ),
            autonomous_boundaries=(
                session.autonomous_boundaries
            ),
            session_capture_directory=None,
        ),
        stream_opener=lambda _url: (
            open_replay_source(session)
        ),
    )

    def on_started(
        codec: str,
        metaint: int,
    ) -> None:
        print(f"replay codec: {codec}")
        print(f"replay icy-metaint: {metaint}")
        print(
            f"replay source: "
            f"{session.stream_path}"
        )
        print(
            f"replay output: {args.output}"
        )

    def on_metadata(
        event: TimedMetadataEvent,
    ) -> None:
        record = replay_event_record(event)
        replayed_events.append(record)

        print(
            f"[{event.audio_time_seconds:10.3f}s] "
            f"{event.title}",
            flush=True,
        )

    try:
        result = runner.run(
            on_started=on_started,
            on_metadata=on_metadata,
            on_routing_observation=write_routing,
        )
    finally:
        if routing_trace is not None:
            routing_trace.close()

    validate_replayed_events(
        captured=captured_events,
        replayed=tuple(replayed_events),
    )

    if result.codec != session.codec:
        raise ReplaySessionError(
            "replayed codec does not match manifest: "
            f"{result.codec!r} != {session.codec!r}"
        )

    if result.metaint != session.metaint:
        raise ReplaySessionError(
            "replayed metaint does not match manifest: "
            f"{result.metaint!r} != {session.metaint!r}"
        )

    print(
        "replay metadata validation: PASS"
    )
    print(
        f"replay metadata events: "
        f"{len(replayed_events)}"
    )

    return 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except (
        ReplaySessionError,
        RippingRunError,
    ) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
