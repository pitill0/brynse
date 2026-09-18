"""Command-line entry point for FluxTuner Ripper."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from fluxtuner_ripper.integrations.radio.models import TimedMetadataEvent
from fluxtuner_ripper.integrations.radio.runner import (
    RippingRunConfig,
    RippingRunError,
    RippingRunner,
    normalized_content_type,
    open_stream_source,
    resolve_codec,
    resolve_metaint,
)
from fluxtuner_ripper.integrations.radio.session_output import WrittenTrack
from fluxtuner_ripper.source import StreamSource


class CliError(RuntimeError):
    """Raised for user-facing CLI failures."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluxtuner-ripper",
        description="Native ICY radio stream ripper.",
    )
    parser.add_argument("url", help="HTTP/HTTPS radio stream URL")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for ripped tracks",
    )
    parser.add_argument(
        "--codec",
        choices=("auto", "mp3", "aac"),
        default="auto",
        help="Stream codec override (default: auto)",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg binary for later acoustic/finalization stages",
    )
    parser.add_argument(
        "--metadata-threshold",
        type=float,
        default=8.0,
        metavar="SECONDS",
        help="Metadata durability threshold (default: 8)",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=8.0,
        metavar="SECONDS",
        help="Acoustic boundary search radius (default: 8)",
    )
    parser.add_argument(
        "--transient-exclusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude short metadata intervals explicitly marked as ads/jingles (default: enabled)",
    )
    return parser


def _normalized_content_type(headers: Mapping[str, str]) -> str:
    return normalized_content_type(headers)


def _resolve_codec(requested: str, headers: Mapping[str, str]) -> str:
    try:
        return resolve_codec(requested, headers)
    except RippingRunError as exc:
        message = str(exc).replace(
            "codec='mp3' or codec='aac'",
            "--codec mp3 or --codec aac",
        )
        raise CliError(message) from exc


def _resolve_metaint(headers: Mapping[str, str]) -> int:
    try:
        return resolve_metaint(headers)
    except RippingRunError as exc:
        raise CliError(str(exc)) from exc


def _open_stream(url: str) -> StreamSource:
    return open_stream_source(url)


def _validate_args(args: argparse.Namespace) -> None:
    if args.metadata_threshold <= 0:
        raise CliError("--metadata-threshold must be greater than zero")
    if args.search_radius <= 0:
        raise CliError("--search-radius must be greater than zero")
    if not args.ffmpeg.strip():
        raise CliError("--ffmpeg must not be empty")


def _run_stream(args: argparse.Namespace) -> int:
    _validate_args(args)

    runner = RippingRunner(
        RippingRunConfig(
            url=args.url,
            output_directory=args.output,
            codec=args.codec,
            ffmpeg_binary=args.ffmpeg,
            metadata_threshold_seconds=args.metadata_threshold,
            search_radius_seconds=args.search_radius,
            transient_exclusion=args.transient_exclusion,
        ),
        stream_opener=_open_stream,
    )

    def print_started(codec: str, metaint: int) -> None:
        print(f"codec: {codec}")
        print(f"icy-metaint: {metaint}")
        print(f"output: {args.output}")
        print("ripping stream; press Ctrl+C to stop")

    def print_metadata(event: TimedMetadataEvent) -> None:
        print(
            f"[{event.audio_time_seconds:10.3f}s] {event.title}",
            flush=True,
        )

    def print_written(written: WrittenTrack) -> None:
        print(f"written: {written.path}", flush=True)

    try:
        result = runner.run(
            on_started=print_started,
            on_metadata=print_metadata,
            on_track_written=print_written,
        )
    except KeyboardInterrupt:
        runner.stop()
        current_title = runner.current_track_title
        if current_title is not None:
            print(
                f"incomplete track not finalized: {current_title}",
                file=sys.stderr,
            )
        print("stopped", file=sys.stderr)
        return 130
    print(f"codec: {result.codec}")
    print(f"icy-metaint: {result.metaint}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _run_stream(args)
    except BrokenPipeError:
        return 1
    except RippingRunError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except CliError as exc:
        parser.error(str(exc))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
