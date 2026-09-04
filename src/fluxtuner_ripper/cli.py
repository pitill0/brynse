"""Command-line entry point for FluxTuner Ripper."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

from fluxtuner_ripper import (
    AcousticWindowExtractor,
    FfmpegAcousticDecoder,
    MetadataSemanticTracker,
    NearestBoundaryMatcher,
    RippingOrchestrator,
    RippingSession,
    RippingStreamIngestor,
    SessionOutputWriter,
)

_CHUNK_SIZE = 64 * 1024
_CONTENT_TYPE_CODECS = {
    "audio/aac": "aac",
    "audio/aacp": "aac",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}


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
    return parser


def _normalized_content_type(headers: Mapping[str, str]) -> str:
    raw = headers.get("Content-Type", "")
    return raw.split(";", 1)[0].strip().lower()


def _resolve_codec(requested: str, headers: Mapping[str, str]) -> str:
    if requested != "auto":
        return requested

    content_type = _normalized_content_type(headers)
    codec = _CONTENT_TYPE_CODECS.get(content_type)
    if codec is None:
        raise CliError(
            "could not detect stream codec from Content-Type "
            f"{content_type or '<missing>'!r}; use --codec mp3 or --codec aac"
        )
    return codec


def _resolve_metaint(headers: Mapping[str, str]) -> int:
    raw = headers.get("icy-metaint")
    if raw is None:
        raise CliError(
            "stream did not return icy-metaint; ICY metadata is required "
            "for the current ripping pipeline"
        )

    try:
        metaint = int(raw)
    except ValueError as exc:
        raise CliError(f"invalid icy-metaint header: {raw!r}") from exc

    if metaint <= 0:
        raise CliError(f"invalid icy-metaint header: {raw!r}")
    return metaint


def _open_stream(url: str) -> tuple[BinaryIO, Mapping[str, str]]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise CliError("stream URL must use http or https")
    if not parsed.netloc:
        raise CliError("stream URL must include a host")

    request = urllib.request.Request(
        url,
        headers={
            "Icy-MetaData": "1",
            "User-Agent": "FluxTuner-Ripper/0.1",
        },
    )

    try:
        response = urllib.request.urlopen(request, timeout=20)  # nosec B310
    except (urllib.error.URLError, ValueError) as exc:
        raise CliError(f"could not open stream: {exc}") from exc

    return response, response.headers


def _validate_args(args: argparse.Namespace) -> None:
    if args.metadata_threshold <= 0:
        raise CliError("--metadata-threshold must be greater than zero")
    if args.search_radius <= 0:
        raise CliError("--search-radius must be greater than zero")
    if not args.ffmpeg.strip():
        raise CliError("--ffmpeg must not be empty")


def _run_stream(args: argparse.Namespace) -> int:
    _validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    stream, headers = _open_stream(args.url)
    try:
        codec = _resolve_codec(args.codec, headers)
        metaint = _resolve_metaint(headers)

        ingestor = RippingStreamIngestor(
            metaint=metaint,
            codec=codec,
            ring_max_bytes=16 * 1024 * 1024,
        )
        orchestrator = RippingOrchestrator(
            window_extractor=AcousticWindowExtractor(
                search_radius_seconds=args.search_radius,
            ),
            decoder=FfmpegAcousticDecoder(
                ffmpeg_binary=args.ffmpeg,
            ),
            matcher=NearestBoundaryMatcher(
                search_radius_seconds=args.search_radius,
            ),
        )
        session = RippingSession(
            ingestor=ingestor,
            metadata_tracker=MetadataSemanticTracker(
                transient_threshold_seconds=args.metadata_threshold,
            ),
            orchestrator=orchestrator,
        )
        output_writer = SessionOutputWriter(
            ingestor=ingestor,
            directory=args.output,
            codec=codec,
        )

        print(f"codec: {codec}")
        print(f"icy-metaint: {metaint}")
        print(f"output: {args.output}")
        print("ripping stream; press Ctrl+C to stop")

        try:
            while True:
                chunk = stream.read(_CHUNK_SIZE)
                if not chunk:
                    break

                result = session.feed(chunk)

                for event in result.ingest.timed_metadata_events:
                    print(
                        f"[{event.audio_time_seconds:10.3f}s] {event.title}",
                        flush=True,
                    )

                for transition in result.transitions:
                    written = output_writer.write_transition(transition)
                    print(f"written: {written.path}", flush=True)
        except KeyboardInterrupt:
            if session.current_track is not None:
                print(
                    f"incomplete track not finalized: {session.current_track.title}",
                    file=sys.stderr,
                )
            print("stopped", file=sys.stderr)
            return 130
    finally:
        stream.close()

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _run_stream(args)
    except CliError as exc:
        parser.error(str(exc))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
