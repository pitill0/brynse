"""Command-line entry point for source-agnostic stream segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from brynse.external_boundaries import (
    ExternalBoundaryParseError,
    load_external_boundaries,
)
from brynse.generic_output import GenericSegmentWriter, MaterializedSegment
from brynse.generic_runner import GenericRunner
from brynse.ingest import EncodedStreamIngestor
from brynse.orchestrator import CandidateResolution, DefaultCandidateResolver
from brynse.output import SegmentFileWriteError, SegmentFinalizeError
from brynse.providers import (
    BoundaryProvider,
    ExternalBoundaryProvider,
    FixedIntervalBoundaryProvider,
    ManualBoundaryProvider,
)
from brynse.source import BinaryIOStreamSource, StreamSource
from brynse.spooling_ingest import create_safe_spooling_ingestor
from brynse.streaming_runner import StreamingGenericRunner
from brynse.streaming_sink import StreamingSegmentSink

_STREAMING_RING_MAX_BYTES = 16 * 1024 * 1024
_STREAMING_SETTLE_SECONDS = 8.0


class GenericCliError(RuntimeError):
    """Raised for user-facing generic CLI failures."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brynse",
        description="Segment encoded streams using source-agnostic boundary providers.",
    )
    parser.add_argument(
        "input",
        help="Encoded input file, or '-' to read from stdin",
    )
    parser.add_argument(
        "--codec",
        choices=("mp3", "aac"),
        required=True,
        help="Encoded stream codec",
    )
    parser.add_argument(
        "--provider",
        choices=("fixed", "manual", "external"),
        default="fixed",
        help="Boundary provider to use (default: fixed)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        metavar="SECONDS",
        help="Fixed boundary interval in seconds",
    )
    parser.add_argument(
        "--boundary",
        type=float,
        action="append",
        default=[],
        metavar="SECONDS",
        help="Explicit manual boundary time; may be repeated",
    )
    parser.add_argument(
        "--boundaries-file",
        type=Path,
        help="External boundary input in JSON or JSONL format",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write finalized segments to this directory",
    )
    parser.add_argument(
        "--min-tail",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Merge a final tail shorter than this into the previous segment (default: 1.0)",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.provider == "fixed":
        if args.interval is None:
            raise GenericCliError("--interval is required when --provider=fixed")
        if args.interval <= 0:
            raise GenericCliError("--interval must be greater than zero")

    if args.provider == "manual":
        if not args.boundary:
            raise GenericCliError("at least one --boundary is required when --provider=manual")
        if any(value < 0 for value in args.boundary):
            raise GenericCliError("--boundary values must be non-negative")

    if args.provider == "external" and args.boundaries_file is None:
        raise GenericCliError("--boundaries-file is required when --provider=external")

    if args.min_tail <= 0:
        raise GenericCliError("--min-tail must be greater than zero")

    if args.input != "-":
        path = Path(args.input)

        if not path.exists():
            raise GenericCliError(f"input file does not exist: {path}")

        if not path.is_file():
            raise GenericCliError(f"input path is not a file: {path}")


class _NonClosingStreamSource:
    def __init__(self, stream: object) -> None:
        self._stream = stream

    @property
    def metadata(self) -> dict[str, str]:
        return {}

    def read(self, max_bytes: int) -> bytes:
        return self._stream.read(max_bytes)  # type: ignore[attr-defined]

    def close(self) -> None:
        pass


def _open_input_source(input_value: str) -> StreamSource:
    if input_value == "-":
        return _NonClosingStreamSource(sys.stdin.buffer)

    path = Path(input_value)
    return BinaryIOStreamSource(path.open("rb"))


def _iter_input(
    input_value: str,
    *,
    chunk_size: int = 64 * 1024,
) -> Iterator[bytes]:
    """Yield encoded input incrementally from a StreamSource."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    source = _open_input_source(input_value)

    try:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        source.close()


def _read_input(input_name: str) -> bytes:
    if input_name == "-":
        return sys.stdin.buffer.read()

    return Path(input_name).read_bytes()


def _run_generic_pipeline(
    *,
    data: bytes | None = None,
    chunks: Iterable[bytes] | None = None,
    codec: str,
    provider_name: str = "fixed",
    interval_seconds: float | None = None,
    boundary_times_seconds: tuple[float, ...] = (),
    boundaries_file: Path | None = None,
    output_directory: Path | None = None,
    minimum_tail_seconds: float = 1.0,
) -> dict[str, object]:
    if data is not None and chunks is not None:
        raise GenericCliError("provide either data or chunks, not both")

    if data is not None:
        if not data:
            raise GenericCliError("input contains no encoded data")
        input_chunks: Iterable[bytes] = (data,)
        ring_max_bytes = len(data)
    elif chunks is not None:
        iterator = iter(chunks)

        for first_chunk in iterator:
            if first_chunk:
                break
        else:
            raise GenericCliError("input contains no encoded data")

        def incremental_chunks() -> Iterator[bytes]:
            yield first_chunk
            yield from iterator

        input_chunks = incremental_chunks()
        ring_max_bytes = _STREAMING_RING_MAX_BYTES
    else:
        raise GenericCliError("input contains no encoded data")

    ingestor = EncodedStreamIngestor(
        codec=codec,
        ring_max_bytes=ring_max_bytes,
    )
    provider: BoundaryProvider

    if provider_name == "fixed":
        if interval_seconds is None:
            raise GenericCliError("--interval is required when --provider=fixed")
        provider = FixedIntervalBoundaryProvider(
            interval_seconds=interval_seconds,
        )
    elif provider_name == "manual":
        if not boundary_times_seconds:
            raise GenericCliError("at least one --boundary is required when --provider=manual")
        provider = ManualBoundaryProvider(
            boundary_times_seconds=boundary_times_seconds,
        )
    elif provider_name == "external":
        if boundaries_file is None:
            raise GenericCliError("--boundaries-file is required when --provider=external")

        try:
            external_boundaries = load_external_boundaries(boundaries_file)
        except ExternalBoundaryParseError as exc:
            raise GenericCliError(str(exc)) from exc

        provider = ExternalBoundaryProvider(
            boundaries=external_boundaries,
        )
    else:
        raise GenericCliError(f"unsupported provider: {provider_name}")
    resolver = DefaultCandidateResolver()

    resolutions: tuple[CandidateResolution, ...]
    segments: tuple[MaterializedSegment, ...] | None = None

    if output_directory is not None and chunks is not None:
        spooling_ingestor = create_safe_spooling_ingestor(
            ingestor=ingestor,
            spool_directory=output_directory,
        )
        streaming_runner = StreamingGenericRunner(
            ingestor=spooling_ingestor,
            resolver=resolver,
            settle_seconds=_STREAMING_SETTLE_SECONDS,
            provider=provider,
        )
        sink = StreamingSegmentSink(
            ingestor=ingestor,
            directory=output_directory,
            codec=codec,
            spool=spooling_ingestor.spool,
            initial_start_source=spooling_ingestor,
        )

        bytes_ingested = 0
        completed: list[CandidateResolution] = []
        materialized: list[MaterializedSegment] = []

        try:
            for chunk in input_chunks:
                bytes_ingested += len(chunk)

                for streaming_result in streaming_runner.feed(chunk):
                    if streaming_result.resolution is None:
                        continue

                    completed.append(streaming_result.resolution)
                    materialized.append(sink.accept(streaming_result.resolution))

            eof_resolutions: list[CandidateResolution] = []

            for streaming_result in streaming_runner.finalize():
                if streaming_result.resolution is None:
                    continue

                eof_resolutions.append(streaming_result.resolution)

            for index, resolution in enumerate(eof_resolutions):
                completed.append(resolution)

                is_last_resolution = index == len(eof_resolutions) - 1

                if is_last_resolution:
                    frames = ingestor.timeline.frames
                    if frames:
                        last_frame = frames[-1]
                        stream_end_time_seconds = (
                            last_frame.time_seconds + last_frame.samples / last_frame.sample_rate
                        )
                        tail_seconds = max(
                            0.0,
                            stream_end_time_seconds - resolution.temporal.incoming_start_seconds,
                        )

                        if tail_seconds < minimum_tail_seconds:
                            continue

                materialized.append(sink.accept(resolution))

            tail = sink.finalize()
            if tail is not None:
                materialized.append(tail)
        finally:
            spooling_ingestor.spool.close()

        resolutions = tuple(completed)
        segments = tuple(materialized)
    else:
        batch_runner = GenericRunner(
            ingestor=ingestor,
            provider=provider,
            resolver=resolver,
        )

        batch_result = batch_runner.run(input_chunks)
        bytes_ingested = batch_result.bytes_ingested
        resolutions = batch_result.resolutions

        if output_directory is not None:
            writer = GenericSegmentWriter(
                ingestor=ingestor,
                directory=output_directory,
                codec=codec,
                minimum_tail_seconds=minimum_tail_seconds,
            )
            segments = writer.write(resolutions)

    payload: dict[str, object] = {
        "bytes_ingested": bytes_ingested,
        "codec": codec,
        "provider": ("fixed_interval" if provider_name == "fixed" else provider_name),
        "interval_seconds": interval_seconds,
        "boundaries": [
            {
                "requested_time_seconds": resolution.candidate.time_seconds,
                "source": resolution.candidate.source,
                "resolved_time_seconds": resolution.temporal.incoming_start_seconds,
                "incoming_start_offset": resolution.split.incoming_start,
                "outgoing_end_offset": resolution.split.outgoing_end,
                "split_kind": resolution.split.kind.value,
            }
            for resolution in resolutions
        ],
    }

    if segments is not None:
        payload["segments"] = [
            {
                "index": segment.index,
                "start_offset": segment.start_offset,
                "end_offset": segment.end_offset,
                "path": str(segment.path),
            }
            for segment in segments
        ]

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
        chunks = _iter_input(args.input)
        payload = _run_generic_pipeline(
            chunks=chunks,
            codec=args.codec,
            provider_name=args.provider,
            interval_seconds=args.interval,
            boundary_times_seconds=tuple(args.boundary),
            boundaries_file=args.boundaries_file,
            output_directory=args.output_dir,
            minimum_tail_seconds=args.min_tail,
        )
    except GenericCliError as exc:
        parser.error(str(exc))
    except (SegmentFinalizeError, SegmentFileWriteError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    try:
        print(json.dumps(payload, indent=2, sort_keys=True))
    except BrokenPipeError:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
