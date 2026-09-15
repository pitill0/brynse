"""Command-line entry point for source-agnostic stream segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from fluxtuner_ripper.generic_output import GenericSegmentWriter
from fluxtuner_ripper.generic_runner import GenericRunner
from fluxtuner_ripper.ingest import EncodedStreamIngestor
from fluxtuner_ripper.orchestrator import RippingOrchestrator
from fluxtuner_ripper.providers import FixedIntervalBoundaryProvider


class GenericCliError(RuntimeError):
    """Raised for user-facing generic CLI failures."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluxtuner-ripper-segment",
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
        "--interval",
        type=float,
        required=True,
        metavar="SECONDS",
        help="Fixed boundary interval in seconds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write finalized segments to this directory",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        raise GenericCliError("--interval must be greater than zero")

    if args.input != "-":
        path = Path(args.input)

        if not path.exists():
            raise GenericCliError(f"input file does not exist: {path}")

        if not path.is_file():
            raise GenericCliError(f"input path is not a file: {path}")


def _read_input(input_name: str) -> bytes:
    if input_name == "-":
        return sys.stdin.buffer.read()

    return Path(input_name).read_bytes()


def _run_generic_pipeline(
    *,
    data: bytes,
    codec: str,
    interval_seconds: float,
    output_directory: Path | None = None,
) -> dict[str, object]:
    if not data:
        raise GenericCliError("input contains no encoded data")

    ingestor = EncodedStreamIngestor(
        codec=codec,
        ring_max_bytes=len(data),
    )
    provider = FixedIntervalBoundaryProvider(
        interval_seconds=interval_seconds,
    )
    resolver = RippingOrchestrator()

    runner = GenericRunner(
        ingestor=ingestor,
        provider=provider,
        resolver=resolver,
    )

    result = runner.run((data,))

    payload: dict[str, object] = {
        "bytes_ingested": result.bytes_ingested,
        "codec": codec,
        "provider": "fixed_interval",
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
            for resolution in result.resolutions
        ],
    }

    if output_directory is not None:
        writer = GenericSegmentWriter(
            ingestor=ingestor,
            directory=output_directory,
            codec=codec,
        )
        segments = writer.write(result.resolutions)

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
        data = _read_input(args.input)
        payload = _run_generic_pipeline(
            data=data,
            codec=args.codec,
            interval_seconds=args.interval,
            output_directory=args.output_dir,
        )
    except GenericCliError as exc:
        parser.error(str(exc))

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
