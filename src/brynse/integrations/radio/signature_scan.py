"""Offline acoustic signature recognition for captured radio sessions.

Usage:
    python -m brynse.integrations.radio.signature_scan SESSION \
        --catalog signatures.json --json matches.json

Read-only with respect to audio and radio routing. Recognized times
refer to acoustic probe positions, NOT safe boundaries for track cuts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from brynse.integrations.radio.recurrence_scan import decode_session_audio
from brynse.integrations.radio.replay import ReplaySessionError
from brynse.signature_catalog import load_catalog
from brynse.signature_matcher import scan_catalog


def _threshold(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid threshold") from exc

    if not math.isfinite(number) or not 0 < number <= 1:
        raise argparse.ArgumentTypeError("threshold must be finite and in (0, 1]")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize stored jingles in a captured radio session."
    )
    parser.add_argument(
        "session",
        type=Path,
        help="Captured session directory",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Versioned acoustic signature catalog",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_output",
        help="Create a new machine-readable results file",
    )
    parser.add_argument(
        "--threshold",
        type=_threshold,
        default=0.90,
        help="Minimum correlation for both probes (default: 0.90)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        if not args.session.is_dir():
            raise ValueError("session directory does not exist")

        if args.json_output is not None and args.json_output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.json_output}")

        catalog = load_catalog(args.catalog)
        pcm = decode_session_audio(args.session)

        duration_seconds = pcm.sample_count / pcm.sample_rate
        matches = scan_catalog(
            pcm,
            catalog,
            threshold=args.threshold,
        )

        document = {
            "schema_version": 1,
            "mode": "offline_acoustic_signature_recognition",
            "session": str(args.session),
            "station_id": catalog.station_id,
            "catalog": str(args.catalog),
            "duration_seconds": duration_seconds,
            "threshold": args.threshold,
            "boundary_status": "not_assessed",
            "audio_modified": False,
            "matches": [
                {
                    "signature_id": match.signature_id,
                    "time_seconds": match.start_seconds,
                    "correlation_first": match.correlation_first,
                    "correlation_second": match.correlation_second,
                    "boundary_status": "not_assessed",
                }
                for match in matches
            ],
        }

        if args.json_output is not None:
            with args.json_output.open("x", encoding="utf-8") as handle:
                json.dump(
                    document,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.write("\n")

    except (
        ValueError,
        OSError,
        ReplaySessionError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Station: {catalog.station_id} | "
        f"Signatures: {len(catalog.signatures)} | "
        f"Matches: {len(matches)}"
    )

    for match in matches:
        print(
            f"  {match.start_seconds:10.3f}s  "
            f"{match.signature_id}  "
            f"corr={match.correlation_first:.4f}/"
            f"{match.correlation_second:.4f}"
        )

    if not matches:
        print("No acoustic signatures matched.")

    print("Recognition only: no track boundaries were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
