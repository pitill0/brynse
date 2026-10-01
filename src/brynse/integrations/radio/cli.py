"""Command-line entry point for the Brynse radio integration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from brynse.integrations.radio.models import TimedMetadataEvent
from brynse.integrations.radio.runner import (
    RippingRunConfig,
    RippingRunError,
    RippingRunner,
    normalized_content_type,
    open_stream_source,
    resolve_codec,
    resolve_metaint,
)
from brynse.integrations.radio.session_output import WrittenSegment, WrittenTrack
from brynse.source import StreamSource


class CliError(RuntimeError):
    """Raised for user-facing CLI failures."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brynse-radio",
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
        "--transient-exclusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude short metadata intervals explicitly marked as ads/jingles (default: enabled)",
    )
    parser.add_argument(
        "--autonomous-boundaries",
        action="store_true",
        help="Promote high-confidence autonomous acoustic boundaries (default: disabled)",
    )
    parser.add_argument(
        "--routing-trace",
        type=Path,
        help="Write shadow transition routing observations as JSONL",
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
    if not args.ffmpeg.strip():
        raise CliError("--ffmpeg must not be empty")


def _routing_trace_record(observation) -> dict[str, object]:
    context = observation.context
    evidence = context.evidence
    temporal = observation.temporal

    local_values = (
        evidence.local_discontinuity_250ms,
        evidence.local_discontinuity_500ms,
        evidence.local_discontinuity_1s,
        evidence.local_discontinuity_2s,
    )

    hypotheses_detail = []
    for hypothesis in context.hypotheses:
        geometry = hypothesis.basin_geometry
        hypotheses_detail.append(
            {
                "time_seconds": hypothesis.time_seconds,
                "sources": sorted(
                    proposal.source.value for proposal in hypothesis.proposals
                ),
                "confidence": hypothesis.confidence.value,
                "basin_start_seconds": (
                    None if geometry is None else geometry.start_seconds
                ),
                "basin_minimum_seconds": (
                    None if geometry is None else geometry.minimum_seconds
                ),
                "basin_recovery_seconds": (
                    None if geometry is None else geometry.recovery_seconds
                ),
            }
        )

    return {
        "reference_time_seconds": context.reference_time_seconds,
        "router_kind": observation.kind.value,
        "temporal_kind": temporal.kind.value,
        "incoming_start_seconds": temporal.incoming_start_seconds,
        "outgoing_end_seconds": temporal.outgoing_end_seconds,
        "fade_rise": context.fade.rise is not None,
        "fade_basins": len(context.fade.basins),
        "hypotheses": len(context.hypotheses),
        "hypotheses_detail": hypotheses_detail,
        "local_complete": all(value is not None for value in local_values),
        "local_discontinuity_250ms": evidence.local_discontinuity_250ms,
        "local_discontinuity_500ms": evidence.local_discontinuity_500ms,
        "local_discontinuity_1s": evidence.local_discontinuity_1s,
        "local_discontinuity_2s": evidence.local_discontinuity_2s,
    }


def _run_stream(args: argparse.Namespace) -> int:
    _validate_args(args)

    runner = RippingRunner(
        RippingRunConfig(
            url=args.url,
            output_directory=args.output,
            codec=args.codec,
            ffmpeg_binary=args.ffmpeg,
            metadata_threshold_seconds=args.metadata_threshold,
            transient_exclusion=args.transient_exclusion,
            autonomous_boundaries=args.autonomous_boundaries,
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

    def print_segment_written(written: WrittenSegment) -> None:
        print(f"autonomous written: {written.path}", flush=True)


    def print_incoming(observation) -> None:
        selected = (
            "-"
            if observation.selected_time_seconds is None
            else f"{observation.selected_time_seconds:.3f}"
        )
        decision = observation.decision
        print(
            "incoming:"
            f" outgoing={observation.outgoing_end_seconds:.3f}"
            f" hypotheses={len(observation.hypotheses)}"
            f" analyses={len(observation.analyses)}"
            f" selected={selected}"
            f" result={decision.kind.value}"
            f" incoming={decision.incoming_start_seconds:.3f}"
            f" outgoing_end={decision.outgoing_end_seconds:.3f}",
            flush=True,
        )

        for analysis in observation.analyses:
            sources = ",".join(
                sorted(proposal.source.value for proposal in analysis.hypothesis.proposals)
            )
            print(
                "  incoming-analysis:"
                f" time={analysis.hypothesis.time_seconds:.3f}"
                f" sources={sources}"
                f" confidence={analysis.assessment.candidate_confidence.value}",
                flush=True,
            )

    def print_shadow(analysis) -> None:
        sources = ",".join(
            sorted(proposal.source.value for proposal in analysis.hypothesis.proposals)
        )
        print(
            "shadow:"
            f" time={analysis.hypothesis.time_seconds:.3f}"
            f" sources={sources}"
            f" confidence={analysis.assessment.candidate_confidence.value}",
            flush=True,
        )

    routing_trace = None
    write_routing = None

    if args.routing_trace is not None:
        args.routing_trace.parent.mkdir(parents=True, exist_ok=True)
        try:
            routing_trace = args.routing_trace.open("w", encoding="utf-8")
        except OSError as exc:
            raise CliError(
                f"could not open routing trace {args.routing_trace}: {exc}"
            ) from exc

        def write_routing(observation) -> None:
            assert routing_trace is not None
            routing_trace.write(
                json.dumps(
                    _routing_trace_record(observation),
                    sort_keys=True,
                )
                + "\n"
            )
            routing_trace.flush()

    try:
        result = runner.run(
            on_started=print_started,
            on_metadata=print_metadata,
            on_track_written=print_written,
            on_segment_written=print_segment_written,
            on_shadow_analysis=print_shadow,
            on_incoming_observation=print_incoming,
            on_routing_observation=write_routing,
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
    finally:
        if routing_trace is not None:
            routing_trace.close()

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
