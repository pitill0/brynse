"""Command-line entry point for the Brynse radio integration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from brynse.integrations.radio.models import TimedMetadataEvent
from brynse.integrations.radio.naming import METADATA_ORDERS
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
    parser.add_argument(
        "--session-capture",
        type=Path,
        help=(
            "Capture the exact raw ICY input and resolved metadata "
            "for deterministic offline replay"
        ),
    )
    parser.add_argument(
        "--metadata-order",
        choices=METADATA_ORDERS,
        default="preserve",
        help=(
            "Declared artist/title order in source metadata; "
            "used only for output filenames (default: preserve)"
        ),
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
    outgoing_selection = observation.outgoing_fade_selection
    incoming_selection = observation.incoming_fade_selection
    bounded_insert = observation.bounded_insert
    bounded_insert_routing = observation.bounded_insert_routing
    fade_routing = observation.fade_routing
    interstitial_routing = observation.interstitial_routing
    hard_cut_routing = observation.hard_cut_routing
    route = observation.route
    semantic_roles = (
        None
        if route is None
        else route.semantic_roles
    )
    fade_rise = context.fade.rise

    pcm_boundary_scan = observation.pcm_boundary_scan

    def pcm_boundary_detail(item):
        return {
            "time_seconds": item.time_seconds,
            "horizons": [
                {
                    "horizon_seconds": horizon.horizon_seconds,
                    "pre_rms": horizon.pre_rms,
                    "post_rms": horizon.post_rms,
                    "energy_change": horizon.energy_change,
                    "spectral_change": horizon.spectral_change,
                }
                for horizon in item.features.horizons
            ],
            "delta_from_outgoing_seconds": round(
                item.time_seconds - temporal.outgoing_end_seconds,
                6,
            ),
            "delta_from_incoming_seconds": round(
                item.time_seconds - temporal.incoming_start_seconds,
                6,
            ),
        }

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
                "proposals_detail": [
                    {
                        "time_seconds": proposal.time_seconds,
                        "source": proposal.source.value,
                        "strength": proposal.strength,
                    }
                    for proposal in hypothesis.proposals
                ],
                "evidence_detail": [
                    {
                        "proposal_time_seconds": item.proposal.time_seconds,
                        "proposal_source": item.proposal.source.value,
                        "proposal_strength": item.proposal.strength,
                        "basin_depth": item.basin_depth,
                        "basin_start_seconds": item.basin_start_seconds,
                        "basin_recovery_seconds": item.basin_recovery_seconds,
                        "local_change": item.local_change,
                        "persistent_change": item.persistent_change,
                        "structural_novelty": item.structural_novelty,
                        "metadata_support": item.metadata_support,
                    }
                    for item in hypothesis.evidence
                ],
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

    outgoing_fades_detail = [
        {
            "start_time_seconds": fade.start_time_seconds,
            "floor_time_seconds": fade.floor_time_seconds,
            "rise_time_seconds": fade.rise_time_seconds,
            "start_rms": fade.start_rms,
            "floor_rms": fade.floor_rms,
            "descent_ratio": fade.descent_ratio,
            "descent_consistency": fade.descent_consistency,
        }
        for fade in context.fade.outgoing_fades
    ]

    incoming_fades_detail = [
        {
            "start_time_seconds": fade.start_time_seconds,
            "established_time_seconds": fade.established_time_seconds,
            "start_rms": fade.start_rms,
            "established_rms": fade.established_rms,
            "ascent_ratio": fade.ascent_ratio,
            "ascent_consistency": fade.ascent_consistency,
            "persistence_ratio": fade.persistence_ratio,
        }
        for fade in context.fade.incoming_fades
    ]

    acoustic_islands_detail = [
        {
            "incoming_handoff_seconds": (
                island.incoming_handoff.time_seconds
            ),
            "terminal_decay_start_seconds": (
                None
                if island.terminal_decay is None
                else island.terminal_decay.start_time_seconds
            ),
            "terminal_decay_floor_seconds": (
                None
                if island.terminal_decay is None
                else island.terminal_decay.floor_time_seconds
            ),
            "source_reset_start_seconds": (
                None
                if island.source_reset is None
                else island.source_reset.start_time_seconds
            ),
            "source_reset_established_seconds": (
                None
                if island.source_reset is None
                else island.source_reset.established_time_seconds
            ),
            "observed_region_left_seconds": (
                None
                if island.observed_region is None
                else island.observed_region.left.time_seconds
            ),
            "observed_region_right_seconds": (
                None
                if island.observed_region is None
                else island.observed_region.right.time_seconds
            ),
        }
        for island in context.acoustic_islands
    ]

    fade_geometries_detail = [
        {
            "outgoing_floor_seconds": geometry.outgoing_floor_seconds,
            "incoming_start_seconds": geometry.incoming_start_seconds,
            "separation_seconds": round(geometry.separation_seconds, 6),
            "overlap_seconds": round(geometry.overlap_seconds, 6),
            "gap_seconds": round(geometry.gap_seconds, 6),
            "outgoing_descent_consistency": (
                geometry.outgoing.descent_consistency
            ),
            "incoming_ascent_consistency": (
                geometry.incoming.ascent_consistency
            ),
        }
        for geometry in context.fade.geometries
    ]

    return {
        "reference_time_seconds": context.reference_time_seconds,
        "router_kind": observation.kind.value,
        "acoustic_island_count": len(context.acoustic_islands),
        "acoustic_island_detail": acoustic_islands_detail,
        "semantic_outgoing_handoff_seconds": (
            None
            if (
                semantic_roles is None
                or semantic_roles.outgoing_handoff is None
            )
            else semantic_roles.outgoing_handoff.time_seconds
        ),
        "semantic_incoming_handoff_seconds": (
            None
            if (
                semantic_roles is None
                or semantic_roles.incoming_handoff is None
            )
            else semantic_roles.incoming_handoff.time_seconds
        ),
        "semantic_inserted_region_kind": (
            None
            if (
                semantic_roles is None
                or semantic_roles.inserted_region is None
                or route is None
                or route.bounded_insert_ownership is None
            )
            else route.bounded_insert_ownership.value
        ),
        "fade_routing_candidate": (
            None
            if fade_routing is None
            else fade_routing.legacy_candidate
        ),
        "fade_routing_two_sided_support": (
            None
            if fade_routing is None
            else fade_routing.has_two_sided_support
        ),
        "fade_routing_outgoing_association": (
            None
            if fade_routing is None
            else fade_routing.outgoing_association is not None
        ),
        "fade_routing_incoming_association": (
            None
            if fade_routing is None
            else fade_routing.incoming_association is not None
        ),
        "fade_routing_outgoing_support_seconds": (
            None
            if (
                fade_routing is None
                or fade_routing.outgoing_association is None
            )
            else fade_routing.outgoing_association.hypothesis.time_seconds
        ),
        "fade_routing_incoming_support_seconds": (
            None
            if (
                fade_routing is None
                or fade_routing.incoming_association is None
            )
            else fade_routing.incoming_association.hypothesis.time_seconds
        ),
        "interstitial_routing_candidate": (
            None
            if interstitial_routing is None
            else interstitial_routing.legacy_candidate
        ),
        "interstitial_routing_handoff_count": (
            None
            if interstitial_routing is None
            else len(interstitial_routing.routing_handoffs)
        ),
        "interstitial_supported_handoff_count": (
            None
            if interstitial_routing is None
            else len(interstitial_routing.supported_handoffs)
        ),
        "interstitial_absorbed_right_edge_fade": (
            None
            if interstitial_routing is None
            else interstitial_routing.absorbed_right_edge_fade
        ),
        "interstitial_terminal_fade_support": (
            None
            if interstitial_routing is None
            else interstitial_routing.has_terminal_fade_support
        ),
        "bounded_insert_ownership": (
            None
            if (
                bounded_insert_routing is None
                or bounded_insert_routing.ownership is None
            )
            else bounded_insert_routing.ownership.value
        ),
        "bounded_insert_symmetric_candidate": (
            None
            if bounded_insert_routing is None
            else bounded_insert_routing.symmetric_assessment.candidate
        ),
        "bounded_insert_asymmetric_candidate": (
            None
            if bounded_insert_routing is None
            else bounded_insert_routing.asymmetric_assessment.candidate
        ),
        "hard_cut_fallback_eligible": (
            None
            if hard_cut_routing is None
            else hard_cut_routing.fallback_eligible
        ),
        "hard_cut_supported_handoff_count": (
            None
            if hard_cut_routing is None
            else len(hard_cut_routing.supported_handoffs)
        ),
        "temporal_kind": temporal.kind.value,
        "incoming_start_seconds": temporal.incoming_start_seconds,
        "outgoing_end_seconds": temporal.outgoing_end_seconds,
        "pcm_boundary_scan_count": len(pcm_boundary_scan),
        "pcm_boundary_scan_detail": [
            pcm_boundary_detail(item)
            for item in pcm_boundary_scan
        ],
        "fade_rise": fade_rise is not None,
        "fade_rise_time_seconds": (
            None
            if fade_rise is None
            else fade_rise.time_seconds
        ),
        "fade_rise_jump_ratio": (
            None
            if fade_rise is None
            else fade_rise.jump_ratio
        ),
        "fade_rise_persistence_ratio": (
            None
            if fade_rise is None
            else fade_rise.persistence_ratio
        ),
        "fade_rise_after_global_ratio": (
            None
            if fade_rise is None
            else fade_rise.after_global_ratio
        ),
        "fade_basins": len(context.fade.basins),
        "fade_outgoing_count": len(context.fade.outgoing_fades),
        "fade_incoming_count": len(context.fade.incoming_fades),
        "fade_geometry_count": len(context.fade.geometries),
        "fade_outgoing_selected": outgoing_selection is not None,
        "fade_outgoing_selected_floor_seconds": (
            None
            if outgoing_selection is None
            else outgoing_selection.fade.floor_time_seconds
        ),
        "fade_outgoing_selected_support_seconds": (
            None
            if outgoing_selection is None
            else outgoing_selection.hypothesis.time_seconds
        ),
        "fade_outgoing_selected_distance_seconds": (
            None
            if outgoing_selection is None
            else round(outgoing_selection.support_distance_seconds, 6)
        ),
        "fade_incoming_selected": incoming_selection is not None,
        "fade_incoming_selected_start_seconds": (
            None
            if incoming_selection is None
            else incoming_selection.fade.start_time_seconds
        ),
        "fade_incoming_selected_established_seconds": (
            None
            if incoming_selection is None
            else incoming_selection.fade.established_time_seconds
        ),
        "fade_incoming_selected_support_seconds": (
            None
            if incoming_selection is None
            else incoming_selection.hypothesis.time_seconds
        ),
        "fade_incoming_selected_distance_seconds": (
            None
            if incoming_selection is None
            else round(incoming_selection.support_distance_seconds, 6)
        ),
        "fade_outgoing_detail": outgoing_fades_detail,
        "fade_incoming_detail": incoming_fades_detail,
        "fade_geometry_detail": fade_geometries_detail,
        "hypotheses": len(context.hypotheses),
        "hypotheses_detail": hypotheses_detail,
        "bounded_insert_count": len(context.bounded_insert_regions),
        "bounded_insert_routing_candidate": (
            None
            if bounded_insert is None
            else bounded_insert.candidate
        ),
        "bounded_insert_routing_related_region_count": (
            None
            if bounded_insert is None
            else bounded_insert.related_region_count
        ),
        "bounded_insert_routing_unique_region": (
            None
            if bounded_insert is None
            else bounded_insert.unique_region
        ),
        "bounded_insert_routing_left_structural_peak": (
            None
            if bounded_insert is None
            else bounded_insert.left_is_context_structural_peak
        ),
        "bounded_insert_routing_right_structural_peak": (
            None
            if bounded_insert is None
            else bounded_insert.right_is_context_structural_peak
        ),
        "bounded_insert_routing_reference_precedes_left": (
            None
            if bounded_insert is None
            else (
                bounded_insert
                .reference_precedes_left_support_inside_basin
            )
        ),
        "bounded_insert_routing_structural_only_interior_count": (
            None
            if bounded_insert is None
            else bounded_insert.structural_only_interior_count
        ),
        "bounded_insert_detail": [
            {
                "reference_time_seconds": region.reference_time_seconds,
                "left_support_seconds": region.left.time_seconds,
                "right_support_seconds": region.right.time_seconds,
                "support_span_seconds": region.support_span_seconds,
                "acoustic_gap_seconds": region.acoustic_gap_seconds,
                "reference_inside_left_basin": (
                    region.reference_inside_left_basin
                ),
                "reference_inside_right_basin": (
                    region.reference_inside_right_basin
                ),
                "reference_offset_from_left_support_seconds": (
                    region.reference_offset_from_left_support_seconds
                ),
                "reference_offset_to_right_support_seconds": (
                    region.reference_offset_to_right_support_seconds
                ),
                "interior_hypotheses": len(region.interior_hypotheses),
                "structural_only_interior_count": (
                    region.structural_only_interior_count
                ),
                "interior_hypotheses_detail": [
                    {
                        "time_seconds": hypothesis.time_seconds,
                        "sources": sorted(
                            proposal.source.value
                            for proposal in hypothesis.proposals
                        ),
                    }
                    for hypothesis in region.interior_hypotheses
                ],
                "left_edge": {
                    "confidence": region.left_edge.confidence.value,
                    "proposals": [
                        {
                            "time_seconds": proposal.time_seconds,
                            "source": proposal.source.value,
                            "strength": proposal.strength,
                        }
                        for proposal in region.left_edge.proposals
                    ],
                    "evidence": [
                        {
                            "proposal_time_seconds": item.proposal.time_seconds,
                            "proposal_source": item.proposal.source.value,
                            "proposal_strength": item.proposal.strength,
                            "basin_depth": item.basin_depth,
                            "basin_start_seconds": item.basin_start_seconds,
                            "basin_recovery_seconds": (
                                item.basin_recovery_seconds
                            ),
                            "local_change": item.local_change,
                            "persistent_change": item.persistent_change,
                            "structural_novelty": item.structural_novelty,
                            "metadata_support": item.metadata_support,
                        }
                        for item in region.left_edge.evidence
                    ],
                },
                "right_edge": {
                    "confidence": region.right_edge.confidence.value,
                    "proposals": [
                        {
                            "time_seconds": proposal.time_seconds,
                            "source": proposal.source.value,
                            "strength": proposal.strength,
                        }
                        for proposal in region.right_edge.proposals
                    ],
                    "evidence": [
                        {
                            "proposal_time_seconds": item.proposal.time_seconds,
                            "proposal_source": item.proposal.source.value,
                            "proposal_strength": item.proposal.strength,
                            "basin_depth": item.basin_depth,
                            "basin_start_seconds": item.basin_start_seconds,
                            "basin_recovery_seconds": (
                                item.basin_recovery_seconds
                            ),
                            "local_change": item.local_change,
                            "persistent_change": item.persistent_change,
                            "structural_novelty": item.structural_novelty,
                            "metadata_support": item.metadata_support,
                        }
                        for item in region.right_edge.evidence
                    ],
                },
                "left_basin_start_seconds": (
                    region.left_geometry.start_seconds
                ),
                "left_basin_minimum_seconds": (
                    region.left_geometry.minimum_seconds
                ),
                "left_basin_recovery_seconds": (
                    region.left_geometry.recovery_seconds
                ),
                "right_basin_start_seconds": (
                    region.right_geometry.start_seconds
                ),
                "right_basin_minimum_seconds": (
                    region.right_geometry.minimum_seconds
                ),
                "right_basin_recovery_seconds": (
                    region.right_geometry.recovery_seconds
                ),
            }
            for region in context.bounded_insert_regions
        ],
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
            metadata_order=args.metadata_order,
            transient_exclusion=args.transient_exclusion,
            autonomous_boundaries=args.autonomous_boundaries,
            session_capture_directory=args.session_capture,
        ),
        stream_opener=_open_stream,
    )

    def print_started(codec: str, metaint: int) -> None:
        print(f"codec: {codec}")
        print(f"icy-metaint: {metaint}")
        print(f"output: {args.output}")
        if args.session_capture is not None:
            print(f"session-capture: {args.session_capture}")
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
