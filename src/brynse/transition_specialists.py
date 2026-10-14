"""Temporal specialists for routed acoustic transitions.

The router decides WHO owns a transition.
A specialist decides WHERE its temporal boundaries are.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from brynse.boundaries import BoundaryConfidence, BoundaryProposalSource
from brynse.models import TemporalSplitDecision, TemporalSplitKind
from brynse.pcm_temporal_edge import PcmTemporalSearchBounds
from brynse.transition_router import (
    BoundedInsertOwnership,
    BoundedInsertRoutingProvenance,
    FadeRoutingAssessment,
    HardCutRoutingAssessment,
    InterstitialRoutingAssessment,
    TransitionKind,
    TransitionRoute,
    TransitionRouter,
    TransitionRoutingContext,
    TransitionSemanticRoles,
)


@dataclass(frozen=True)
class TransitionSpecialistInput:
    """Neutral observation material handed to one transition specialist."""

    routing: TransitionRoutingContext
    selected: Any
    profile: Any
    window: Any
    pcm8: Any
    pcm16: Any
    bounded_insert_ownership: BoundedInsertOwnership | None = None
    bounded_insert_assessment: BoundedInsertRoutingProvenance | None = None
    hard_cut_assessment: HardCutRoutingAssessment | None = None
    fade_assessment: FadeRoutingAssessment | None = None
    interstitial_assessment: InterstitialRoutingAssessment | None = None
    semantic_roles: TransitionSemanticRoles | None = None


@dataclass(frozen=True)
class TransitionSpecialistResult:
    """Temporal decision plus optional specialist-owned diagnostics."""

    temporal: TemporalSplitDecision | None
    diagnostics: Any = None


def _append_pcm_handoff_times(
    times: list[float],
    handoff,
) -> None:
    if handoff is None:
        return

    time_seconds = getattr(
        handoff,
        "time_seconds",
        None,
    )

    if isinstance(time_seconds, (int, float)):
        times.append(float(time_seconds))

    for proposal in getattr(
        handoff,
        "proposals",
        (),
    ):
        value = getattr(
            proposal,
            "time_seconds",
            None,
        )

        if isinstance(value, (int, float)):
            times.append(float(value))

    geometry = getattr(
        handoff,
        "basin_geometry",
        None,
    )

    if geometry is None:
        return

    for name in (
        "start_seconds",
        "minimum_seconds",
        "recovery_seconds",
    ):
        value = getattr(
            geometry,
            name,
            None,
        )

        if isinstance(value, (int, float)):
            times.append(float(value))


def _append_pcm_fade_times(
    times: list[float],
    fade,
) -> None:
    if fade is None:
        return

    for name in (
        "start_time_seconds",
        "floor_time_seconds",
        "established_time_seconds",
    ):
        value = getattr(
            fade,
            name,
            None,
        )

        if isinstance(value, (int, float)):
            times.append(float(value))


def _owned_pcm_edge_times(
    *,
    observation: TransitionSpecialistInput,
    edge: str,
) -> list[float]:
    """Collect semantic evidence that bounds one owned temporal edge."""

    times: list[float] = []
    roles = observation.semantic_roles

    if roles is not None:
        handoff = (
            roles.outgoing_handoff
            if edge == "outgoing"
            else roles.incoming_handoff
        )

        _append_pcm_handoff_times(
            times,
            handoff,
        )

        edge_fade = (
            roles.outgoing_edge_fade
            if edge == "outgoing"
            else roles.incoming_edge_fade
        )

        _append_pcm_fade_times(
            times,
            edge_fade,
        )

        region = roles.inserted_region

        if region is not None:
            if edge == "outgoing":
                region_edge = getattr(
                    region,
                    "left",
                    None,
                )

                if region_edge is None:
                    region_edge = getattr(
                        region,
                        "primary",
                        None,
                    )
            else:
                region_edge = getattr(
                    region,
                    "right",
                    None,
                )

                if region_edge is None:
                    region_edge = getattr(
                        region,
                        "complementary",
                        None,
                    )

            _append_pcm_handoff_times(
                times,
                region_edge,
            )

        island = roles.acoustic_island

        if island is not None:
            if edge == "outgoing":
                _append_pcm_fade_times(
                    times,
                    island.terminal_decay,
                )
                _append_pcm_fade_times(
                    times,
                    island.source_reset,
                )
            else:
                _append_pcm_handoff_times(
                    times,
                    island.incoming_handoff,
                )

    # A Router-owned FadeToFloor is a complete direct-transition topology.
    # An independently observed AcousticIsland may still remain available as
    # descriptive context, but it must not expand this specialist's PCM
    # ownership back into unrelated earlier material.
    if (
        roles is not None
        and getattr(
            roles,
            "fade_to_floor",
            None,
        )
        is not None
    ):
        return times

    # AcousticIsland may coexist with a different Router ownership.
    # It contributes bounds only; it never changes TransitionKind.
    for island in observation.routing.acoustic_islands:
        if edge == "outgoing":
            _append_pcm_fade_times(
                times,
                island.terminal_decay,
            )
            _append_pcm_fade_times(
                times,
                island.source_reset,
            )
        else:
            _append_pcm_handoff_times(
                times,
                island.incoming_handoff,
            )

    return times


def _pcm_bounds(
    times: list[float],
) -> PcmTemporalSearchBounds | None:
    if not times:
        return None

    return PcmTemporalSearchBounds(
        start_time_seconds=min(times),
        end_time_seconds=max(times),
    )


def _verify_owned_temporal_decision(
    *,
    verifier,
    observation: TransitionSpecialistInput,
    decision: TemporalSplitDecision,
    outgoing_extra_times=(),
    incoming_extra_times=(),
) -> TemporalSplitDecision:
    """Let PCM choose final timestamps without changing semantic kind."""

    if verifier is None:
        return decision

    absolute_start = observation.window.start_time_seconds

    outgoing_times = _owned_pcm_edge_times(
        observation=observation,
        edge="outgoing",
    )
    incoming_times = _owned_pcm_edge_times(
        observation=observation,
        edge="incoming",
    )

    outgoing_times.extend(
        float(value)
        for value in outgoing_extra_times
        if isinstance(value, (int, float))
    )
    incoming_times.extend(
        float(value)
        for value in incoming_extra_times
        if isinstance(value, (int, float))
    )

    def verify_edge(
        *,
        provisional: float,
        times: list[float],
    ):
        bounds = _pcm_bounds(times)
        kwargs = {}

        if bounds is not None:
            kwargs["search_bounds"] = bounds

        return verifier.verify(
            pcm=observation.pcm16,
            absolute_start_time_seconds=absolute_start,
            provisional_time_seconds=provisional,
            **kwargs,
        )

    if decision.kind is TemporalSplitKind.HARD_CUT:
        shared_times = [
            *outgoing_times,
            *incoming_times,
        ]

        if not shared_times:
            shared_times.extend(
                (
                    observation.selected.time_seconds,
                    observation.routing.reference_time_seconds,
                )
            )

        verified = verify_edge(
            provisional=decision.outgoing_end_seconds,
            times=shared_times,
        )

        if verified is None:
            return decision

        boundary = verified.verified_time_seconds

        if (
            boundary == decision.outgoing_end_seconds
            and boundary == decision.incoming_start_seconds
        ):
            return decision

        return replace(
            decision,
            outgoing_end_seconds=boundary,
            incoming_start_seconds=boundary,
        )

    outgoing_verified = verify_edge(
        provisional=decision.outgoing_end_seconds,
        times=outgoing_times,
    )
    incoming_verified = verify_edge(
        provisional=decision.incoming_start_seconds,
        times=incoming_times,
    )

    outgoing = (
        outgoing_verified.verified_time_seconds
        if outgoing_verified is not None
        else decision.outgoing_end_seconds
    )
    incoming = (
        incoming_verified.verified_time_seconds
        if incoming_verified is not None
        else decision.incoming_start_seconds
    )

    if decision.kind is TemporalSplitKind.CROSSFADE:
        valid = incoming < outgoing
    elif decision.kind is TemporalSplitKind.EXCLUSION:
        valid = outgoing < incoming
    else:
        valid = False

    if not valid:
        return decision

    if (
        outgoing == decision.outgoing_end_seconds
        and incoming == decision.incoming_start_seconds
    ):
        return decision

    return replace(
        decision,
        outgoing_end_seconds=outgoing,
        incoming_start_seconds=incoming,
    )


class TransitionSpecialist(Protocol):
    """Resolve temporal boundaries for one owned transition phenomenon."""

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        """Return the specialist's complete resolution."""
        ...


def _recover_later_reference_basin_hard_cut(
    *,
    decision: TemporalSplitDecision,
    observation: TransitionSpecialistInput,
) -> TemporalSplitDecision:
    """Recover a failed direct transition from its supported right basin.

    INTERSTITIAL and AMBIGUOUS specialists may conservatively fall back to a
    hard cut when incoming refinement cannot prove an exclusion.  If bounded
    region observation independently shows that the semantic reference lies
    inside one unique, later BASIN+STRUCTURAL right edge, that edge describes
    a stronger direct-transition boundary than the earlier fallback cut.

    This recovery deliberately applies only to an already-failed HARD_CUT.
    Successful exclusions remain owned by their original specialist.
    """

    if decision.kind is not TemporalSplitKind.HARD_CUT:
        return decision

    regions = observation.routing.bounded_insert_regions

    if len(regions) != 1:
        return decision

    region = regions[0]

    if not region.reference_inside_right_basin:
        return decision

    geometry = region.right_edge.basin_geometry

    if geometry is None:
        return decision

    sources = {
        proposal.source
        for proposal in region.right_edge.proposals
    }

    if sources != {
        BoundaryProposalSource.BASIN,
        BoundaryProposalSource.STRUCTURAL,
    }:
        return decision

    boundary = geometry.minimum_seconds

    if boundary <= decision.outgoing_end_seconds:
        return decision

    return TemporalSplitDecision(
        kind=TemporalSplitKind.HARD_CUT,
        outgoing_end_seconds=boundary,
        incoming_start_seconds=boundary,
    )


class HardCutSpecialist:
    """Resolve the acoustic boundary of a genuine abrupt transition."""

    def __init__(
        self,
        *,
        temporal_edge_verifier=None,
    ) -> None:
        self._temporal_edge_verifier = temporal_edge_verifier

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        result = self._resolve_geometry(
            observation=observation,
        )

        if result.temporal is None:
            return result

        extra_times = [
            observation.selected.time_seconds,
            observation.routing.reference_time_seconds,
        ]

        for hypothesis in observation.routing.hypotheses:
            geometry = hypothesis.basin_geometry

            if geometry is None:
                continue

            selected_inside = (
                geometry.start_seconds
                <= observation.selected.time_seconds
                <= geometry.recovery_seconds
            )
            reference_inside = (
                geometry.start_seconds
                <= observation.routing.reference_time_seconds
                <= geometry.recovery_seconds
            )

            if selected_inside or reference_inside:
                extra_times.extend(
                    (
                        geometry.start_seconds,
                        geometry.minimum_seconds,
                        geometry.recovery_seconds,
                    )
                )

        verified = _verify_owned_temporal_decision(
            verifier=self._temporal_edge_verifier,
            observation=observation,
            decision=result.temporal,
            outgoing_extra_times=extra_times,
            incoming_extra_times=extra_times,
        )

        if verified is result.temporal:
            return result

        return replace(
            result,
            temporal=verified,
        )

    def _resolve_geometry(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        boundary = self._resolve_boundary(
            selected_time_seconds=observation.selected.time_seconds,
            reference_time_seconds=(
                observation.routing.reference_time_seconds
            ),
            hypotheses=observation.routing.hypotheses,
        )

        terminal_handoff = self._refine_terminal_handoff(
            boundary=boundary,
            observation=observation,
        )

        if terminal_handoff is not None:
            boundary = terminal_handoff

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                incoming_start_seconds=boundary,
                outgoing_end_seconds=boundary,
            )
        )

    @classmethod
    def _refine_terminal_handoff(
        cls,
        *,
        boundary: float,
        observation: TransitionSpecialistInput,
    ) -> float | None:
        """Recover a corroborated acoustic handoff inside terminal basin recovery.

        The ordinary hard-cut resolver remains authoritative.  This refinement
        applies only when that resolver completed the basin containing the
        selected boundary through its recovery edge.

        If an outgoing fade terminates after that basin's minimum and before
        its recovery, and an independently observed incoming acoustic phase
        begins at exactly the same point, that shared point owns the abrupt
        handoff more precisely than the trailing basin recovery.

        Routing ownership and temporal kind remain HARD_CUT.
        """
        fade = observation.routing.fade

        if fade.rise is not None:
            return None

        selected_time_seconds = observation.selected.time_seconds
        reference_time_seconds = (
            observation.routing.reference_time_seconds
        )

        recovery_candidates = tuple(
            (
                hypothesis,
                hypothesis.basin_geometry,
            )
            for hypothesis in observation.routing.hypotheses
            if (
                hypothesis.basin_geometry is not None
                and hypothesis.basin_geometry.recovery_seconds
                == boundary
                and hypothesis.basin_geometry.start_seconds
                <= selected_time_seconds
                <= hypothesis.basin_geometry.recovery_seconds
                and selected_time_seconds
                >= hypothesis.basin_geometry.minimum_seconds
            )
        )

        if len(recovery_candidates) != 1:
            return None

        hypothesis, geometry = recovery_candidates[0]

        sources = cls._proposal_sources(hypothesis)

        if sources != {
            BoundaryProposalSource.BASIN,
            BoundaryProposalSource.STRUCTURAL,
        }:
            return None

        if reference_time_seconds < geometry.recovery_seconds:
            return None

        matching_fades = tuple(
            outgoing_fade
            for outgoing_fade in fade.outgoing_fades
            if (
                outgoing_fade.start_time_seconds
                < outgoing_fade.floor_time_seconds
                and geometry.minimum_seconds
                <= outgoing_fade.floor_time_seconds
                < geometry.recovery_seconds
            )
        )

        if len(matching_fades) != 1:
            return None

        outgoing_fade = matching_fades[0]
        handoff = outgoing_fade.floor_time_seconds

        incoming_phases = tuple(
            incoming
            for incoming in fade.incoming_fades
            if (
                incoming.start_time_seconds == handoff
                and incoming.established_time_seconds > handoff
            )
        )

        if len(incoming_phases) != 1:
            return None

        return handoff

    @classmethod
    def _resolve_boundary(
        cls,
        *,
        selected_time_seconds: float,
        reference_time_seconds: float,
        hypotheses,
    ) -> float:
        geometries = tuple(
            (
                hypothesis,
                hypothesis.basin_geometry,
            )
            for hypothesis in hypotheses
            if hypothesis.basin_geometry is not None
        )

        selected_geometry = cls._containing_geometry(
            time_seconds=selected_time_seconds,
            geometries=geometries,
        )

        if selected_geometry is not None:
            if (
                selected_time_seconds
                >= selected_geometry.minimum_seconds
            ):
                return selected_geometry.recovery_seconds

            selected_hypothesis = cls._hypothesis_for_geometry(
                geometry=selected_geometry,
                geometries=geometries,
            )
            selected_sources = cls._proposal_sources(
                selected_hypothesis,
            )

            if (
                selected_sources is not None
                and BoundaryProposalSource.STRUCTURAL
                in selected_sources
                and reference_time_seconds
                >= selected_geometry.recovery_seconds
            ):
                return selected_geometry.recovery_seconds

            if (
                selected_sources is not None
                and BoundaryProposalSource.BASIN
                in selected_sources
                and BoundaryProposalSource.STRUCTURAL
                not in selected_sources
                and selected_time_seconds
                < selected_geometry.minimum_seconds
                and selected_geometry.start_seconds
                <= reference_time_seconds
                <= selected_geometry.minimum_seconds
            ):
                return reference_time_seconds

            if (
                selected_geometry.start_seconds
                <= reference_time_seconds
                <= selected_geometry.minimum_seconds
            ):
                return selected_geometry.minimum_seconds

            return selected_time_seconds

        reference_geometry = cls._containing_geometry(
            time_seconds=reference_time_seconds,
            geometries=geometries,
        )

        if reference_geometry is not None:
            if (
                reference_time_seconds
                <= reference_geometry.minimum_seconds
            ):
                reference_hypothesis = cls._hypothesis_for_geometry(
                    geometry=reference_geometry,
                    geometries=geometries,
                )
                reference_sources = cls._proposal_sources(
                    reference_hypothesis,
                )

                if (
                    reference_sources is not None
                    and BoundaryProposalSource.BASIN
                    in reference_sources
                    and BoundaryProposalSource.STRUCTURAL
                    not in reference_sources
                ):
                    return reference_time_seconds

                return reference_geometry.minimum_seconds

            return reference_geometry.recovery_seconds

        enclosed = tuple(
            geometry
            for _, geometry in geometries
            if (
                selected_time_seconds
                < geometry.start_seconds
                and geometry.recovery_seconds
                < reference_time_seconds
            )
        )

        if len(enclosed) == 1:
            return enclosed[0].recovery_seconds

        if geometries:
            return selected_time_seconds

        if hypotheses:
            hypothesis = min(
                hypotheses,
                key=lambda item: (
                    abs(
                        item.time_seconds
                        - reference_time_seconds
                    ),
                    item.time_seconds,
                ),
            )
            return hypothesis.time_seconds

        return selected_time_seconds

    @staticmethod
    def _hypothesis_for_geometry(
        *,
        geometry,
        geometries,
    ):
        for hypothesis, candidate in geometries:
            if candidate is geometry:
                return hypothesis

        return None

    @staticmethod
    def _proposal_sources(hypothesis):
        proposals = getattr(
            hypothesis,
            "proposals",
            None,
        )

        if proposals is None:
            return None

        return {
            proposal.source
            for proposal in proposals
            if hasattr(proposal, "source")
        }

    @staticmethod
    def _containing_geometry(
        *,
        time_seconds: float,
        geometries,
    ):
        containing = tuple(
            geometry
            for _, geometry in geometries
            if (
                geometry.start_seconds
                <= time_seconds
                <= geometry.recovery_seconds
            )
        )

        if not containing:
            return None

        return min(
            containing,
            key=lambda geometry: (
                abs(
                    geometry.minimum_seconds
                    - time_seconds
                ),
                geometry.minimum_seconds,
            ),
        )


def _refine_acoustic_island_left_edge(
    *,
    decision: TemporalSplitDecision,
    island,
    region,
) -> TemporalSplitDecision | None:
    """Retract one owned insert left edge from acoustic-island evidence.

    The owning specialist has already established a valid exclusion.
    Acoustic-island evidence may only move its outgoing edge earlier when
    an independently observed source reset interrupts, or immediately
    follows, the outgoing terminal decay.

    Ownership and the incoming edge remain unchanged.
    """
    if decision.kind is not TemporalSplitKind.EXCLUSION:
        return None

    if island.observed_region is not region:
        return None

    terminal_decay = island.terminal_decay
    source_reset = island.source_reset

    if terminal_decay is None or source_reset is None:
        return None

    outgoing = min(
        terminal_decay.floor_time_seconds,
        source_reset.start_time_seconds,
    )

    if outgoing >= decision.outgoing_end_seconds:
        return None

    if outgoing >= decision.incoming_start_seconds:
        return None

    return replace(
        decision,
        outgoing_end_seconds=outgoing,
    )



def _interstitial_left_structure_precedes_basin(
    handoff,
) -> bool:
    """Return whether structural support lies wholly before the handoff basin."""

    geometry = getattr(
        handoff,
        "basin_geometry",
        None,
    )

    if geometry is None:
        return False

    if handoff.time_seconds >= geometry.start_seconds:
        return False

    structural_times = tuple(
        float(proposal.time_seconds)
        for proposal in getattr(
            handoff,
            "proposals",
            (),
        )
        if getattr(
            getattr(
                proposal,
                "source",
                None,
            ),
            "value",
            getattr(
                proposal,
                "source",
                None,
            ),
        )
        == "structural"
    )

    return (
        bool(structural_times)
        and all(
            value < geometry.start_seconds
            for value in structural_times
        )
    )


class InterstitialSpecialist:
    """Resolve track -> interstitial -> incoming-track geometry.

    Incoming refinement establishes the right edge of an interstitial region.
    Once that region exists, this specialist independently analyzes acoustic
    hypotheses to refine its left edge.
    """

    def __init__(
        self,
        *,
        incoming_pipeline,
        temporal_edge_verifier=None,
    ) -> None:
        self._incoming_pipeline = incoming_pipeline
        self._temporal_edge_verifier = temporal_edge_verifier

    def _verify_pcm_exclusion(
        self,
        *,
        decision: TemporalSplitDecision,
        observation: TransitionSpecialistInput,
    ) -> TemporalSplitDecision:
        """Compatibility hook; final PCM authority is applied by resolve()."""

        return decision

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        result = self._resolve_geometry(
            observation=observation,
        )

        if result.temporal is None:
            return result

        outgoing_extra_times = ()
        incoming_extra_times = ()

        semantic_roles = observation.semantic_roles

        if (
            result.temporal.kind is TemporalSplitKind.EXCLUSION
            and semantic_roles is not None
            and semantic_roles.acoustic_island is not None
        ):
            island = semantic_roles.acoustic_island
            region = island.observed_region

            if (
                region is not None
                and semantic_roles.inserted_region is region
                and semantic_roles.outgoing_handoff is region.left
                and semantic_roles.incoming_handoff
                is island.incoming_handoff
            ):
                reference = (
                    observation.routing.reference_time_seconds
                )

                outgoing_extra_times = (reference,)
                incoming_extra_times = (reference,)

                if _interstitial_left_structure_precedes_basin(
                    semantic_roles.outgoing_handoff
                ):
                    outgoing_extra_times = (
                        reference,
                        island.incoming_handoff.time_seconds,
                    )

        verified = _verify_owned_temporal_decision(
            verifier=self._temporal_edge_verifier,
            observation=observation,
            decision=result.temporal,
            outgoing_extra_times=outgoing_extra_times,
            incoming_extra_times=incoming_extra_times,
        )

        if verified is result.temporal:
            return result

        return replace(
            result,
            temporal=verified,
        )

    def _resolve_geometry(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        boundary = observation.selected.time_seconds

        semantic_roles = observation.semantic_roles
        fade_assessment = observation.fade_assessment

        if (
            semantic_roles is not None
            and semantic_roles.acoustic_island is not None
        ):
            island = semantic_roles.acoustic_island
            region = island.observed_region

            if (
                region is None
                and island.terminal_decay is not None
                and island.source_reset is not None
                and semantic_roles.incoming_handoff
                is island.incoming_handoff
            ):
                outgoing = min(
                    island.terminal_decay.floor_time_seconds,
                    island.source_reset.start_time_seconds,
                )
                incoming = island.incoming_handoff.time_seconds

                if outgoing < incoming:
                    reset_pair_decision = TemporalSplitDecision(
                        kind=TemporalSplitKind.EXCLUSION,
                        outgoing_end_seconds=outgoing,
                        incoming_start_seconds=incoming,
                    )

                    return TransitionSpecialistResult(
                        temporal=self._verify_pcm_exclusion(
                            decision=reset_pair_decision,
                            observation=observation,
                        ),
                    )

            if (
                region is not None
                and island.terminal_decay is not None
                and island.source_reset is not None
                and semantic_roles.inserted_region is region
                and semantic_roles.outgoing_handoff is region.left
                and semantic_roles.incoming_handoff
                is island.incoming_handoff
            ):
                initial_island = TemporalSplitDecision(
                    kind=TemporalSplitKind.EXCLUSION,
                    outgoing_end_seconds=region.left.time_seconds,
                    incoming_start_seconds=(
                        island.incoming_handoff.time_seconds
                    ),
                )

                refined_island = _refine_acoustic_island_left_edge(
                    decision=initial_island,
                    island=island,
                    region=region,
                )

                if refined_island is not None:
                    return TransitionSpecialistResult(
                        temporal=refined_island,
                    )

        if (
            semantic_roles is not None
            and semantic_roles.outgoing_edge_fade is not None
            and semantic_roles.incoming_edge_fade is not None
        ):
            outgoing = (
                semantic_roles.outgoing_edge_fade.floor_time_seconds
            )
            incoming = (
                semantic_roles.incoming_edge_fade.start_time_seconds
            )

            if incoming == outgoing:
                kind = TemporalSplitKind.HARD_CUT
            elif incoming < outgoing:
                kind = TemporalSplitKind.CROSSFADE
            else:
                kind = TemporalSplitKind.EXCLUSION

            return TransitionSpecialistResult(
                temporal=TemporalSplitDecision(
                    kind=kind,
                    outgoing_end_seconds=outgoing,
                    incoming_start_seconds=incoming,
                ),
            )

        if (
            semantic_roles is not None
            and semantic_roles.outgoing_edge_fade is not None
            and semantic_roles.incoming_edge_fade is None
            and semantic_roles.incoming_handoff is not None
            and observation.interstitial_assessment is not None
            and semantic_roles.incoming_handoff
            in observation.interstitial_assessment.supported_handoffs
        ):
            incoming_handoff = semantic_roles.incoming_handoff
            incoming_geometry = incoming_handoff.basin_geometry

            if incoming_geometry is not None:
                outgoing = (
                    semantic_roles.outgoing_edge_fade
                    .floor_time_seconds
                )
                incoming = max(
                    incoming_handoff.time_seconds,
                    incoming_geometry.recovery_seconds,
                )

                if outgoing < incoming:
                    return TransitionSpecialistResult(
                        temporal=TemporalSplitDecision(
                            kind=TemporalSplitKind.EXCLUSION,
                            outgoing_end_seconds=outgoing,
                            incoming_start_seconds=incoming,
                        ),
                    )

        if (
            semantic_roles is not None
            and semantic_roles.outgoing_handoff is not None
            and semantic_roles.outgoing_handoff
            is semantic_roles.incoming_handoff
            and fade_assessment is not None
            and fade_assessment.outgoing_association is not None
            and fade_assessment.incoming_association is not None
        ):
            shared = semantic_roles.outgoing_handoff
            outgoing_association = fade_assessment.outgoing_association
            incoming_association = fade_assessment.incoming_association

            if (
                outgoing_association.hypothesis is shared
                and incoming_association.hypothesis is shared
            ):
                outgoing = (
                    outgoing_association.fade.floor_time_seconds
                )
                incoming = (
                    incoming_association.fade.start_time_seconds
                )

                if incoming == outgoing:
                    kind = TemporalSplitKind.HARD_CUT
                elif incoming < outgoing:
                    kind = TemporalSplitKind.CROSSFADE
                else:
                    kind = TemporalSplitKind.EXCLUSION

                return TransitionSpecialistResult(
                    temporal=TemporalSplitDecision(
                        kind=kind,
                        outgoing_end_seconds=outgoing,
                        incoming_start_seconds=incoming,
                    ),
                )

        initial = TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            incoming_start_seconds=boundary,
            outgoing_end_seconds=boundary,
        )

        if self._incoming_pipeline is None:
            return TransitionSpecialistResult(
                temporal=initial,
            )

        refined = self._incoming_pipeline.refine(
            decision=initial,
            window=observation.window,
            pcm8=observation.pcm8,
            pcm16=observation.pcm16,
            hypotheses=observation.routing.hypotheses,
        )

        if refined.kind is not TemporalSplitKind.EXCLUSION:
            semantic_roles = observation.semantic_roles

            recovered = self._recover_supported_exclusion(
                selected_time_seconds=boundary,
                hypotheses=observation.routing.hypotheses,
                supported_handoffs=(
                    observation.interstitial_assessment.supported_handoffs
                    if observation.interstitial_assessment is not None
                    else None
                ),
                outgoing_handoff=(
                    semantic_roles.outgoing_handoff
                    if semantic_roles is not None
                    else None
                ),
                incoming_handoff=(
                    semantic_roles.incoming_handoff
                    if semantic_roles is not None
                    else None
                ),
            )

            if recovered is not None:
                return TransitionSpecialistResult(
                    temporal=self._verify_pcm_exclusion(
                        decision=recovered,
                        observation=observation,
                    ),
                )

            recovered_direct = _recover_later_reference_basin_hard_cut(
                decision=refined,
                observation=observation,
            )

            return TransitionSpecialistResult(
                temporal=self._verify_pcm_exclusion(
                    decision=recovered_direct,
                    observation=observation,
                ),
            )

        if (
            semantic_roles is not None
            and semantic_roles.outgoing_handoff is not None
            and semantic_roles.incoming_handoff is not None
            and semantic_roles.outgoing_handoff
            is not semantic_roles.incoming_handoff
            and observation.interstitial_assessment is not None
        ):
            recovered = self._recover_supported_exclusion(
                selected_time_seconds=boundary,
                hypotheses=observation.routing.hypotheses,
                supported_handoffs=(
                    observation.interstitial_assessment.supported_handoffs
                ),
                outgoing_handoff=semantic_roles.outgoing_handoff,
                incoming_handoff=semantic_roles.incoming_handoff,
            )

            if recovered is not None:
                return TransitionSpecialistResult(
                    temporal=self._verify_pcm_exclusion(
                        decision=recovered,
                        observation=observation,
                    ),
                )

        phase_refined = self._refine_dominant_incoming_basin(
            decision=refined,
            reference_time_seconds=(
                observation.routing.reference_time_seconds
            ),
            hypotheses=observation.routing.hypotheses,
        )

        if phase_refined is not None:
            return TransitionSpecialistResult(
                temporal=self._verify_pcm_exclusion(
                    decision=phase_refined,
                    observation=observation,
                ),
            )

        analyze_hypotheses = getattr(
            self._incoming_pipeline,
            "analyze_hypotheses",
            None,
        )
        if analyze_hypotheses is None:
            return TransitionSpecialistResult(
                temporal=self._verify_pcm_exclusion(
                    decision=refined,
                    observation=observation,
                ),
            )

        analyses = analyze_hypotheses(
            hypotheses=observation.routing.hypotheses,
            window=observation.window,
            pcm16=observation.pcm16,
        )

        outgoing = self._discover_outgoing(
            selected_time_seconds=boundary,
            incoming_start_seconds=refined.incoming_start_seconds,
            analyses=analyses,
            outgoing_handoff=(
                observation.semantic_roles.outgoing_handoff
                if observation.semantic_roles is not None
                else None
            ),
        )

        if outgoing is None:
            terminal_refined = self._refine_terminal_reference_basin_fade(
                decision=refined,
                observation=observation,
            )

            terminal_decision = (
                terminal_refined
                if terminal_refined is not None
                else refined
            )

            return TransitionSpecialistResult(
                temporal=self._verify_pcm_exclusion(
                    decision=terminal_decision,
                    observation=observation,
                ),
            )

        final = TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=refined.incoming_start_seconds,
        )

        return TransitionSpecialistResult(
            temporal=self._verify_pcm_exclusion(
                decision=final,
                observation=observation,
            ),
        )

    @staticmethod
    def _refine_terminal_reference_basin_fade(
        *,
        decision: TemporalSplitDecision,
        observation: TransitionSpecialistInput,
    ) -> TemporalSplitDecision | None:
        """Extend an interstitial outgoing edge through one terminal fade.

        This is deliberately a fallback after ordinary HIGH-confidence
        outgoing discovery has abstained.  A unique bounded region must put
        the semantic reference inside its right basin while the left edge,
        but not the right edge, owns the strongest structural event.

        The current outgoing boundary must already lie inside one observed
        descending fade, and that fade must terminate inside the same right
        basin.  The basin therefore independently confirms that the fade
        floor still belongs to the outgoing track.
        """
        from brynse.boundaries import BoundaryProposalSource

        if decision.kind is not TemporalSplitKind.EXCLUSION:
            return None

        assessment = observation.interstitial_assessment

        if assessment is not None:
            matching_fades = tuple(
                outgoing_fade
                for outgoing_fade in assessment.terminal_fade_candidates
                if (
                    outgoing_fade.start_time_seconds
                    <= decision.outgoing_end_seconds
                    < outgoing_fade.floor_time_seconds
                    < decision.incoming_start_seconds
                )
            )

            if len(matching_fades) != 1:
                return None

            return TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=(
                    matching_fades[0].floor_time_seconds
                ),
                incoming_start_seconds=decision.incoming_start_seconds,
            )

        fade = observation.routing.fade

        if fade.rise is not None:
            return None

        regions = observation.routing.bounded_insert_regions

        if len(regions) != 1:
            return None

        region = regions[0]

        if not region.reference_inside_right_basin:
            return None

        if region.reference_inside_left_basin:
            return None

        if region.structural_only_interior_count != 0:
            return None

        right_geometry = region.right_edge.basin_geometry

        if right_geometry is None:
            return None

        structural_strengths = tuple(
            proposal.strength
            for hypothesis in observation.routing.hypotheses
            for proposal in hypothesis.proposals
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        if not structural_strengths:
            return None

        context_peak = max(structural_strengths)

        left_strengths = tuple(
            proposal.strength
            for proposal in region.left_edge.proposals
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )
        right_strengths = tuple(
            proposal.strength
            for proposal in region.right_edge.proposals
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        left_peak = (
            bool(left_strengths)
            and max(left_strengths) == context_peak
        )
        right_peak = (
            bool(right_strengths)
            and max(right_strengths) == context_peak
        )

        if not left_peak or right_peak:
            return None

        matching_fades = tuple(
            outgoing_fade
            for outgoing_fade in fade.outgoing_fades
            if (
                outgoing_fade.start_time_seconds
                <= decision.outgoing_end_seconds
                < outgoing_fade.floor_time_seconds
                < decision.incoming_start_seconds
                and right_geometry.start_seconds
                <= outgoing_fade.floor_time_seconds
                <= right_geometry.recovery_seconds
            )
        )

        if len(matching_fades) != 1:
            return None

        boundary = matching_fades[0].floor_time_seconds

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=boundary,
            incoming_start_seconds=decision.incoming_start_seconds,
        )

    @staticmethod
    def _refine_dominant_incoming_basin(
        *,
        decision: TemporalSplitDecision,
        reference_time_seconds: float,
        hypotheses,
    ) -> TemporalSplitDecision | None:
        from brynse.boundaries import BoundaryProposalSource

        structural_strengths = tuple(
            proposal.strength
            for hypothesis in hypotheses
            for proposal in getattr(
                hypothesis,
                "proposals",
                (),
            )
            if (
                proposal.source is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        if not structural_strengths:
            return None

        context_peak = max(structural_strengths)

        matching = []

        for hypothesis in hypotheses:
            geometry = hypothesis.basin_geometry

            if geometry is None:
                continue

            if (
                geometry.recovery_seconds
                != decision.incoming_start_seconds
            ):
                continue

            strengths = tuple(
                proposal.strength
                for proposal in getattr(
                    hypothesis,
                    "proposals",
                    (),
                )
                if (
                    proposal.source
                    is BoundaryProposalSource.STRUCTURAL
                    and proposal.strength is not None
                )
            )

            if not strengths:
                continue

            if max(strengths) != context_peak:
                continue

            matching.append(
                (
                    hypothesis,
                    geometry,
                )
            )

        if len(matching) != 1:
            return None

        hypothesis, geometry = matching[0]

        basin_proposal_times = tuple(
            proposal.time_seconds
            for proposal in getattr(
                hypothesis,
                "proposals",
                (),
            )
            if (
                proposal.source
                is BoundaryProposalSource.BASIN
            )
        )

        if len(basin_proposal_times) != 1:
            return None

        if (
            reference_time_seconds
            < basin_proposal_times[0]
        ):
            return None

        if not (
            decision.outgoing_end_seconds
            < geometry.start_seconds
            < decision.incoming_start_seconds
        ):
            return None

        dominant_structural_times = tuple(
            proposal.time_seconds
            for proposal in getattr(
                hypothesis,
                "proposals",
                (),
            )
            if (
                proposal.source
                is BoundaryProposalSource.STRUCTURAL
                and proposal.strength == context_peak
            )
        )

        outgoing = geometry.start_seconds

        if geometry.recovery_seconds in dominant_structural_times:
            outgoing = geometry.minimum_seconds

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=decision.incoming_start_seconds,
        )

    @staticmethod
    def _recover_supported_exclusion(
        *,
        selected_time_seconds: float,
        hypotheses,
        supported_handoffs=None,
        outgoing_handoff=None,
        incoming_handoff=None,
    ) -> TemporalSplitDecision | None:
        from brynse.boundaries import BoundaryProposalSource

        complete_role_pair = (
            outgoing_handoff is not None
            and incoming_handoff is not None
            and outgoing_handoff is not incoming_handoff
        )

        if complete_role_pair:
            left = outgoing_handoff
            right = incoming_handoff
        else:
            ordered = tuple(
                sorted(
                    hypotheses,
                    key=lambda hypothesis: hypothesis.time_seconds,
                )
            )

            before = tuple(
                hypothesis
                for hypothesis in ordered
                if hypothesis.time_seconds < selected_time_seconds
            )
            after = tuple(
                hypothesis
                for hypothesis in ordered
                if hypothesis.time_seconds > selected_time_seconds
            )

            if not before or not after:
                return None

            left = before[-1]
            right = after[0]

        if supported_handoffs is None:
            def supported(hypothesis) -> bool:
                sources = {
                    proposal.source
                    for proposal in hypothesis.proposals
                }

                return (
                    hypothesis.basin_geometry is not None
                    and BoundaryProposalSource.BASIN in sources
                    and BoundaryProposalSource.STRUCTURAL in sources
                )
        else:
            def supported(hypothesis) -> bool:
                return hypothesis in supported_handoffs

        if not supported(left) or not supported(right):
            return None

        if left.time_seconds >= right.time_seconds:
            return None

        left_distance = (
            selected_time_seconds
            - left.time_seconds
        )
        right_distance = (
            right.time_seconds
            - selected_time_seconds
        )

        if left_distance <= right_distance:
            outgoing = min(
                left.time_seconds,
                left.basin_geometry.recovery_seconds,
            )
        else:
            outgoing = selected_time_seconds

        incoming = max(
            right.time_seconds,
            right.basin_geometry.recovery_seconds,
        )

        if outgoing >= incoming:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=incoming,
        )

    @staticmethod
    def _discover_outgoing(
        *,
        selected_time_seconds: float,
        incoming_start_seconds: float,
        analyses,
        outgoing_handoff=None,
    ) -> float | None:
        if outgoing_handoff is not None:
            matching = tuple(
                analysis
                for analysis in analyses
                if analysis.hypothesis is outgoing_handoff
            )

            if len(matching) != 1:
                return None

            analysis = matching[0]

            if (
                analysis.assessment.candidate_confidence
                is not BoundaryConfidence.HIGH
            ):
                return None

            if analysis.hypothesis.time_seconds >= incoming_start_seconds:
                return None

            return analysis.hypothesis.time_seconds

        supported = tuple(
            analysis.hypothesis.time_seconds
            for analysis in analyses
            if (
                analysis.assessment.candidate_confidence
                is BoundaryConfidence.HIGH
                and analysis.hypothesis.time_seconds
                < incoming_start_seconds
            )
        )

        if not supported:
            return None

        return min(
            supported,
            key=lambda time_seconds: (
                abs(time_seconds - selected_time_seconds),
                time_seconds,
            ),
        )


class BoundedInsertSpecialist:
    """Resolve temporal geometry for acoustically bounded inserted material."""

    def __init__(
        self,
        *,
        temporal_edge_verifier=None,
    ) -> None:
        self._temporal_edge_verifier = temporal_edge_verifier

    @staticmethod
    def _proposal_sources(hypothesis):
        return {
            proposal.source
            for proposal in hypothesis.proposals
        }

    @classmethod
    def _is_supported_edge(cls, hypothesis) -> bool:
        from brynse.boundaries import BoundaryProposalSource

        sources = cls._proposal_sources(hypothesis)

        return (
            BoundaryProposalSource.BASIN in sources
            and BoundaryProposalSource.STRUCTURAL in sources
            and hypothesis.basin_geometry is not None
        )

    @classmethod
    def _is_structural_only(cls, hypothesis) -> bool:
        from brynse.boundaries import BoundaryProposalSource

        return cls._proposal_sources(hypothesis) == {
            BoundaryProposalSource.STRUCTURAL
        }

    @classmethod
    def propose_structural_bridge_cut(
        cls,
        region,
    ):
        """Collapse one structural bridge that owns the semantic handoff."""
        interior = region.interior_hypotheses

        if len(interior) != 1:
            return None

        bridge = interior[0]

        if not cls._is_structural_only(bridge):
            return None

        nearest = min(
            (
                region.left,
                bridge,
                region.right,
            ),
            key=lambda hypothesis: (
                abs(
                    hypothesis.time_seconds
                    - region.reference_time_seconds
                ),
                hypothesis.time_seconds,
            ),
        )

        if nearest is not bridge:
            return None

        boundary = bridge.time_seconds

        return TemporalSplitDecision(
            kind=TemporalSplitKind.HARD_CUT,
            outgoing_end_seconds=boundary,
            incoming_start_seconds=boundary,
        )

    @classmethod
    def propose_symmetric_outer_extent(
        cls,
        *,
        region,
        hypotheses,
        semantic_roles=None,
    ):
        """Resolve symmetric outer geometry inside already assigned roles."""
        anchor = region.left

        if semantic_roles is not None:
            predecessor = semantic_roles.outgoing_handoff
            incoming_handoff = semantic_roles.incoming_handoff

            if predecessor is None or incoming_handoff is None:
                return None

            if semantic_roles.inserted_region is not region:
                return None
        else:
            previous_supported = tuple(
                hypothesis
                for hypothesis in hypotheses
                if (
                    hypothesis.time_seconds < anchor.time_seconds
                    and cls._is_supported_edge(hypothesis)
                )
            )

            if not previous_supported:
                return None

            predecessor = max(
                previous_supported,
                key=lambda hypothesis: hypothesis.time_seconds,
            )
            incoming_handoff = region.right

        if predecessor.time_seconds >= anchor.time_seconds:
            return None

        interior = tuple(
            hypothesis
            for hypothesis in hypotheses
            if (
                predecessor.time_seconds
                < hypothesis.time_seconds
                < anchor.time_seconds
            )
        )

        if not interior:
            return None

        if not all(
            cls._is_structural_only(hypothesis)
            for hypothesis in interior
        ):
            return None

        outgoing = predecessor.time_seconds
        incoming = incoming_handoff.time_seconds

        if outgoing >= incoming:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=incoming,
        )

    @staticmethod
    def propose_symmetric_phase_aware(
        region,
        *,
        reference_time_seconds=None,
        semantic_roles=None,
    ):
        """Propose a phase-aware symmetric exclusion without applying it."""
        if reference_time_seconds is None:
            reference_time_seconds = region.reference_time_seconds

        if (
            semantic_roles is not None
            and semantic_roles.inserted_region is region
            and semantic_roles.outgoing_handoff is None
            and semantic_roles.incoming_handoff is region.right
            and region.structural_only_interior_count > 0
        ):
            left_geometry = region.left_edge.basin_geometry
            right_geometry = region.right_edge.basin_geometry

            if left_geometry is not None and right_geometry is not None:
                outgoing = left_geometry.start_seconds
                incoming = right_geometry.recovery_seconds

                if outgoing < incoming:
                    return TemporalSplitDecision(
                        kind=TemporalSplitKind.EXCLUSION,
                        outgoing_end_seconds=outgoing,
                        incoming_start_seconds=incoming,
                    )

        outgoing = region.left.time_seconds

        geometry = region.left_edge.basin_geometry

        if geometry is not None:
            outgoing = min(
                outgoing,
                geometry.recovery_seconds,
            )

        later_reference_basins = tuple(
            evidence
            for evidence in region.left.evidence
            if (
                evidence.basin_start_seconds is not None
                and evidence.basin_recovery_seconds is not None
                and evidence.proposal.time_seconds
                > region.left.time_seconds
                and evidence.basin_start_seconds
                <= reference_time_seconds
                <= evidence.basin_recovery_seconds
            )
        )

        if len(later_reference_basins) == 1:
            outgoing = (
                later_reference_basins[0]
                .basin_recovery_seconds
            )

        incoming = region.right.time_seconds

        if outgoing >= incoming:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=incoming,
        )

    @classmethod
    def _refine_symmetric_predecessor_terminal_fade(
        cls,
        *,
        decision: TemporalSplitDecision,
        region,
        observation: TransitionSpecialistInput,
        semantic_roles=None,
    ) -> TemporalSplitDecision | None:
        """Recover a terminal outgoing fade before a symmetric insert.

        The symmetric region still owns the inserted material and therefore
        keeps its resolved incoming edge.  This refinement only retracts an
        outgoing edge when an immediately preceding supported basin contains
        both the end of one observed outgoing fade and the start of an
        independently observed incoming acoustic phase.

        This deliberately refines temporal geometry only; Router ownership
        and the symmetric insert extent remain unchanged.
        """
        from brynse.boundaries import BoundaryProposalSource

        if decision.kind is not TemporalSplitKind.EXCLUSION:
            return None

        fade = observation.routing.fade

        if fade.rise is not None:
            return None

        if len(region.interior_hypotheses) != 1:
            return None

        if not cls._is_structural_only(
            region.interior_hypotheses[0]
        ):
            return None

        if (
            region.reference_inside_left_basin
            or region.reference_inside_right_basin
        ):
            return None

        if not (
            region.left.time_seconds
            <= observation.routing.reference_time_seconds
            <= region.right.time_seconds
        ):
            return None

        left_geometry = region.left_edge.basin_geometry

        if left_geometry is None:
            return None

        if (
            decision.outgoing_end_seconds
            != left_geometry.recovery_seconds
        ):
            return None

        structural_strengths = tuple(
            proposal.strength
            for hypothesis in observation.routing.hypotheses
            for proposal in hypothesis.proposals
            if (
                proposal.source
                is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        if not structural_strengths:
            return None

        context_peak = max(structural_strengths)

        left_strengths = tuple(
            proposal.strength
            for proposal in region.left_edge.proposals
            if (
                proposal.source
                is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )
        right_strengths = tuple(
            proposal.strength
            for proposal in region.right_edge.proposals
            if (
                proposal.source
                is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        left_peak = (
            bool(left_strengths)
            and max(left_strengths) == context_peak
        )
        right_peak = (
            bool(right_strengths)
            and max(right_strengths) == context_peak
        )

        if left_peak or right_peak:
            return None

        if semantic_roles is not None:
            if semantic_roles.inserted_region is not region:
                return None

            predecessor = semantic_roles.outgoing_handoff

            if predecessor is None:
                return None
        else:
            ordered_before_left = tuple(
                hypothesis
                for hypothesis in sorted(
                    observation.routing.hypotheses,
                    key=lambda hypothesis: hypothesis.time_seconds,
                )
                if hypothesis.time_seconds < region.left.time_seconds
            )

            if not ordered_before_left:
                return None

            predecessor = ordered_before_left[-1]

        if not cls._is_supported_edge(predecessor):
            return None

        if predecessor.time_seconds >= region.left.time_seconds:
            return None

        predecessor_geometry = predecessor.basin_geometry

        if (
            predecessor_geometry.recovery_seconds
            >= left_geometry.start_seconds
        ):
            return None

        matching_fades = tuple(
            outgoing_fade
            for outgoing_fade in fade.outgoing_fades
            if (
                outgoing_fade.start_time_seconds
                < outgoing_fade.floor_time_seconds
                < decision.outgoing_end_seconds
                and predecessor_geometry.start_seconds
                <= outgoing_fade.floor_time_seconds
                <= predecessor_geometry.recovery_seconds
            )
        )

        if len(matching_fades) != 1:
            return None

        outgoing_fade = matching_fades[0]
        boundary = outgoing_fade.floor_time_seconds

        incoming_phase = tuple(
            incoming
            for incoming in fade.incoming_fades
            if (
                incoming.start_time_seconds == boundary
                and incoming.established_time_seconds > boundary
            )
        )

        if len(incoming_phase) != 1:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=boundary,
            incoming_start_seconds=decision.incoming_start_seconds,
        )

    @classmethod
    def propose_asymmetric_outer_extent(
        cls,
        *,
        asymmetric_region,
        symmetric_regions,
        hypotheses,
        semantic_roles=None,
    ):
        """Resolve asymmetric exterior geometry inside assigned roles."""
        anchor = asymmetric_region.primary

        ordered = tuple(
            sorted(
                hypotheses,
                key=lambda hypothesis: hypothesis.time_seconds,
            )
        )

        if semantic_roles is not None:
            if semantic_roles.inserted_region is not asymmetric_region:
                return None

            predecessor = semantic_roles.outgoing_handoff
            incoming_handoff = semantic_roles.incoming_handoff

            if predecessor is None or incoming_handoff is None:
                return None
        else:
            previous_supported = tuple(
                hypothesis
                for hypothesis in ordered
                if (
                    hypothesis.time_seconds < anchor.time_seconds
                    and cls._is_supported_edge(hypothesis)
                )
            )

            if not previous_supported:
                return None

            predecessor = max(
                previous_supported,
                key=lambda hypothesis: hypothesis.time_seconds,
            )

            matching_symmetric = tuple(
                region
                for region in symmetric_regions
                if region.left.time_seconds == anchor.time_seconds
            )

            if len(matching_symmetric) != 1:
                return None

            incoming_handoff = matching_symmetric[0].right

        if predecessor.time_seconds >= anchor.time_seconds:
            return None

        interior = tuple(
            hypothesis
            for hypothesis in ordered
            if (
                predecessor.time_seconds
                < hypothesis.time_seconds
                < anchor.time_seconds
            )
        )

        if not all(
            cls._is_structural_only(hypothesis)
            for hypothesis in interior
        ):
            return None

        outgoing = predecessor.time_seconds
        incoming = incoming_handoff.time_seconds

        if outgoing >= incoming:
            return None

        return TemporalSplitDecision(
            kind=TemporalSplitKind.EXCLUSION,
            outgoing_end_seconds=outgoing,
            incoming_start_seconds=incoming,
        )

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        result = self._resolve_geometry(
            observation=observation,
        )

        if result.temporal is None:
            return result

        verified = _verify_owned_temporal_decision(
            verifier=self._temporal_edge_verifier,
            observation=observation,
            decision=result.temporal,
        )

        if verified is result.temporal:
            return result

        return replace(
            result,
            temporal=verified,
        )

    def _resolve_geometry(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        """Resolve the complete bounded-insert provisional geometry."""
        symmetric = observation.routing.bounded_insert_regions
        asymmetric = (
            observation.routing.bounded_insert_asymmetric_regions
        )
        ownership = observation.bounded_insert_ownership
        provenance = observation.bounded_insert_assessment
        semantic_roles = observation.semantic_roles

        if provenance is not None:
            ownership = provenance.ownership

        if ownership is BoundedInsertOwnership.SYMMETRIC:
            if provenance is not None:
                region = provenance.symmetric_region

                if region is None:
                    return TransitionSpecialistResult(
                        temporal=None,
                    )
            else:
                if len(symmetric) != 1:
                    return TransitionSpecialistResult(
                        temporal=None,
                    )

                region = symmetric[0]

            if (
                semantic_roles is not None
                and semantic_roles.inserted_region is not region
            ):
                return TransitionSpecialistResult(
                    temporal=None,
                )

            def finalize_symmetric(
                decision: TemporalSplitDecision,
            ) -> TransitionSpecialistResult:
                islands = observation.routing.acoustic_islands

                if len(islands) == 1:
                    refined = _refine_acoustic_island_left_edge(
                        decision=decision,
                        island=islands[0],
                        region=region,
                    )

                    if refined is not None:
                        return TransitionSpecialistResult(
                            temporal=refined,
                        )

                return TransitionSpecialistResult(
                    temporal=decision,
                )

            proposal = self.propose_structural_bridge_cut(
                region,
            )

            if proposal is not None:
                return finalize_symmetric(proposal)

            if observation.routing.fade.rise is None:
                proposal = self.propose_symmetric_outer_extent(
                    region=region,
                    hypotheses=observation.routing.hypotheses,
                    semantic_roles=semantic_roles,
                )

                if proposal is not None:
                    return finalize_symmetric(proposal)

            proposal = self.propose_symmetric_phase_aware(
                region,
                reference_time_seconds=(
                    observation.routing.reference_time_seconds
                ),
                semantic_roles=semantic_roles,
            )

            if proposal is not None:
                terminal_refined = (
                    self._refine_symmetric_predecessor_terminal_fade(
                        decision=proposal,
                        region=region,
                        observation=observation,
                        semantic_roles=semantic_roles,
                    )
                )

                resolved = (
                    terminal_refined
                    if terminal_refined is not None
                    else proposal
                )

                return finalize_symmetric(resolved)

            fallback = self._resolve_symmetric_fallback(
                region
            )

            if fallback.temporal is None:
                return fallback

            return finalize_symmetric(
                fallback.temporal,
            )

        if ownership is BoundedInsertOwnership.ASYMMETRIC:
            if provenance is not None:
                region = provenance.asymmetric_region

                if region is None:
                    return TransitionSpecialistResult(
                        temporal=None,
                    )
            else:
                if len(asymmetric) != 1:
                    return TransitionSpecialistResult(
                        temporal=None,
                    )

                region = asymmetric[0]

            if (
                semantic_roles is not None
                and semantic_roles.inserted_region is not region
            ):
                return TransitionSpecialistResult(
                    temporal=None,
                )

            proposal = None

            if observation.routing.fade.rise is None:
                proposal = self.propose_asymmetric_outer_extent(
                    asymmetric_region=region,
                    symmetric_regions=symmetric,
                    hypotheses=observation.routing.hypotheses,
                    semantic_roles=semantic_roles,
                )

            if proposal is not None:
                return TransitionSpecialistResult(
                    temporal=proposal,
                )

            return self._resolve_asymmetric_fallback(
                region,
                fade=observation.routing.fade,
            )

        # Compatibility path for direct specialist use without
        # Router provenance.  Preserve the pre-Phase-2 behavior.
        if len(symmetric) == 1:
            return self._resolve_symmetric_fallback(
                symmetric[0]
            )

        if symmetric:
            return TransitionSpecialistResult(
                temporal=None,
            )

        if len(asymmetric) != 1:
            return TransitionSpecialistResult(
                temporal=None,
            )

        return self._resolve_asymmetric_fallback(
            asymmetric[0]
        )

    @staticmethod
    def _resolve_symmetric_fallback(
        region,
    ) -> TransitionSpecialistResult:
        outgoing = region.left.time_seconds
        incoming = region.right.time_seconds

        if outgoing >= incoming:
            return TransitionSpecialistResult(
                temporal=None,
            )

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=outgoing,
                incoming_start_seconds=incoming,
            )
        )

    @staticmethod
    def _resolve_asymmetric_fallback(
        region,
        *,
        fade=None,
    ) -> TransitionSpecialistResult:
        primary_geometry = region.primary_edge.basin_geometry

        if primary_geometry is None:
            return TransitionSpecialistResult(
                temporal=None,
            )

        outgoing = primary_geometry.recovery_seconds

        from brynse.boundaries import BoundaryProposalSource

        basin_proposal_times = tuple(
            proposal.time_seconds
            for proposal in region.primary.proposals
            if proposal.source is BoundaryProposalSource.BASIN
        )

        structural_proposals = tuple(
            proposal
            for proposal in region.primary.proposals
            if (
                proposal.source
                is BoundaryProposalSource.STRUCTURAL
                and proposal.strength is not None
            )
        )

        if (
            len(basin_proposal_times) == 1
            and structural_proposals
        ):
            strongest_structural = max(
                structural_proposals,
                key=lambda proposal: proposal.strength,
            )
            basin_time = basin_proposal_times[0]

            if (
                strongest_structural.time_seconds
                < region.primary.time_seconds
                < basin_time
                < primary_geometry.recovery_seconds
            ):
                outgoing = basin_time

        # Phase 4f:
        # A single structural proposal immediately preceding the
        # primary basin can describe the true outer-left extent of
        # an asymmetric insert when independent incoming-phase
        # evidence becomes established between that structural
        # onset and the basin entry.
        #
        # This refines temporal geometry only.  Router ownership
        # remains unchanged, and direct/legacy specialist use
        # receives fade=None and therefore preserves its existing
        # fallback behavior.
        if (
            fade is not None
            and fade.rise is None
            and len(structural_proposals) == 1
        ):
            structural = structural_proposals[0]

            if structural.time_seconds < primary_geometry.start_seconds:
                incoming_phase_reaches_basin = any(
                    (
                        structural.time_seconds
                        <= incoming.established_time_seconds
                        <= primary_geometry.start_seconds
                    )
                    for incoming in fade.incoming_fades
                )

                if incoming_phase_reaches_basin:
                    outgoing = structural.time_seconds

        complementary_geometry = (
            region.complementary_edge.basin_geometry
        )

        incoming = (
            region.complementary.time_seconds
            if complementary_geometry is None
            else complementary_geometry.recovery_seconds
        )

        if outgoing >= incoming:
            return TransitionSpecialistResult(
                temporal=None,
            )

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.EXCLUSION,
                outgoing_end_seconds=outgoing,
                incoming_start_seconds=incoming,
            )
        )


class LegacyAmbiguousSpecialist:
    """Preserve neutral ownership while verifying one shared temporal edge.

    AMBIGUOUS deliberately has no acoustic phenomenon ownership.  It therefore
    does not call legacy temporal resolution, incoming refinement, or geometry
    recovery.

    The semantic reference is only a provisional compatibility anchor.  When
    a shared-edge PCM verifier is available, this specialist may retract that
    anchor to an earlier locally verified shared edge without changing
    transition ownership or temporal kind.
    """

    def __init__(
        self,
        *,
        shared_edge_verifier=None,
    ) -> None:
        self._shared_edge_verifier = shared_edge_verifier

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> TransitionSpecialistResult:
        boundary = observation.routing.reference_time_seconds
        verified = None

        if self._shared_edge_verifier is not None:
            verified = self._shared_edge_verifier.verify(
                pcm=observation.pcm16,
                absolute_start_time_seconds=(
                    observation.window.start_time_seconds
                ),
                provisional_time_seconds=boundary,
            )

            if verified is not None:
                boundary = verified.verified_time_seconds

        return TransitionSpecialistResult(
            temporal=TemporalSplitDecision(
                kind=TemporalSplitKind.HARD_CUT,
                outgoing_end_seconds=boundary,
                incoming_start_seconds=boundary,
            ),
            diagnostics=verified,
        )


class RoutedTransitionResolver:
    """Route one neutral observation to exactly one temporal specialist."""

    def __init__(
        self,
        *,
        router: TransitionRouter,
        hard_cut,
        fade,
        interstitial,
        bounded_insert,
        ambiguous,
    ) -> None:
        self._router = router
        self._specialists = {
            TransitionKind.HARD_CUT: hard_cut,
            TransitionKind.FADE: fade,
            TransitionKind.INTERSTITIAL: interstitial,
            TransitionKind.BOUNDED_INSERT: bounded_insert,
            TransitionKind.AMBIGUOUS: ambiguous,
        }

    def resolve(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> tuple[TransitionKind, TransitionSpecialistResult]:
        """Route once, then delegate to exactly one temporal specialist."""
        kind, result, _route = self.resolve_with_route(
            observation=observation,
        )
        return kind, result

    def resolve_with_route(
        self,
        *,
        observation: TransitionSpecialistInput,
    ) -> tuple[
        TransitionKind,
        TransitionSpecialistResult,
        TransitionRoute | None,
    ]:
        """Resolve and expose the exact Router-v2 route used for dispatch."""
        route = None

        route_resolver = getattr(
            self._router,
            "route",
            None,
        )

        routed_observation = observation

        if callable(route_resolver):
            route = route_resolver(
                context=observation.routing,
            )

            if not isinstance(route, TransitionRoute):
                raise TypeError(
                    "transition router route() must return TransitionRoute"
                )

            kind = route.kind
            bounded_insert_ownership = route.bounded_insert_ownership
            bounded_insert_assessment = route.bounded_insert
            hard_cut_assessment = route.hard_cut
            fade_assessment = route.fade_assessment
            interstitial_assessment = route.interstitial_assessment
            semantic_roles = route.semantic_roles
        else:
            # Compatibility path for third-party and test routers that still
            # implement only the original classify() contract.
            kind = self._router.classify(
                context=observation.routing,
            )
            bounded_insert_ownership = None
            bounded_insert_assessment = None
            hard_cut_assessment = None
            fade_assessment = None
            interstitial_assessment = None
            semantic_roles = None

            if kind is TransitionKind.BOUNDED_INSERT:
                ownership_resolver = getattr(
                    self._router,
                    "bounded_insert_ownership",
                    None,
                )

                if ownership_resolver is not None:
                    bounded_insert_ownership = ownership_resolver(
                        context=observation.routing,
                    )

        if (
            kind is TransitionKind.BOUNDED_INSERT
            and bounded_insert_ownership is not None
        ):
            routed_observation = replace(
                observation,
                bounded_insert_ownership=bounded_insert_ownership,
                bounded_insert_assessment=bounded_insert_assessment,
            )

        if (
            kind is TransitionKind.HARD_CUT
            and hard_cut_assessment is not None
        ):
            routed_observation = replace(
                routed_observation,
                hard_cut_assessment=hard_cut_assessment,
            )

        if (
            kind is TransitionKind.FADE
            and fade_assessment is not None
        ):
            routed_observation = replace(
                routed_observation,
                fade_assessment=fade_assessment,
            )

        if (
            kind is TransitionKind.INTERSTITIAL
            and interstitial_assessment is not None
        ):
            routed_observation = replace(
                routed_observation,
                interstitial_assessment=interstitial_assessment,
            )

        if semantic_roles is not None:
            routed_observation = replace(
                routed_observation,
                semantic_roles=semantic_roles,
            )

        specialist = self._specialists[kind]

        result = specialist.resolve(
            observation=routed_observation,
        )

        return kind, result, route
