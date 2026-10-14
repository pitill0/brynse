"""Descriptive acoustic-island evidence discovery.

An acoustic island is independently observed material between an outgoing
source and a later incoming handoff. Radio jingles, adverts, station IDs and
promos are examples, but this module deliberately models only acoustic
topology.

Observer discovers evidence. Router may later assign semantic ownership.
A specialist remains solely responsible for final temporal boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from brynse.boundaries import BoundaryHypothesis, BoundaryProposalSource
from brynse.bounded_insert import BoundedInsertRegionEvidence
from brynse.transition_evidence import (
    FadeRoutingEvidence,
    IncomingFadeEvidence,
    OutgoingFadeEvidence,
)

DEFAULT_SEARCH_HORIZON_SECONDS = 8.0
DEFAULT_MINIMUM_RESET_RATIO = 10.0 ** (12.0 / 20.0)


@dataclass(frozen=True)
class AcousticIslandEvidence:
    """Neutral evidence for independently bounded intermediate material."""

    incoming_handoff: BoundaryHypothesis
    terminal_decay: OutgoingFadeEvidence | None = None
    source_reset: IncomingFadeEvidence | None = None
    observed_region: BoundedInsertRegionEvidence | None = None

    def __post_init__(self) -> None:
        reset_pair = (
            self.terminal_decay is not None
            and self.source_reset is not None
        )

        if self.observed_region is None and not reset_pair:
            raise ValueError(
                'acoustic island requires an observed region or '
                'terminal-decay/source-reset evidence'
            )

        if self.terminal_decay is None and self.source_reset is not None:
            raise ValueError('source_reset requires terminal_decay')

        if self.terminal_decay is not None and self.source_reset is None:
            raise ValueError('terminal_decay requires source_reset')


class AcousticIslandDetector:
    """Discover acoustic-island topology without choosing temporal cuts."""

    def __init__(
        self,
        *,
        search_horizon_seconds: float = DEFAULT_SEARCH_HORIZON_SECONDS,
        minimum_reset_ratio: float = DEFAULT_MINIMUM_RESET_RATIO,
    ) -> None:
        if search_horizon_seconds <= 0:
            raise ValueError('search_horizon_seconds must be greater than zero')
        if minimum_reset_ratio <= 1.0:
            raise ValueError('minimum_reset_ratio must be greater than one')

        self._search_horizon_seconds = search_horizon_seconds
        self._minimum_reset_ratio = minimum_reset_ratio

    def observe(
        self,
        *,
        reference_time_seconds: float,
        fade: FadeRoutingEvidence,
        hypotheses: tuple[BoundaryHypothesis, ...],
        bounded_insert_regions: tuple[BoundedInsertRegionEvidence, ...] = (),
    ) -> tuple[AcousticIslandEvidence, ...]:
        """Return at most one conservative acoustic-island observation."""
        if reference_time_seconds < 0:
            raise ValueError('reference_time_seconds must be non-negative')

        canonical_region = self._canonical_bracketing_region(
            reference_time_seconds=reference_time_seconds,
            regions=bounded_insert_regions,
        )

        correction = self._terminal_reset_candidate(
            reference_time_seconds=reference_time_seconds,
            fade=fade,
            hypotheses=hypotheses,
            canonical_region=canonical_region,
        )

        if correction is None and canonical_region is None:
            correction = self._long_terminal_reset_candidate(
                reference_time_seconds=reference_time_seconds,
                fade=fade,
                hypotheses=hypotheses,
            )

        if canonical_region is None and correction is None:
            return ()

        if correction is None:
            assert canonical_region is not None
            return (
                AcousticIslandEvidence(
                    incoming_handoff=canonical_region.right,
                    observed_region=canonical_region,
                ),
            )

        terminal_decay, source_reset, incoming_handoff = correction

        return (
            AcousticIslandEvidence(
                incoming_handoff=incoming_handoff,
                terminal_decay=terminal_decay,
                source_reset=source_reset,
                observed_region=canonical_region,
            ),
        )

    @staticmethod
    def _canonical_bracketing_region(
        *,
        reference_time_seconds: float,
        regions: tuple[BoundedInsertRegionEvidence, ...],
    ) -> BoundedInsertRegionEvidence | None:
        candidates = tuple(
            region
            for region in regions
            if (
                region.left.time_seconds < reference_time_seconds < region.right.time_seconds
                and not region.reference_inside_left_basin
                and not region.reference_inside_right_basin
            )
        )

        if len(candidates) != 1:
            return None
        return candidates[0]

    @staticmethod
    def _proposal_sources(
        hypothesis: BoundaryHypothesis,
    ) -> set[BoundaryProposalSource]:
        return {proposal.source for proposal in hypothesis.proposals}

    @classmethod
    def _is_supported_handoff(cls, hypothesis: BoundaryHypothesis) -> bool:
        sources = cls._proposal_sources(hypothesis)
        return (
            hypothesis.basin_geometry is not None
            and BoundaryProposalSource.BASIN in sources
            and BoundaryProposalSource.STRUCTURAL in sources
        )

    @classmethod
    def _unique_supported_owner(
        cls,
        *,
        time_seconds: float,
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> BoundaryHypothesis | None:
        owners = tuple(
            hypothesis
            for hypothesis in hypotheses
            if (
                cls._is_supported_handoff(hypothesis)
                and hypothesis.basin_geometry is not None
                and hypothesis.basin_geometry.start_seconds
                <= time_seconds
                <= hypothesis.basin_geometry.recovery_seconds
            )
        )
        if len(owners) != 1:
            return None
        return owners[0]

    def _is_source_reset(self, incoming: IncomingFadeEvidence) -> bool:
        return (
            incoming.ascent_ratio >= self._minimum_reset_ratio
            and incoming.persistence_ratio >= self._minimum_reset_ratio
        )

    @staticmethod
    def _reset_strength(incoming: IncomingFadeEvidence) -> float:
        return min(incoming.ascent_ratio, incoming.persistence_ratio)

    @staticmethod
    def _post_floor_grace_seconds(outgoing: OutgoingFadeEvidence) -> float:
        duration = max(
            0.0,
            outgoing.floor_time_seconds - outgoing.start_time_seconds,
        )
        return min(2.0, max(0.5, duration * 0.25))

    def _long_terminal_reset_candidate(
        self,
        *,
        reference_time_seconds: float,
        fade: FadeRoutingEvidence,
        hypotheses: tuple[BoundaryHypothesis, ...],
    ) -> tuple[
        OutgoingFadeEvidence,
        IncomingFadeEvidence,
        BoundaryHypothesis,
    ] | None:
        """Observe a long island as a chain of locally supported edges.

        This path is deliberately narrower than ordinary terminal-reset
        observation.  It applies only when a strong terminal decay and a
        later strong reset are locally related, but the reset-to-handoff
        interval itself exceeds the ordinary search horizon.
        """

        supported = tuple(
            hypothesis
            for hypothesis in hypotheses
            if self._is_supported_handoff(hypothesis)
        )

        candidates = []

        for outgoing in fade.outgoing_fades:
            if outgoing.descent_ratio < self._minimum_reset_ratio:
                continue

            floor = outgoing.floor_time_seconds
            grace = self._post_floor_grace_seconds(outgoing)

            for incoming in fade.incoming_fades:
                if not self._is_source_reset(incoming):
                    continue

                reset = incoming.start_time_seconds
                reset_gap = reset - floor

                # Ordinary terminal-reset observation already owns resets
                # inside the post-floor grace period.
                if reset_gap <= grace:
                    continue

                # The decay and reset must still be one local acoustic event.
                if reset_gap > self._search_horizon_seconds:
                    continue

                later_handoffs = tuple(
                    hypothesis
                    for hypothesis in supported
                    if hypothesis.time_seconds > reset
                )

                for handoff in later_handoffs:
                    island_duration = handoff.time_seconds - reset

                    # A short reset-to-handoff interval belongs to the
                    # ordinary local topology.  This path exists specifically
                    # for genuinely long intermediate material.
                    if island_duration <= self._search_horizon_seconds:
                        continue

                    if (
                        abs(
                            handoff.time_seconds
                            - reference_time_seconds
                        )
                        > self._search_horizon_seconds
                    ):
                        continue

                    candidates.append(
                        (
                            abs(
                                handoff.time_seconds
                                - reference_time_seconds
                            ),
                            reset_gap,
                            island_duration,
                            outgoing,
                            incoming,
                            handoff,
                        )
                    )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        best = candidates[0]

        equally_near = tuple(
            candidate
            for candidate in candidates
            if abs(candidate[0] - best[0]) < 1e-9
        )

        if len(equally_near) != 1:
            return None

        return best[3], best[4], best[5]

    def _terminal_reset_candidate(
        self,
        *,
        reference_time_seconds: float,
        fade: FadeRoutingEvidence,
        hypotheses: tuple[BoundaryHypothesis, ...],
        canonical_region: BoundedInsertRegionEvidence | None,
    ) -> tuple[OutgoingFadeEvidence, IncomingFadeEvidence, BoundaryHypothesis] | None:
        supported = tuple(
            hypothesis
            for hypothesis in hypotheses
            if self._is_supported_handoff(hypothesis)
        )
        candidates = []

        for outgoing in fade.outgoing_fades:
            floor = outgoing.floor_time_seconds
            if abs(floor - reference_time_seconds) > self._search_horizon_seconds:
                continue

            grace = self._post_floor_grace_seconds(outgoing)
            associated_resets = tuple(
                incoming
                for incoming in fade.incoming_fades
                if (
                    outgoing.start_time_seconds
                    <= incoming.start_time_seconds
                    <= floor + grace
                    and incoming.start_time_seconds < reference_time_seconds
                    and self._is_source_reset(incoming)
                )
            )

            unexplained = tuple(
                incoming
                for incoming in associated_resets
                if self._unique_supported_owner(
                    time_seconds=incoming.start_time_seconds,
                    hypotheses=hypotheses,
                ) is None
            )
            if not unexplained:
                continue

            source_reset = min(unexplained, key=lambda incoming: incoming.start_time_seconds)
            source_strength = self._reset_strength(source_reset)

            later_owned_reset = any(
                (
                    incoming.start_time_seconds > source_reset.start_time_seconds
                    and self._is_source_reset(incoming)
                    and self._reset_strength(incoming) >= source_strength
                    and self._unique_supported_owner(
                        time_seconds=incoming.start_time_seconds,
                        hypotheses=hypotheses,
                    ) is not None
                )
                for incoming in associated_resets
            )
            if later_owned_reset:
                continue

            if canonical_region is not None:
                incoming_handoff = canonical_region.right
            else:
                anchor = max(
                    reference_time_seconds,
                    floor,
                    source_reset.established_time_seconds,
                )
                right_handoffs = tuple(
                    hypothesis
                    for hypothesis in supported
                    if hypothesis.time_seconds > anchor
                )
                if not right_handoffs:
                    continue
                incoming_handoff = min(
                    right_handoffs,
                    key=lambda hypothesis: hypothesis.time_seconds,
                )

            candidates.append(
                (source_strength, outgoing, source_reset, incoming_handoff)
            )

        if not candidates:
            return None

        _strength, terminal_decay, source_reset, incoming_handoff = max(
            candidates,
            key=lambda item: (item[0], -item[2].start_time_seconds),
        )
        return terminal_decay, source_reset, incoming_handoff
