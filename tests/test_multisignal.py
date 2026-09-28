from types import SimpleNamespace

import numpy as np
import pytest

from fluxtuner_ripper.models import AcousticWindow, DecodedPcm
from fluxtuner_ripper.multisignal import (
    AcousticMultiSignalCandidateBuilder,
    FamilyRankCandidate,
    MinimaxBoundarySelector,
    MultiSignalCandidate,
    MultiSignalCandidateResolver,
    assign_family_rrf,
    build_union_candidates,
    forward_novelty,
)


def test_minimax_prefers_candidate_supported_by_both_rankings() -> None:
    candidates = (
        MultiSignalCandidate(
            time_seconds=1.0,
            family_rrf=0.9,
            forward_novelty=0.1,
        ),
        MultiSignalCandidate(
            time_seconds=2.0,
            family_rrf=0.8,
            forward_novelty=0.8,
        ),
        MultiSignalCandidate(
            time_seconds=3.0,
            family_rrf=0.1,
            forward_novelty=0.9,
        ),
    )

    selected = MinimaxBoundarySelector().select(candidates)

    assert selected == candidates[1]


def test_minimax_uses_frozen_tie_break_order() -> None:
    candidates = (
        MultiSignalCandidate(
            time_seconds=3.0,
            family_rrf=0.8,
            forward_novelty=0.9,
        ),
        MultiSignalCandidate(
            time_seconds=2.0,
            family_rrf=0.9,
            forward_novelty=0.8,
        ),
    )

    selected = MinimaxBoundarySelector().select(candidates)

    # Both candidates have the same rank_max and rank_sum.
    # Frozen MINIMAX tie-break:
    # better RRF rank -> better novelty rank -> earlier time.
    assert selected == candidates[1]


def test_assign_family_rrf_uses_frozen_family_definition() -> None:
    candidates = (
        FamilyRankCandidate(
            time_seconds=1.0,
            d250=0.9,
            d500=0.8,
            d1=0.7,
            d2=0.6,
            mfcc_geo=0.5,
        ),
        FamilyRankCandidate(
            time_seconds=2.0,
            d250=0.8,
            d500=0.9,
            d1=0.6,
            d2=0.5,
            mfcc_geo=0.7,
        ),
        FamilyRankCandidate(
            time_seconds=3.0,
            d250=0.7,
            d500=0.6,
            d1=0.9,
            d2=0.8,
            mfcc_geo=0.6,
        ),
    )

    ranked = assign_family_rrf(candidates)

    first = ranked[0]

    assert first.family_short_rank == 2.0
    assert first.family_long_rank == 2.0
    assert first.family_regime_rank == 3.0

    expected = 1.0 / 2.0 + 1.0 / 2.0 + 1.0 / 3.0

    assert first.family_rrf == expected


def test_forward_novelty_detects_unexplained_post_boundary_spectrum() -> None:
    sample_rate = 8000
    center_seconds = 1.0
    window_samples = 6000

    audio = np.zeros(
        sample_rate * 2,
        dtype=np.float64,
    )

    pre_time = np.arange(window_samples) / sample_rate
    post_time = np.arange(window_samples) / sample_rate

    center = sample_rate

    audio[center - window_samples : center] = np.sin(2.0 * np.pi * 440.0 * pre_time)

    audio[center : center + window_samples] = np.sin(2.0 * np.pi * 880.0 * post_time)

    novelty = forward_novelty(
        audio,
        center_seconds=center_seconds,
    )

    assert novelty == pytest.approx(
        0.9999995477020619,
        abs=1e-12,
    )


def test_build_union_candidates_matches_frozen_full_union_policy() -> None:
    candidates = build_union_candidates(
        semantic_time=10.0,
        basin_times=(
            2.500,  # outside: offset 7.5
            3.000,  # accepted
            5.250,  # accepted
            9.000,  # accepted
            9.250,  # outside: offset .75
        ),
        d2_points=(
            (3.0004, 0.4),
            (3.2500, 0.8),
            (3.5000, 0.6),
            (4.0000, 0.2),
            (4.2500, 0.2),
            (4.5000, 0.1),
            (9.0000, 0.9),
        ),
    )

    assert [
        (
            candidate.time_seconds,
            candidate.sources,
        )
        for candidate in candidates
    ] == [
        (3.0, ("BASIN",)),
        (3.25, ("D2",)),
        (4.25, ("D2",)),
        (5.25, ("BASIN",)),
        (9.0, ("BASIN", "D2")),
    ]


def test_multisignal_resolver_selects_frozen_minimax_boundary() -> None:
    resolver = MultiSignalCandidateResolver(
        candidate_builder=lambda **_: (
            MultiSignalCandidate(
                time_seconds=3.0,
                family_rrf=0.9,
                forward_novelty=0.1,
            ),
            MultiSignalCandidate(
                time_seconds=4.0,
                family_rrf=0.8,
                forward_novelty=0.8,
            ),
            MultiSignalCandidate(
                time_seconds=5.0,
                family_rrf=0.1,
                forward_novelty=0.9,
            ),
        ),
    )

    resolution = resolver.resolve_selected_boundary(
        semantic_time_seconds=8.0,
    )

    assert resolution is not None
    assert resolution.time_seconds == 4.0


def test_acoustic_multisignal_builder_scores_full_union_candidates() -> None:
    class FakeRmsAnalyzer:
        def analyze(self, pcm):
            return object()

    class FakeHybrid:
        def _basins(self, **_):
            return (
                SimpleNamespace(
                    center_time_seconds=3.0,
                ),
                SimpleNamespace(
                    center_time_seconds=5.0,
                ),
            )

    class FakeLocalAnalyzer:
        def analyze(
            self,
            *,
            pcm,
            boundary_time_seconds,
            absolute_start_time_seconds,
        ):
            del pcm
            del absolute_start_time_seconds

            values = {
                3.0: (0.9, 0.8, 0.7, 0.6),
                3.25: (0.8, 0.9, 0.6, 0.9),
                5.0: (0.7, 0.6, 0.9, 0.8),
            }

            if boundary_time_seconds not in values:
                baseline = max(
                    0.001,
                    0.2 - 0.01 * boundary_time_seconds,
                )

                return SimpleNamespace(
                    change_250ms=baseline,
                    change_500ms=baseline,
                    change_1s=baseline,
                    change_2s=baseline,
                )

            d250, d500, d1, d2 = values[boundary_time_seconds]

            return SimpleNamespace(
                change_250ms=d250,
                change_500ms=d500,
                change_1s=d1,
                change_2s=d2,
            )

    builder = AcousticMultiSignalCandidateBuilder(
        rms_analyzer=FakeRmsAnalyzer(),
        hybrid=FakeHybrid(),
        local_analyzer=FakeLocalAnalyzer(),
        regime_ratio=lambda **kwargs: {
            3.0: 0.5,
            3.25: 0.7,
            5.0: 0.6,
        }[kwargs["candidate_time"]],
        novelty=lambda **kwargs: {
            3.0: 0.1,
            3.25: 0.8,
            5.0: 0.9,
        }[kwargs["candidate_time"]],
        d2_step_seconds=0.25,
    )

    window = AcousticWindow(
        start_offset=0,
        end_offset=1,
        start_time_seconds=0.0,
        end_time_seconds=20.0,
        data=b"x",
    )

    pcm8 = DecodedPcm(
        sample_rate=8000,
        channels=1,
        sample_width_bytes=2,
        data=b"\x00\x00" * (8000 * 20),
    )

    pcm16 = DecodedPcm(
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
        data=b"\x00\x00" * (16000 * 20),
    )

    candidates = builder.build(
        semantic_time=10.0,
        window=window,
        pcm8=pcm8,
        pcm16=pcm16,
    )

    assert [candidate.time_seconds for candidate in candidates] == [
        3.0,
        3.25,
        5.0,
    ]

    assert [candidate.forward_novelty for candidate in candidates] == [
        0.1,
        0.8,
        0.9,
    ]

    assert candidates[0].family_rrf == pytest.approx(1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 3.0)


def test_multisignal_resolver_selects_from_acoustic_builder() -> None:
    class FakeAcousticBuilder:
        def build(
            self,
            *,
            semantic_time,
            window,
            pcm8,
            pcm16,
        ):
            assert semantic_time == 10.0
            assert window is acoustic_window
            assert pcm8 is decoded_pcm8
            assert pcm16 is decoded_pcm16

            return (
                MultiSignalCandidate(
                    time_seconds=3.0,
                    family_rrf=0.9,
                    forward_novelty=0.1,
                ),
                MultiSignalCandidate(
                    time_seconds=4.0,
                    family_rrf=0.8,
                    forward_novelty=0.8,
                ),
                MultiSignalCandidate(
                    time_seconds=5.0,
                    family_rrf=0.1,
                    forward_novelty=0.9,
                ),
            )

    acoustic_window = AcousticWindow(
        start_offset=0,
        end_offset=1,
        start_time_seconds=0.0,
        end_time_seconds=20.0,
        data=b"x",
    )

    decoded_pcm8 = DecodedPcm(
        sample_rate=8000,
        channels=1,
        sample_width_bytes=2,
        data=b"\x00\x00" * (8000 * 20),
    )

    decoded_pcm16 = DecodedPcm(
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
        data=b"\x00\x00" * (16000 * 20),
    )

    resolver = MultiSignalCandidateResolver(
        candidate_builder=FakeAcousticBuilder(),
    )

    selected = resolver.resolve_acoustic_boundary(
        semantic_time_seconds=10.0,
        window=acoustic_window,
        pcm8=decoded_pcm8,
        pcm16=decoded_pcm16,
    )

    assert selected is not None
    assert selected.time_seconds == 4.0
