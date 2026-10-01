from types import SimpleNamespace

from brynse.matching import TemporalSplitAligner
from brynse.models import (
    EncodedAudioFrame,
    SplitKind,
    TemporalSplitDecision,
    TemporalSplitKind,
)


def _frame(*, offset: int, time_seconds: float) -> EncodedAudioFrame:
    return EncodedAudioFrame(
        codec="mp3",
        offset=offset,
        length=417,
        sample_rate=44100,
        samples=1152,
        time_seconds=time_seconds,
    )


def test_exclusion_collapsing_to_same_frame_becomes_hard_cut() -> None:
    timeline = SimpleNamespace(
        frames=(
            _frame(offset=1000, time_seconds=10.000),
            _frame(offset=1417, time_seconds=10.026),
            _frame(offset=1834, time_seconds=10.052),
        )
    )
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=10.026000,
        incoming_start_seconds=10.026204,
    )

    aligned = TemporalSplitAligner().align(
        decision=decision,
        timeline=timeline,
    )

    assert aligned.kind is SplitKind.HARD_CUT
    assert aligned.outgoing_end == 1417
    assert aligned.incoming_start == 1417


def test_exclusion_preserves_gap_when_boundaries_align_to_distinct_frames() -> None:
    timeline = SimpleNamespace(
        frames=(
            _frame(offset=1000, time_seconds=10.000),
            _frame(offset=1417, time_seconds=10.026),
            _frame(offset=1834, time_seconds=10.052),
        )
    )
    decision = TemporalSplitDecision(
        kind=TemporalSplitKind.EXCLUSION,
        outgoing_end_seconds=10.026,
        incoming_start_seconds=10.052,
    )

    aligned = TemporalSplitAligner().align(
        decision=decision,
        timeline=timeline,
    )

    assert aligned.kind is SplitKind.EXCLUSION
    assert aligned.outgoing_end == 1417
    assert aligned.incoming_start == 1834
