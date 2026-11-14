"""F-2: excluded encoded bytes must be explicitly represented."""

from types import SimpleNamespace

import pytest

from brynse.models import SegmentByteRange, SegmentWritePlan, SplitKind
from brynse.output import SegmentRangePlanner


@pytest.mark.parametrize(
    ("outgoing_end", "incoming_start", "expected"),
    [
        (400, 500, SegmentByteRange(400, 500)),
        (400, 400, None),
        (430, 420, None),
    ],
)
def test_plan_exposes_only_positive_unowned_intervals(
    outgoing_end: int,
    incoming_start: int,
    expected: SegmentByteRange | None,
) -> None:
    decision = SimpleNamespace(
        kind=SplitKind.HARD_CUT,
        outgoing_end=outgoing_end,
        incoming_start=incoming_start,
    )

    plan = SegmentRangePlanner().plan(
        previous_start_offset=100,
        next_end_offset=1000,
        decision=decision,
    )

    assert plan.outgoing == SegmentByteRange(100, outgoing_end)
    assert plan.incoming == SegmentByteRange(incoming_start, 1000)
    assert plan.unowned == expected


def test_backward_compatible_plan_defaults_to_no_unowned() -> None:
    plan = SegmentWritePlan(
        outgoing=SegmentByteRange(100, 400),
        incoming=SegmentByteRange(400, 1000),
    )

    assert plan.unowned is None


def test_nonempty_unowned_range_is_required() -> None:
    with pytest.raises(ValueError):
        SegmentByteRange(400, 400)
