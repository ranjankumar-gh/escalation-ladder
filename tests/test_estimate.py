"""The behaviours Chapter 12 argues for, asserted against the estimator.

Four claims are load-bearing and each one has a test that goes red if it stops
being true:

1. An architecture that cannot answer the Termination Test gets a refusal, not a
   number. Chapter 3's rule applied to a prediction.
2. Cost is convex in turn count, so an estimate built from a MEAN turn count is a
   floor. This is the arithmetic behind Chapter 12's Level 5 row.
3. A growing carry and a flat carry are indistinguishable at the turn count you
   observe and separated by nearly threefold at the turn count you configured.
   The whole reason the carry discipline is worth naming.
4. The declared shapes are ROUNDED design-document numbers, not the measurements
   fed back in. Without this the error table is a tautology.
"""
from __future__ import annotations

import pytest

from escalation_ladder.estimate import (
    RAW_CARRY_GROWTH,
    RAW_LEVEL_6,
    SHAPES,
    Shape,
    curve,
    error,
    estimate,
    estimate_from_mean,
)

UNBOUNDED = Shape(
    turns=None,
    ceiling=None,
    first_input=5_000,
    growth=200,
    carry="flat",
    output=300,
    tools=True,
    counted_by="a loop with no configured cap, which is the shape under test",
)


def test_an_architecture_with_no_ceiling_is_refused_rather_than_estimated() -> None:
    """A number here would launder 'we did not bound it' into a budget."""
    result = estimate(UNBOUNDED)
    assert result.input_tokens is None
    assert result.tokens is None
    assert result.requests is None
    assert "Termination Test" in result.decided_by


def test_a_configured_ceiling_is_costed_and_labelled_as_a_ceiling() -> None:
    """Level 6 ships Budget(8). The estimate must use it and say that it did."""
    result = estimate(SHAPES["Level 6"])
    assert result.input_tokens == 49_400
    assert "ceiling" in result.decided_by
    assert "not a budget" in result.decided_by


def test_a_mean_turn_count_produces_a_floor_not_a_point_estimate() -> None:
    """Cost is convex in turns, so the mean of the costs exceeds the cost of the mean.

    Asserted directly rather than argued: two cases either side of a mean cost
    more, averaged, than the mean case costs.
    """
    spread = Shape(
        turns=(1, 5),
        ceiling=5,
        first_input=5_000,
        growth=500,
        carry="flat",
        output=300,
        tools=True,
        counted_by="a synthetic two-case spread with a mean of three turns",
    )
    from_distribution = estimate(spread)
    from_mean = estimate_from_mean(spread, 3.0)

    assert from_distribution.input_tokens is not None
    assert from_mean.input_tokens is not None
    assert from_distribution.input_tokens > from_mean.input_tokens
    assert from_mean.floor is True
    assert from_distribution.floor is False


def test_a_flat_carry_is_never_marked_a_floor_from_a_mean() -> None:
    """With no growth the convexity vanishes, so a mean is exact.

    The complement of the test above, and the reason `floor` is computed from
    `growth` rather than hard-coded: a single-turn or no-carry architecture can
    be estimated from a mean with no penalty.
    """
    flat = Shape(
        turns=(1, 5),
        ceiling=5,
        first_input=5_000,
        growth=0,
        carry="none",
        output=300,
        tools=False,
        counted_by="the same spread with the carry removed",
    )
    assert estimate_from_mean(flat, 3.0).floor is False
    assert estimate_from_mean(flat, 3.0).input_tokens == 15_000


def test_the_carry_discipline_is_invisible_early_and_dominant_late() -> None:
    """Chapter 12's headline. Measured at both ends of the same Budget(8)."""
    bounded = dict(curve(SHAPES["Level 6"], (3, 8)))
    growing = dict(curve(RAW_LEVEL_6, (3, 8)))

    at_the_observed_mean = growing[3] / bounded[3]
    at_the_configured_cap = growing[8] / bounded[8]

    assert at_the_observed_mean < 1.2, (
        "the two carry disciplines must be nearly indistinguishable at the "
        "round count this system actually reaches, or the chapter's surprise "
        "is not a surprise"
    )
    assert at_the_configured_cap > 2.5, (
        "and separated by more than twofold at the cap that was configured"
    )


def test_a_tool_turn_costs_two_vendor_requests() -> None:
    """Chapter 8's trap, pinned. A ledger counts turns; a rate limit counts requests."""
    with_tools = estimate(SHAPES["Level 5"])
    without = estimate(SHAPES["Level 3"])
    assert with_tools.requests == 4  # 2.20 turns, doubled and rounded
    assert without.requests == 2  # 2.00 turns, one request each


def test_a_single_turn_rung_still_costs_two_requests_when_it_uses_tools() -> None:
    """Level 4 takes one turn and two vendor requests, and both must be reported.

    The tempting error is to reason that the seam measures a tool round as one
    turn, so the request count should be one too. That conflates the two units:
    the TOKEN total already spans both requests, while the REQUEST count is what
    a rate limit consumes and it is genuinely two. Getting this wrong surfaces as
    a throughput incident rather than as a wrong bill, which is why it is pinned.
    """
    level_4 = SHAPES["Level 4"]
    assert level_4.turns == (1,)
    assert estimate(level_4).requests == 2


def test_the_tools_flag_moves_requests_and_never_tokens() -> None:
    """The invariant behind the test above, asserted directly.

    Without this, 'fixing' a request count by editing `tools` could silently
    move a token estimate and no row in the error table would explain why.
    """
    from dataclasses import replace

    level_4 = SHAPES["Level 4"]
    without = estimate(replace(level_4, tools=False))
    with_tools = estimate(level_4)

    assert with_tools.input_tokens == without.input_tokens
    assert with_tools.output_tokens == without.output_tokens
    assert with_tools.requests == 2 * without.requests


@pytest.mark.parametrize("rung", sorted(SHAPES))
def test_every_declared_shape_carries_its_provenance(rung: str) -> None:
    """An estimate whose inputs cannot be reviewed is a guess with a table around it."""
    assert len(SHAPES[rung].counted_by) > 30


@pytest.mark.parametrize("rung", sorted(SHAPES))
def test_the_declared_inputs_are_rounded_rather_than_measured(rung: str) -> None:
    """Without this the error table is a tautology.

    Every token figure in `SHAPES` must be a round number of the kind a design
    document carries. A shape holding 1,209 rather than 1,200 would mean the
    measurement was fed back in, and the estimator would be scoring itself
    against its own inputs.
    """
    shape = SHAPES[rung]
    assert shape.first_input % 100 == 0, "first_input is not a round figure"
    assert shape.growth % 50 == 0, "growth is not a round figure"
    assert shape.output % 50 == 0, "output is not a round figure"


def test_the_growth_constant_matches_the_measurement_it_came_from() -> None:
    """Chapter 8 measured increments of 1,046 then 1,650 over three steps.

    Pinned so that a later edit to `RAW_CARRY_GROWTH` has to go back to the
    measurement rather than to whatever makes the curve look better.
    """
    assert RAW_CARRY_GROWTH == pytest.approx(1_650 / 1_046, abs=0.01)


def test_an_error_ratio_is_none_rather_than_zero_when_nothing_was_measured() -> None:
    """The measurement-honesty rule this book applies to every script, applied here.

    A rung with no measurement must not compare as 0.00x, which reads as a
    perfect estimate of nothing.
    """
    assert error(None, 1_000) is None
    assert error(1_000, 0) is None
    assert error(1_100, 1_000) == pytest.approx(1.1)
