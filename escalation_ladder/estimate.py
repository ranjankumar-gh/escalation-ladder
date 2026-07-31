"""Cost estimation from an architecture's shape, before the architecture exists.

Every other module in this package measures. This one predicts, and it is the
only file here that never touches a model, a recording, or a network - which is
the point. Everything it needs is readable off the code and the prompts of a
system that has not been built: how many times it goes to the model, how big the
first request is, and how much bigger each later request gets.

Three deliberate omissions.

There is no price. Prose in this book carries ratios and Appendix B is the only
home for per-million rates, so an estimate here is denominated in tokens and
requests. A rate turns tokens into money by multiplication and dates in months;
the token count dates in years.

There is no latency. Latency is the other half of cost and it is not estimable -
it depends on queue depth, thinking budget, and whose datacentre is on fire.
Estimating it would produce a number with the same typography as the token count
and none of its standing.

Nothing here registers a rung. Estimation adds no capability, so it appears in no
cross-rung cost table, the same call `diagnose.py` and `router.py` already made.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

# How much a growing carry multiplies its own increment per turn. MEASURED in
# Chapter 8 over three steps of one incident - increments of 1,046 then 1,650
# tokens - and three points license a slope, not a law. It is a module constant
# rather than a field because no architecture in this book ships a growing carry:
# it is the counterfactual the bounded ones are compared against.
RAW_CARRY_GROWTH = 1.58

Carry = Literal["none", "flat", "growing"]


@dataclass(frozen=True)
class Shape:
    """What a design document has to state before its cost can be estimated.

    A declaration rather than a parsed signal, so it carries `counted_by` in
    place of the `decided_by` that every result object in this package carries.
    Same purpose: an estimate whose inputs have no provenance cannot be argued
    with in a review, and an estimate nobody can argue with is a guess with a
    table around it.
    """

    turns: tuple[int, ...] | None
    """Model turns per request, one entry per observed or assumed case.

    A fixed architecture has one entry and it is structural: Chapter 6's
    pipeline runs two stages, always. A loop's distribution is not knowable
    before it runs, which is why `ceiling` exists and why this may be None.
    """

    ceiling: int | None
    """The Termination Test's answer: the most turns this can take.

    None means the architecture cannot answer the Termination Test, which is
    Level 6 and above. A configured iteration cap is a ceiling; it is not an
    answer, and `estimate` refuses to treat it as one.
    """

    first_input: int
    """Input tokens in the first turn: system prompt, schema or tool
    definitions, and one representative payload. Countable before anything is
    built - `count_tokens` prices a prompt and generates nothing."""

    growth: int
    """Tokens each later turn adds to the first.

    Absorbs two different causes on purpose, because the bill cannot tell them
    apart: state carried forward from earlier turns, and a later stage whose
    fixed prompt is simply larger. You have to tell them apart, because a fatter
    prompt has a ceiling and a carry does not.
    """

    carry: Carry
    """"none" for a single turn, "flat" for a constant increment, "growing" for
    an increment that itself grows - see `RAW_CARRY_GROWTH`."""

    output: int
    """Output tokens per turn. The one input here that is genuinely a guess, and
    the one that matters least: it is under a tenth of the total above Level 3."""

    tools: bool
    """Whether a turn offers tools. A tool turn costs TWO vendor requests - the
    one that chooses tools and the one that reads their results - so it doubles
    the Termination Test's answer without changing the turn count. This is the
    trap Chapter 8 named: a ledger counts turns and a rate limit counts
    requests."""

    counted_by: str
    """Where these numbers came from. Prose, not a code reference: "system
    prompt counted at 1,004 tokens, schema at 705, report sampled at 200"."""


@dataclass(frozen=True)
class Estimate:
    """A per-request estimate, or a refusal naming what is missing."""

    input_tokens: int | None
    output_tokens: int | None
    requests: int | None
    """Vendor requests, not turns - what a rate limit and the Termination Test
    both count."""

    floor: bool
    """True when this is a lower bound rather than a point estimate. Cost is
    convex in turn count, so an estimate built from a MEAN turn count sits below
    the mean cost. See `_turns_cost`."""

    decided_by: str

    @property
    def tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def _turns_cost(turns: float, first_input: int, growth: int, carry: Carry) -> float:
    """Input tokens for `turns` turns, summing the growth as it accumulates.

    For a flat carry this is the arithmetic series
    `turns * first + growth * turns * (turns - 1) / 2`, which is CONVEX in
    `turns` whenever growth is positive - its second difference is exactly
    `growth`. That convexity is why `estimate` marks a mean-derived answer as a
    floor: by Jensen's inequality the mean of the costs exceeds the cost of the
    mean, so an architecture whose turn count varies costs more than its average
    turn count suggests. The gap is the variance of the turn count times half the
    growth, and on this book's Level 5 chain it is ten percent.
    """
    if turns <= 0:
        return 0.0
    if carry == "none" or growth == 0:
        return turns * first_input
    if carry == "growing":
        # Increment i_n = growth * RAW_CARRY_GROWTH^(n-1), accumulated. Written
        # as a loop rather than a closed form because the fractional tail of a
        # geometric series reads as a claim about turns nobody took.
        total = 0.0
        carried = 0.0
        increment = float(growth)
        remaining = turns
        while remaining > 0:
            weight = min(1.0, remaining)
            total += weight * (first_input + carried)
            carried += increment
            increment *= RAW_CARRY_GROWTH
            remaining -= 1.0
        return total
    return turns * first_input + growth * turns * (turns - 1) / 2.0


def estimate(shape: Shape) -> Estimate:
    """Estimate one request's cost, or refuse and say what is missing.

    Refuses rather than defaults, which is Chapter 3's rule applied to a
    prediction: a shape that answers neither `turns` nor `ceiling` describes an
    architecture whose cost is not bounded by anything its authors have written
    down, and returning a number for it would launder that fact into a budget.
    """
    if shape.turns is None and shape.ceiling is None:
        return Estimate(
            input_tokens=None,
            output_tokens=None,
            requests=None,
            floor=False,
            decided_by=(
                "no turn count and no ceiling: the architecture cannot answer "
                "the Termination Test, so it has no estimable cost"
            ),
        )

    if shape.turns:
        mean_turns = sum(shape.turns) / len(shape.turns)
        # Average the per-case COSTS rather than costing the average case. The
        # two differ by the convexity above, and the whole reason to hold a
        # distribution instead of a mean is to not pay that error.
        input_tokens = sum(
            _turns_cost(turn, shape.first_input, shape.growth, shape.carry)
            for turn in shape.turns
        ) / len(shape.turns)
        # Not a floor: each case is costed at its own turn count and the costs
        # are averaged, so the convexity is paid rather than skipped.
        floor = False
        decided_by = (
            f"{len(shape.turns)} case(s), mean {mean_turns:.2f} turns, "
            f"costed per case"
        )
    else:
        assert shape.ceiling is not None
        mean_turns = float(shape.ceiling)
        input_tokens = _turns_cost(
            mean_turns, shape.first_input, shape.growth, shape.carry
        )
        floor = False
        decided_by = (
            f"no turn distribution: costed at the Termination Test ceiling of "
            f"{shape.ceiling} turns, which is an upper bound and not a budget"
        )

    per_turn_requests = 2 if shape.tools else 1
    return Estimate(
        input_tokens=round(input_tokens),
        output_tokens=round(mean_turns * shape.output),
        requests=round(mean_turns * per_turn_requests),
        floor=floor,
        decided_by=decided_by,
    )


def estimate_from_mean(shape: Shape, mean_turns: float) -> Estimate:
    """Estimate from a mean turn count instead of a distribution.

    Kept separate rather than folded into `estimate` because the result is a
    FLOOR and the caller has to see that in the type. This is what a design
    document actually contains - "about three rounds" - and the gap between this
    and `estimate` over the same architecture is the price of not knowing the
    spread.
    """
    input_tokens = _turns_cost(
        mean_turns, shape.first_input, shape.growth, shape.carry
    )
    return Estimate(
        input_tokens=round(input_tokens),
        output_tokens=round(mean_turns * shape.output),
        requests=round(mean_turns * (2 if shape.tools else 1)),
        floor=shape.growth > 0,
        decided_by=(
            f"costed at a mean of {mean_turns:.2f} turns; a floor, because cost "
            f"is convex in turn count"
        ),
    )


def error(estimated: int | None, measured: int) -> float | None:
    """The estimate as a multiple of the measurement. None if not estimable.

    Returned as a ratio rather than a percentage or a signed difference, for the
    reason Chapter 2 gives: a ratio survives a price change and a difference does
    not.
    """
    if estimated is None or not measured:
        return None
    return estimated / measured


# One shape per rung of this book's own triage system, with every number a
# ROUNDED figure of the kind a design document carries rather than the exact
# replayed mean. Feeding the measurements back in would make the comparison in
# `scripts/estimate_error.py` a tautology; rounding keeps it a test.
#
# Level 0 is absent on purpose. It makes no turn, so its estimate is the Zero
# Row and there is nothing to model.
SHAPES: dict[str, Shape] = {
    "Level 1": Shape(
        turns=(1,),
        ceiling=1,
        first_input=1_200,
        growth=0,
        carry="none",
        output=100,
        tools=False,
        counted_by="system prompt and service catalog ~500, Classification "
                   "schema as a strict tool ~700, one report sampled at ~200",
    ),
    "Level 2": Shape(
        turns=(1,),
        ceiling=1,
        first_input=1_800,
        growth=0,
        carry="none",
        output=150,
        tools=False,
        counted_by="Level 1's request plus three retrieved runbook chunks at "
                   "~200 each, from the shipped relevance floor",
    ),
    "Level 3": Shape(
        turns=(2,),
        ceiling=2,
        first_input=1_200,
        growth=700,
        carry="flat",
        output=100,
        tools=False,
        counted_by="two stages read off the pipeline: classify at Level 1's "
                   "size, draft at ~1,900 with the retrieved runbook in it",
    ),
    "Level 4": Shape(
        turns=(1,),
        ceiling=1,
        first_input=5_700,
        growth=0,
        carry="none",
        output=900,
        tools=True,
        counted_by="five tool definitions counted at ~4,200 together, plus the "
                   "system prompt and report at ~1,500",
    ),
    "Level 5": Shape(
        turns=(1, 2, 2, 3, 3),
        ceiling=3,
        first_input=4_400,
        growth=600,
        carry="flat",
        output=500,
        tools=True,
        counted_by="tool definitions at ~4,200 plus a step prompt; growth from "
                   "the carried findings list, which the code caps at one line "
                   "per surviving claim",
    ),
    "Level 6": Shape(
        turns=None,
        ceiling=8,
        first_input=5_300,
        growth=250,
        carry="flat",
        output=350,
        tools=True,
        counted_by="Level 5's request plus the scratchpad; growth from "
                   "CARRY_VERBATIM, which carries conclusions rather than "
                   "results. Ceiling is the shipped Budget(8)",
    ),
}

# `tools` multiplies REQUESTS and never tokens, and keeping those two straight is
# the sharpest thing in this file. This book's seam measures one tool round as
# ONE turn covering TWO vendor requests: the ledger's token total already spans
# both, so `_turns_cost` must not double it - and the request count must, because
# that is the number a rate limit consumes.
#
# Level 4 is where the distinction bites, because it takes a single turn and
# still costs two requests. Declaring it tools=False would leave its token
# estimate untouched at 1.00x and understate its request count by half, which is
# the kind of error that surfaces as a throughput incident rather than a bill.
# If you take one thing from this module into your own system, take this: write
# down whether your unit is a turn or a request, before you write down the number.

# What the same Level 6 loop costs with no carry discipline - the counterfactual
# Chapter 8 measured and Chapter 9 declined to ship.
RAW_LEVEL_6 = Shape(
    turns=SHAPES["Level 6"].turns,
    ceiling=SHAPES["Level 6"].ceiling,
    first_input=SHAPES["Level 6"].first_input,
    growth=1_000,
    carry="growing",
    output=SHAPES["Level 6"].output,
    tools=True,
    counted_by="Chapter 8's raw-results arm: the first increment measured at "
               "1,046 tokens and each later one 1.58x the last",
)


def curve(shape: Shape, turns: Sequence[int]) -> list[tuple[int, int]]:
    """Cumulative input tokens at each turn count - the loop's cost curve.

    Returned rather than plotted, so the figure in Chapter 12 and the test that
    guards its shape read the same numbers.
    """
    return [
        (turn, round(_turns_cost(turn, shape.first_input, shape.growth, shape.carry)))
        for turn in turns
    ]
