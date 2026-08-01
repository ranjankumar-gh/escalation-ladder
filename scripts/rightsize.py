"""Chapter 16 - which rungs your ladders pay for and do not use.

Runs entirely from the committed recordings. No key, no network, no spend --
which matters here for the same reason it mattered in Chapter 11, and one more:
a descent argued from numbers only the author can regenerate is a descent
nobody can argue with.

The replay and the hole-detection are imported from `scripts.composite_cost`
rather than rewritten, so every population counted here is the population
Chapter 11 published. A second implementation of "which requests did we
actually measure" is a second answer to the only question a Removal Receipt
rests on.

Usage:
    python -m scripts.rightsize                    # every rung on every ladder
    python -m scripts.rightsize --drop diagnose 6  # run stage one and price it
    python -m scripts.rightsize --holds 6          # what would break, by stage
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from escalation_ladder.descend import (  # noqa: E402
    Receipt,
    holds,
    receipt,
    unroute,
    unused,
)
from escalation_ladder.fixtures.incidents import load_incidents  # noqa: E402
from escalation_ladder.llm import RecordedCompleter  # noqa: E402
from escalation_ladder.router import ASKS, LADDERS, Ask, effective_level  # noqa: E402
from escalation_ladder.router import handle  # noqa: E402
from scripts.composite_cost import (  # noqa: E402
    replay,
    single_rung_curve,
    unrecorded,
)


def settled(
    completer: RecordedCompleter,
    ask: Ask,
    ladders: dict[str, tuple[int, ...]] | None = None,
) -> tuple[list[int | None], list[int]]:
    """The rung each measured request settled on, and what each one cost.

    A request with no recording anywhere on its ladder is dropped rather than
    counted as unanswered, which is Chapter 11's rule and matters more here:
    a hole counted as a refusal inflates the population, and a population is
    the numerator of the only clause a descent can actually satisfy.
    """
    reached: list[int | None] = []
    tokens: list[int] = []
    for incident in load_incidents():
        outcome = handle(incident, ask, completer, ladders=ladders)
        if unrecorded(outcome):
            continue
        reached.append(outcome.level_reached)
        tokens.append(outcome.tokens)
    return reached, tokens


def receipts(completer: RecordedCompleter) -> list[Receipt]:
    """One Removal Receipt per rung on per ladder it appears on."""
    found: list[Receipt] = []
    for ask in ASKS:
        reached, _tokens = settled(completer, ask)
        for level in LADDERS[ask]:
            found.append(receipt(level, ask, reached, ladders=LADDERS))
    return found


def print_table(completer: RecordedCompleter) -> None:
    print("Removal receipts -- every rung on every ladder\n")
    header = (
        f"  {'ask':10} {'rung':>5} {'answered':>10} {'floor':>7} "
        f"{'needs':>7} {'authorizes':>11}  why"
    )
    print(header)
    for entry in receipts(completer):
        stage = entry.authorized or "-"
        share = f"{entry.answered}/{entry.population}"
        floor = f"{entry.floor:.2f}" + ("*" if entry.saturated else " ")
        print(
            f"  {entry.ask:10} {'L' + str(entry.level):>5} {share:>10} "
            f"{floor:>7} {entry.needed:>7} {stage:>11}  {entry.decided_by}"
        )
    print("\n  * the floor is the ceiling: this population detects no effect at all.")
    for level in unused(LADDERS):
        grips = [h.render() for h in holds(level, LADDERS) if h.blocks == "code"]
        print(
            f"\n  L{level} is on no ladder: stage one is already done, and it "
            f"emits no counter.\n    rungs blocking stage two: "
            f"{', '.join(grips) or 'none'}"
        )


def print_holds(level: int) -> None:
    print(f"What holds Level {level}\n")
    labels = {
        "route": "blocks stage one, the ladder entry",
        "code": "blocks stage two, a rung that would break",
        None: "part of stage two rather than an obstacle to it",
    }
    for stage in ("route", "code", None):
        grips = [h for h in holds(level, LADDERS) if h.blocks == stage]
        print(f"  {labels[stage]}: {len(grips)}")
        for holder in grips:
            print(f"    {holder.render()}")


def print_drop(completer: RecordedCompleter, ask: Ask, level: int) -> None:
    """Run the ladder with the rung and without it, over one corpus."""
    after = unroute(LADDERS, ask, level)
    curve, _counts = single_rung_curve(completer)
    print(f"Stage one on {ask}: {LADDERS[ask]} -> {after[ask]}\n")
    print(f"  {'ladder':16} {'answered':>10} {'mean tokens':>12} {'eff level':>10}")
    for label, ladders in (("shipped", None), ("unrouted", after)):
        reached, tokens = settled(completer, ask, ladders)
        answered = sum(1 for rung in reached if rung is not None)
        mean = sum(tokens) / len(tokens) if tokens else 0.0
        level_reached = effective_level(mean, curve) if mean else 0.0
        print(
            f"  {label:16} {f'{answered} of {len(reached)}':>10} "
            f"{mean:>12.0f} {level_reached:>10.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rightsize the composite.")
    parser.add_argument(
        "--drop", nargs=2, metavar=("ASK", "LEVEL"),
        help="measure removing one rung from one ask's ladder",
    )
    parser.add_argument(
        "--holds", type=int, metavar="LEVEL",
        help="print every edge that would break if this rung disappeared",
    )
    args = parser.parse_args(argv)

    if args.holds is not None:
        print_holds(args.holds)
        return 0

    completer = replay()
    if args.drop:
        ask, level = args.drop[0], int(args.drop[1])
        if ask not in LADDERS:
            parser.error(f"no ladder for ask {ask!r}")
        print_drop(completer, ask, level)  # type: ignore[arg-type]
        return 0

    print_table(completer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
