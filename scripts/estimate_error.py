"""Score the estimator against what the rungs actually cost.

An estimate nobody checked is an opinion. This script is the check: it runs
`escalation_ladder.estimate` over the shapes declared there, replays every rung
against the same fixture corpus, and prints how far apart they landed.

Two error terms, reported separately rather than multiplied into one number,
because they have different causes and different fixes.

REPLAY ERROR is the gap between a replayed row and the row the chapter published
from a live run. It is documented in Appendix B and it is not this estimator's
fault: pricing a structured-output schema as a strict tool overshoots by roughly
ten percent, and the recorded Level 5 path settles in fewer steps than its live
mean took.

ESTIMATOR ERROR is the gap between the estimate and the replay. That one IS this
file's subject, and it is the number Chapter 12 stands on.

    python -m scripts.estimate_error            # both tables
    python -m scripts.estimate_error --curve    # the loop cost curve, as data

Free. No key, no network, no spend.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from escalation_ladder.estimate import (
    RAW_LEVEL_6,
    SHAPES,
    curve,
    error,
    estimate,
    estimate_from_mean,
)
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import RecordedCompleter, Usage
from escalation_ladder.rungs import load_all
from scripts.build_usage import load_recordings

SIDECAR = Path(__file__).resolve().parent.parent / "tests" / "recordings" / "usage.json"

# Mean INPUT tokens per billed incident as measured live and printed in each
# chapter, duplicated from tests/test_usage.py so this script has no test
# dependency. If these two ever disagree, the test is right.
PUBLISHED: dict[str, int] = {
    "Level 1": 1_098,
    "Level 2": 1_719,
    "Level 3": 2_735,
    "Level 4": 6_055,
    "Level 5": 14_803,
    "Level 6": 18_154,
}

# Live mean turns, from each chapter's own measurement. Level 6 is the only rung
# whose turn count is not fixed by its code, which is the entire point of it.
LIVE_TURNS: dict[str, float] = {"Level 5": 2.55, "Level 6": 3.17}


def replayed() -> dict[str, dict[str, float]]:
    """Mean input, output, and turn count per BILLED incident, from recordings."""
    if not SIDECAR.exists():
        raise SystemExit(f"missing {SIDECAR} - run scripts/build_usage.py once")
    priced = {
        key: Usage(int(entry["input_tokens"]), int(entry["output_tokens"]))
        for key, entry in json.loads(SIDECAR.read_text(encoding="utf-8")).items()
    }
    recordings = load_recordings()
    incidents = load_incidents()

    rows: dict[str, dict[str, float]] = {}
    for rung, fn in load_all().items():
        takes_completer = "completer" in inspect.signature(fn).parameters
        billed = turns = inputs = outputs = 0
        for incident in incidents:
            completer = RecordedCompleter(recordings=recordings, usage_by_key=priced)
            try:
                ledger = fn(incident, completer) if takes_completer else fn(incident)
            except Exception:  # noqa: BLE001 - a rung with no recordings is data
                continue
            priced_turns = [
                m for m in ledger.measurements if m.model_calls and m.input_tokens
            ]
            if not priced_turns:
                continue
            billed += 1
            turns += len(priced_turns)
            inputs += sum(m.input_tokens for m in priced_turns)
            outputs += sum(m.output_tokens for m in priced_turns)
        if billed:
            rows[rung.split(":")[0]] = {
                "incidents": billed,
                "turns": turns / billed,
                "input": inputs / billed,
                "output": outputs / billed,
            }
    return rows


def _cell(value: float | None, fmt: str = "{:.2f}x") -> str:
    return "not measured" if value is None else fmt.format(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curve", action="store_true",
                        help="print the loop cost curve as data instead")
    args = parser.parse_args(argv)

    if args.curve:
        bounded = dict(curve(SHAPES["Level 6"], range(1, 9)))
        raw = dict(curve(RAW_LEVEL_6, range(1, 9)))
        print("| Turns | Bounded carry | Growing carry | Ratio |")
        print("|---|---|---|---|")
        for turn in range(1, 9):
            print(f"| {turn} | {bounded[turn]:,} | {raw[turn]:,} | "
                  f"{raw[turn] / bounded[turn]:.2f}x |")
        return 0

    rows = replayed()

    print("| Rung | Estimated in | Replayed in | Estimator | Published in | Replay |")
    print("|---|---|---|---|---|---|")
    for rung, shape in SHAPES.items():
        est = estimate(shape)
        row = rows.get(rung)
        replay_in = row["input"] if row else None
        estimator = error(est.input_tokens, int(replay_in)) if replay_in else None
        replay_err = (
            replay_in / PUBLISHED[rung]
            if replay_in and rung in PUBLISHED
            else None
        )
        print(
            f"| {rung} | {est.input_tokens:,} | "
            f"{f'{replay_in:,.0f}' if replay_in else 'not measured'} | "
            f"{_cell(estimator)} | {PUBLISHED.get(rung, 0):,} | "
            f"{_cell(replay_err)} |"
        )

    print()
    print("Turn counts, and where the estimate gets them:")
    print("| Rung | Estimated turns | Replayed turns | Live turns | Source |")
    print("|---|---|---|---|---|")
    for rung, shape in SHAPES.items():
        est = estimate(shape)
        row = rows.get(rung)
        turns = (
            f"{sum(shape.turns) / len(shape.turns):.2f}"
            if shape.turns
            else f"{shape.ceiling} (ceiling)"
        )
        source = "code" if shape.turns else "Termination Test ceiling"
        replayed_turns = f"{row['turns']:.2f}" if row else "-"
        print(
            f"| {rung} | {turns} | {replayed_turns} | "
            f"{LIVE_TURNS.get(rung, '-')} | {source} |"
        )

    print()
    level_6 = SHAPES["Level 6"]
    ceiling = estimate(level_6)
    mean = estimate_from_mean(level_6, LIVE_TURNS["Level 6"])
    print("Level 6, the only rung whose turn count its own code does not fix:")
    print(f"  at the Budget(8) ceiling : {ceiling.input_tokens:,} input tokens, "
          f"{ceiling.requests} requests")
    print(f"  at the measured mean     : {mean.input_tokens:,} input tokens, "
          f"{mean.requests} requests"
          f"{' (a floor)' if mean.floor else ''}")
    print(f"  the published live row   : {PUBLISHED['Level 6']:,} input tokens")
    print(f"  ceiling over mean        : "
          f"{ceiling.input_tokens / mean.input_tokens:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
