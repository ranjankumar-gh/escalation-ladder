"""Measure the composite: per-ask cost, the cost curve, and the Effective Level.

Runs entirely from the committed recordings. No key, no network, no spend --
which matters more here than anywhere else in the book, because a composite is
the one thing a reader cannot check by reading a single chapter's numbers.

Usage:
    python -m scripts.composite_cost              # the whole chapter's figures
    python -m scripts.composite_cost --budget 1   # cap every ask at one rung
    python -m scripts.composite_cost --mix 0.54 0.44 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from escalation_ladder.fixtures.incidents import load_incidents  # noqa: E402
from escalation_ladder.llm import RecordedCompleter, Usage  # noqa: E402
from escalation_ladder.router import (  # noqa: E402
    ASKS,
    BUDGET,
    LADDERS,
    RUNGS,
    Ask,
    Outcome,
    effective_level,
    handle,
    max_model_calls,
    mix_cost,
    try_rung,
)

RECORDINGS = Path(__file__).resolve().parent.parent / "tests" / "recordings"

# The mix that produces the figure Chapter 2 and Chapter 10 both quote. It is an
# INPUT, not a result -- see the chapter. Override it with --mix.
DEFAULT_MIX: dict[str, float] = {"page": 0.54, "advise": 0.44, "diagnose": 0.02}


def replay() -> RecordedCompleter:
    """Answer from the recordings and price from the committed sidecar."""
    recordings: dict[str, str] = {}
    for path in sorted(RECORDINGS.glob("ch*.json")):
        if path.name == "usage.json":
            continue
        recordings.update(json.loads(path.read_text(encoding="utf-8")))
    priced = {
        key: Usage(int(entry["input_tokens"]), int(entry["output_tokens"]))
        for key, entry in json.loads(
            (RECORDINGS / "usage.json").read_text(encoding="utf-8")
        ).items()
    }
    return RecordedCompleter(recordings=recordings, usage_by_key=priced)


def unrecorded(outcome: Outcome) -> bool:
    """True when a rung refused only because this replay has no recording for it.

    A missing recording is a hole in the measurement, not a refusal the system
    made, and counting it as one would publish an artifact of what earlier
    chapters happened to measure as a property of the architecture. Chapters 4
    through 6 only ever ran the four incidents Level 0 refuses, so an `advise`
    request about an alert line has no recording anywhere on its ladder.
    """
    return any("no recording" in attempt.decided_by for attempt in outcome.attempts)


def per_ask(completer: RecordedCompleter, budget: dict[Ask, int]) -> dict[str, dict]:
    """Run every incident through every ask, and summarize by ask."""
    incidents = load_incidents()
    summary: dict[str, dict] = {}
    for ask in ASKS:
        rows = []
        for incident in incidents:
            outcome = handle(incident, ask, completer, budget=budget)
            if unrecorded(outcome):
                continue
            rows.append(outcome)
        tokens = [o.tokens for o in rows]
        summary[ask] = {
            "n": len(rows),
            "mean_tokens": sum(tokens) / len(rows) if rows else 0.0,
            "mean_calls": sum(o.ledger.model_calls for o in rows) / len(rows)
            if rows
            else 0.0,
            "answered": sum(1 for o in rows if o.answered),
            "floor": sum(1 for o in rows if o.level_reached == 0),
            "wasted_rungs": sum(o.climbed for o in rows),
            "outcomes": rows,
        }
    return summary


def single_rung_curve(completer: RecordedCompleter) -> tuple[dict[int, float], dict[int, int]]:
    """Mean cost of each rung run alone, over the incidents it has recordings for.

    This is the anchor the Effective Level is read against. Every rung is run on
    every incident independently rather than through a ladder, because a ladder
    only ever reaches an upper rung on the incidents the rungs below it refused.
    Those are the hard ones, and a curve built from them would price the top of
    the ladder against a corpus the composite never faced.
    """
    incidents = load_incidents()
    samples: dict[int, list[int]] = {}
    for level in sorted(RUNGS):
        for incident in incidents:
            attempt, _decision, _step = try_rung(incident, level, completer)
            if "no recording" in attempt.decided_by:
                continue
            samples.setdefault(level, []).append(attempt.tokens)
    curve = {level: sum(v) / len(v) for level, v in samples.items() if v}
    return curve, {level: len(v) for level, v in samples.items()}


def widen_diagnose(completer: RecordedCompleter) -> int:
    """Compare the shipped diagnose ladder against one that starts at retrieval.

    The widened ladder is eleven times cheaper and answers more of the corpus,
    and Chapter 11 argues it is wrong anyway, because a retrieved runbook is a
    hypothesis rather than a confirmation. The comparison is matched on the
    incidents both ladders have recordings for -- an unmatched mean would credit
    the widened ladder with an incident the shipped one was never scored on.
    """
    import escalation_ladder.router as router_module

    shipped = router_module.LADDERS["diagnose"]
    widened = (2, 3) + shipped
    incidents = load_incidents()
    results: dict[tuple[int, ...], dict[str, Outcome]] = {}
    try:
        for ladder in (shipped, widened):
            router_module.LADDERS["diagnose"] = ladder
            rows: dict[str, Outcome] = {}
            for incident in incidents:
                outcome = handle(
                    incident, "diagnose", completer,
                    budget={"diagnose": len(ladder)},
                )
                if not unrecorded(outcome):
                    rows[incident.incident_id] = outcome
            results[ladder] = rows
    finally:
        router_module.LADDERS["diagnose"] = shipped

    shared = sorted(set(results[shipped]) & set(results[widened]))
    curve, _counts = single_rung_curve(completer)
    print(f"Matched on {len(shared)} incidents: {', '.join(shared)}\n")
    print(f"  {'ladder':22} {'answered':>9} {'mean tokens':>12} {'eff level':>10}")
    for ladder in (shipped, widened):
        rows = [results[ladder][key] for key in shared]
        mean = sum(row.tokens for row in rows) / len(rows)
        answered = sum(1 for row in rows if row.answered)
        print(f"  {str(ladder):22} {answered:>4} of {len(rows)} {mean:>12.0f} "
              f"{effective_level(mean, curve):>10.2f}")
    print("\nPer incident, rung reached:")
    for key in shared:
        print(f"  {key}: shipped={results[shipped][key].level_reached} "
              f"widened={results[widened][key].level_reached}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the composite router.")
    parser.add_argument("--budget", type=int, default=None,
                        help="cap every ask at this many rungs")
    parser.add_argument("--mix", type=float, nargs=3, default=None,
                        metavar=("PAGE", "ADVISE", "DIAGNOSE"),
                        help="traffic mix; must be three shares")
    parser.add_argument("--widen-diagnose", action="store_true",
                        help="compare the shipped diagnose ladder against one "
                             "starting at retrieval, matched on shared incidents")
    args = parser.parse_args(argv)

    completer = replay()
    if args.widen_diagnose:
        return widen_diagnose(completer)
    budget: dict[Ask, int] = (
        dict(BUDGET) if args.budget is None
        else {ask: min(args.budget, len(LADDERS[ask])) for ask in ASKS}
    )
    mix = (
        dict(DEFAULT_MIX) if args.mix is None
        else dict(zip(ASKS, args.mix))
    )

    curve, counts = single_rung_curve(completer)
    print("Cost curve -- mean tokens for one rung run alone")
    for level in sorted(curve):
        print(f"  L{level}: {curve[level]:9.1f}  (n={counts[level]})")

    print("\nPer-ask cascade")
    print(f"  {'ask':9} {'n':>3} {'answered':>9} {'floor':>6} "
          f"{'refused rungs':>14} {'tokens':>9} {'calls':>6} {'eff level':>10}")
    summary = per_ask(completer, budget)
    means: dict[Ask, float] = {}
    for ask in ASKS:
        row = summary[ask]
        means[ask] = row["mean_tokens"]
        level = effective_level(row["mean_tokens"], curve) if row["n"] else 0.0
        print(f"  {ask:9} {row['n']:>3} {row['answered']:>9} {row['floor']:>6} "
              f"{row['wasted_rungs']:>14} {row['mean_tokens']:>9.1f} "
              f"{row['mean_calls']:>6.2f} {level:>10.2f}")

    cost = mix_cost(mix, means)
    print(f"\nMix {mix} -> {cost:.1f} mean tokens per request")
    print(f"Effective Level: {effective_level(cost, curve):.2f}")

    print("\nTermination Test on the composite (maximum API requests per ask)")
    for ask, ceiling in max_model_calls(8).items():
        print(f"  {ask:9} {ceiling}")

    if args.budget is None:
        print("\nThe cascade tax -- what the climb cost beyond the rung that answered")
        for ask in ASKS:
            for outcome in summary[ask]["outcomes"]:
                if not outcome.attempts:
                    continue
                paid = outcome.tokens
                answered_at = outcome.level_reached
                alone = curve.get(answered_at) if answered_at is not None else None
                tax = f"{paid / alone:.2f}x" if alone else "-"
                print(f"  {ask:9} {outcome.incident_id} reached="
                      f"{answered_at if answered_at is not None else 'none':>4} "
                      f"paid={paid:>6} tax={tax}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
