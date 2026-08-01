"""Score every rung on every surface, against the committed recordings.

Chapter 13's numbers. Free: no key, no network, no spend. The one thing it will
not print is a judge score, because a judge prompt has no recording and a number
invented for it would be the failure this chapter is about.

Usage:
    python -m scripts.evaluate_rungs              # the surface table
    python -m scripts.evaluate_rungs --pinning    # what an upstream sample costs
    python -m scripts.evaluate_rungs --sizing     # how big an eval set must be
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from escalation_ladder import agent, classify, reasoning, retrieve, tools, workflow  # noqa: E402
from escalation_ladder.evaluate import (  # noqa: E402
    CLAIM_SURFACES,
    LATEST,
    SURFACES,
    claims,
    corpus,
    detectable_lift,
    floor_effect,
    observe,
    re_ask,
    samples_needed,
    score,
)

# The rung whose output each surface is scored on. A surface belongs to the
# LOWEST rung that can be wrong on it, which is why `service` is scored on
# Level 1 rather than on the most expensive rung that also produces one.
RUNGS = {
    1: ("Level 1: Prompt Application", classify.triage),
    2: ("Level 2: Retrieval-Augmented Generation", retrieve.advise),
    3: ("Level 3: LLM Workflow", workflow.advise),
    4: ("Level 4: Tool-Using LLM", tools.investigate),
    5: ("Level 5: Multi-Step Reasoning", reasoning.reason),
    6: ("Level 6: Autonomous Agent", agent.pursue),
}

COLUMNS = ("Surface", "Rung", "Rate", "n", "Input tokens", "Scored by")


def surface_table(pick: int) -> str:
    incidents = corpus()
    observed = {level: observe(rung, incidents, pick) for level, (_, rung) in RUNGS.items()}
    observed_claims = claims(incidents)
    rows = []
    for surface in CLAIM_SURFACES + SURFACES:
        source = observed_claims if surface in CLAIM_SURFACES else observed[surface.level]
        result = score(surface, source)
        rate = result.rate.render() if result.rate is not None else "not measured"
        n = str(result.rate.n) if result.rate is not None else "0"
        tokens = f"{result.input_tokens:,}" if result.input_tokens else "-"
        rows.append(
            (
                surface.name,
                RUNGS[surface.level][0],
                rate,
                n,
                tokens,
                result.scored_by,
            )
        )
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def pinning_report() -> str:
    """What changes when the upstream stage replays a different recorded sample.

    Both samples are real answers to the same prompt from the same model, drawn
    on different days. Nothing else about the pipeline differs - not a prompt,
    not a line of code, not a fixture.
    """
    incidents = corpus()
    lines = [
        "| Upstream sample | Pipelines completing all 5 stages | First stage to fail | "
        "Downstream recording missing |",
        "|---|---|---|---|",
    ]
    for pick, label in ((0, "Chapter 4's draw"), (1, "Chapter 6's draw")):
        observations = observe(workflow.advise, incidents, pick)
        complete = sum(
            1 for o in observations if len(getattr(o.result, "trace", ())) == 5
            and all(s.ok for s in o.result.trace)
        )
        failed = sorted(
            {s.name for o in observations for s in getattr(o.result, "trace", ()) if not s.ok}
        )
        missing = sum(1 for o in observations if o.missing)
        lines.append(
            f"| {label} | {complete} of {len(observations)} | "
            f"{', '.join(failed) or 'none'} | {missing} of {len(observations)} |"
        )
    return "\n".join(lines)


def sizing_report() -> str:
    lines = ["| Lift to detect | Cases per arm |", "|---|---|"]
    for lift in (0.30, 0.20, 0.10, 0.05):
        lines.append(f"| {int(lift * 100)} points | {samples_needed(0.60, lift):,} |")
    lines.append("")
    for n in (6, 30, 100):
        lines.append(
            f"- {n} cases per arm detects a lift of "
            f"{detectable_lift(n, 0.60) * 100:.0f} points or larger."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score every rung on every surface.")
    parser.add_argument("--pick", type=int, default=LATEST,
                        help="which recorded sample to replay (default: the latest)")
    parser.add_argument("--pinning", action="store_true",
                        help="show what an upstream sample choice costs downstream")
    parser.add_argument("--sizing", action="store_true",
                        help="show how large an eval set must be")
    args = parser.parse_args(argv)

    if args.pinning:
        print(pinning_report())
        return 0
    if args.sizing:
        print(sizing_report())
        return 0

    incidents = corpus()
    raw, floored = floor_effect(incidents)
    print(surface_table(args.pick))
    print()
    print(f"severity before the floor: {raw.render()}")
    print(f"severity after the floor:  {floored.render()}")
    print()
    print("| Ask | Right at the rung it stopped on | Re-ask rate | Levels reached |")
    print("|---|---|---|---|")
    for ask in ("page", "advise", "diagnose"):
        rate, reached = re_ask(incidents, ask, args.pick)
        spread = ", ".join(f"L{lvl}x{count}" for lvl, count in sorted(reached.items()))
        missed = rate.n - rate.passes
        print(f"| {ask} | {rate.render()} | {missed}/{rate.n} | {spread} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
