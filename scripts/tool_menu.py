"""Run Chapter 11's tool-menu hypothesis, and refuse to fake it.

Chapter 11 left an open question with a two-part pass condition: put a
`search_runbooks` tool in the Level 4-6 menu, hold `LADDERS['diagnose']` at
(4, 5, 6), and check that INC-1043 converts AND that INC-1044 still reaches
Level 5 rather than settling early on the mesh runbook.

Chapter 15 ships the tool. It does not run the experiment, because running it
costs money this book no longer spends. What this script does is make the
difference between "not run" and "run and green" impossible to blur:

    python -m scripts.tool_menu              # prints what it would cost, and stops
    python -m scripts.tool_menu --live       # needs ANTHROPIC_API_KEY, spends

The refusal is the point. `RecordedCompleter` is keyed on the system and user
prompts and NOT on the tool menu, so replaying this experiment does not miss -
it returns answers produced when the tool was not offered, and reports them as
though the tool had been. A harness that cannot tell those apart is worse than
no harness.

Registers no rung and appears in no cross-rung cost table.
"""
from __future__ import annotations

import argparse
import json
import os

from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import AnthropicCompleter, _wire
from escalation_ladder.router import LADDERS, handle
from escalation_ladder.tools import SEARCH_RUNBOOKS, menu_for

# Chapter 11 named both halves. One is the target and one is the neighbor, and
# a capability change that converts the target while regressing the neighbor is
# the normal outcome rather than the surprising one.
TARGET = "INC-1043"
NEIGHBOR = "INC-1044"


def structural_cost() -> dict[str, int]:
    """What the addition costs before anything runs. No key, no network."""
    base = json.dumps([_wire(spec) for spec in menu_for()])
    wide = json.dumps([_wire(spec) for spec in menu_for(allow_runbooks=True)])
    return {
        "tools_before": len(menu_for()),
        "tools_after": len(menu_for(allow_runbooks=True)),
        "schema_chars_before": len(base),
        "schema_chars_after": len(wide),
        "added_chars": len(wide) - len(base),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    cost = structural_cost()
    print("\n== structural cost of offering search_runbooks ==\n")
    print(f"tools in the menu:  {cost['tools_before']} -> {cost['tools_after']}")
    print(
        f"tool schema:        {cost['schema_chars_before']:,} -> "
        f"{cost['schema_chars_after']:,} chars "
        f"(+{cost['added_chars']:,}, about {cost['added_chars'] // 4} tokens)"
    )
    print("paid on every round at Levels 4, 5, and 6, used or not.")
    print(f"diagnose ladder held at {LADDERS['diagnose']}, unchanged.")
    print(f"blast radius: unchanged, because {SEARCH_RUNBOOKS.name} is read-only.")

    print("\n== the empirical half ==\n")
    print(f"pass condition: {TARGET} converts AND {NEIGHBOR} still reaches Level 5.")

    if not args.live:
        print("\nNOT RUN. Replaying this would not miss - the recordings are keyed")
        print("on the prompt and not on the tool menu, so a replay returns answers")
        print("produced when the tool was never offered. Pass --live to spend.")
        return

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("\nREFUSED: --live needs ANTHROPIC_API_KEY. Nothing was run.")
        return

    incidents = {i.incident_id: i for i in load_incidents()}
    # The widened menu goes to every rung on the diagnose ladder, and the ladder
    # itself is untouched. That is Chapter 11's condition: a capability change,
    # not a routing change, so the two can be told apart afterwards.
    widened = {"tools": menu_for(allow_runbooks=True)}
    options = {level: widened for level in LADDERS["diagnose"]}

    print("\nrunning live against claude-opus-5, one ladder per incident\n")
    for incident_id in (TARGET, NEIGHBOR):
        outcome = handle(
            incidents[incident_id],
            ask="diagnose",
            completer=AnthropicCompleter(),
            options=options,
        )
        answered = outcome.decision.owner is not None
        print(
            f"{incident_id}: level_reached={outcome.level_reached} "
            f"answered={answered} attempts={len(outcome.attempts)}"
        )
    print("\nBoth lines must hold. One of two is a regression, not a result.")


if __name__ == "__main__":
    main()
