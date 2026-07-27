"""Does telling the model its round budget change what it does?

Chapter 7's sharpest measurement, and the one that makes the Level 4/5 boundary
empirical rather than definitional. The rung permits exactly one round of tools.
This script runs the same incidents twice - once with that constraint stated in
the system prompt, once with it withheld - and reports what changed.

The seam is identical in both arms. `llm.invoke` runs one round either way; the
only difference is whether the model knows.

Usage (needs ANTHROPIC_API_KEY):

    python scripts/round_budget.py                 # 3 samples per incident
    python scripts/round_budget.py --samples 10

Every other number in Chapter 7 comes from tools/measure_ch07.py in the book
repo. This one lives here because it proves a design choice rather than pricing
the shipped rung, which is where scripts/compare_retrievers.py (Chapter 5) and
scripts/compare_queries.py (Chapter 6) also live.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field

from escalation_ladder import tools as rung
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import AnthropicCompleter

# The shipped prompt with its round budget removed, and nothing else changed.
# Kept as a diff of the real one so it cannot drift away from what ships.
WITHHELD = rung.SYSTEM.replace(
    """You get one round of tools. Every call you make runs together and you receive
every result at once. There is no second round, so choose the set of calls that
will answer the question rather than a first call whose result you plan to build
on. If one round cannot establish the cause, say so rather than guessing.""",
    "Call the tools you need to establish what happened. You can call several "
    "at once.",
)

INCIDENTS = ("INC-1042", "INC-1043", "INC-1044", "INC-1045", "INC-1046")


@dataclass
class Arm:
    label: str
    calls: list[int] = field(default_factory=list)
    wanted_more: int = 0
    routed: int = 0
    runs: int = 0

    def report(self) -> dict:
        mean = sum(self.calls) / len(self.calls) if self.calls else 0.0
        return {
            "arm": self.label,
            "runs": self.runs,
            "mean_tool_calls": round(mean, 2),
            "wanted_more": f"{self.wanted_more}/{self.runs}",
            "routed": f"{self.routed}/{self.runs}",
        }


def run_arm(label: str, system: str, samples: int) -> Arm:
    incidents = {i.incident_id: i for i in load_incidents()}
    completer = AnthropicCompleter()
    arm = Arm(label=label)
    original = rung.SYSTEM
    rung.SYSTEM = system
    try:
        for incident_id in INCIDENTS:
            for _ in range(samples):
                result = rung.investigate(incidents[incident_id], completer)
                arm.runs += 1
                arm.calls.append(len(result.calls))
                arm.wanted_more += int(result.wanted_more)
                arm.routed += int(result.routed)
                print(
                    f"  {label:<9} {incident_id} tools={len(result.calls)} "
                    f"more={result.wanted_more} routed={result.routed}",
                    flush=True,
                )
    finally:
        rung.SYSTEM = original
    return arm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    stated = run_arm("stated", rung.SYSTEM, args.samples)
    withheld = run_arm("withheld", WITHHELD, args.samples)

    print()
    print(json.dumps([stated.report(), withheld.report()], indent=2))
    print()
    print(
        "The seam enforced one round in both arms. Only the prompt differed, "
        "and a model that does not know its budget spends it planning a second "
        "round - which is Level 5's behaviour, arrived at by omission."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
