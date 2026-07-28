"""How many steps does each incident actually need?

Chapter 8's Failure Receipt, computed rather than argued. The rung ships a
three-step chain because three is what INC-1044 needs. This script runs the same
code over the same corpus at every depth from one step to the full chain and
reports, per incident, the depth at which it first routes.

If every incident needed the same depth, a fixed chain would be the right
architecture and Level 6 would be an indulgence. The interesting result is the
other one: the required depth varies per incident, is not predictable from the
report, and for at least one incident there is no depth in range that suffices.
That is the Termination Test failing on a real corpus - you cannot state the
number of steps in advance because the number is a property of what the
telemetry turns out to say - and it is what earns Chapter 9.

Note what `--depth` is NOT. It does not cap an iteration count at run time; it
slices `reasoning.STEPS`, so each arm is a different architecture with its own
maximum, fixed before its run begins. Every arm passes the Termination Test
individually. The receipt is that no single one of them is right for the corpus.

Usage (needs ANTHROPIC_API_KEY):

    python scripts/chain_depth.py                  # 3 samples per incident/depth
    python scripts/chain_depth.py --samples 10
    python scripts/chain_depth.py --json out.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field

from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import AnthropicCompleter
from escalation_ladder.reasoning import STEPS, reason

INCIDENTS = ("INC-1042", "INC-1043", "INC-1044", "INC-1045", "INC-1046")


# Stop rather than keep paying for a session the vendor is refusing.
ABORT_AFTER = 5


@dataclass
class Cell:
    """One incident at one depth.

    `runs` counts SUCCESSFUL runs only. A vendor failure is not a measurement,
    and this script learned that the hard way: an earlier version counted them,
    and on a run where every call failed it printed "at least one incident routes
    at no depth in range" - the Failure Receipt's conclusion, stated with total
    confidence, from forty-five runs that never reached a model. Zero routed out
    of zero measured must be reported as "not measured", never as zero percent.
    """

    incident_id: str
    depth: int
    runs: int = 0
    routed: int = 0
    exhausted: int = 0
    failed: int = 0
    api_requests: list[int] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    wanted: list[str] = field(default_factory=list)

    def report(self) -> dict:
        if not self.runs:
            return {
                "incident": self.incident_id,
                "depth": self.depth,
                "routed": "not measured",
                "api_failures": self.failed,
            }
        return {
            "incident": self.incident_id,
            "depth": self.depth,
            "routed": f"{self.routed}/{self.runs}",
            "exhausted": f"{self.exhausted}/{self.runs}",
            "api_failures": self.failed,
            "mean_api_requests": round(_mean(self.api_requests), 2),
            "mean_input_tokens": round(_mean(self.input_tokens)),
            # What the chain still wanted when it ran out. This is the text the
            # Failure Receipt prints, so it is kept verbatim rather than counted.
            "still_wanted": self.wanted[:3],
        }


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def sweep(samples: int) -> list[Cell]:
    incidents = {i.incident_id: i for i in load_incidents()}
    completer = AnthropicCompleter()
    cells: list[Cell] = []
    consecutive = 0

    for depth in range(1, len(STEPS) + 1):
        for incident_id in INCIDENTS:
            cell = Cell(incident_id=incident_id, depth=depth)
            for _ in range(samples):
                result = reason(
                    incidents[incident_id], completer, steps=STEPS[:depth]
                )
                why = result.decision.decided_by
                if not result.routed and why.startswith("api error"):
                    cell.failed += 1
                    consecutive += 1
                    print(f"  API FAILURE {why[:100]}", flush=True)
                    if consecutive >= ABORT_AFTER:
                        raise SystemExit(
                            f"aborted after {consecutive} consecutive API "
                            f"failures; nothing is being concluded. Last: {why}"
                        )
                    continue
                consecutive = 0

                cell.runs += 1
                cell.api_requests.append(result.api_requests)
                cell.input_tokens.append(result.ledger.total_input_tokens)
                if result.routed:
                    cell.routed += 1
                if result.exhausted:
                    cell.exhausted += 1
                    if "next check was:" in why:
                        cell.wanted.append(
                            why.split("next check was:", 1)[1].strip()
                        )
            cells.append(cell)
            print(json.dumps(cell.report()), flush=True)
    return cells


def first_routing_depth(cells: list[Cell]) -> dict[str, int | None]:
    """The depth at which each incident first routes on a majority of runs.

    A majority rather than any single run, because Chapter 4 established that one
    pass over a non-deterministic system is an anecdote. An incident that never
    reaches a majority at any depth maps to `None`, and that is the receipt.
    """
    best: dict[str, int | None] = {incident: None for incident in INCIDENTS}
    by_incident: dict[str, list[Cell]] = defaultdict(list)
    for cell in cells:
        by_incident[cell.incident_id].append(cell)
    for incident_id, incident_cells in by_incident.items():
        for cell in sorted(incident_cells, key=lambda c: c.depth):
            if cell.runs and cell.routed * 2 > cell.runs:
                best[incident_id] = cell.depth
                break
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    cells = sweep(args.samples)

    measured = [cell for cell in cells if cell.runs]
    if not measured:
        print("\nNOTHING MEASURED. No conclusion is available from this run.")
        return
    unmeasured = {cell.incident_id for cell in cells if not cell.runs}

    depths = first_routing_depth(cells)

    print()
    print("first depth that routes a majority of runs")
    for incident_id, depth in depths.items():
        if incident_id in unmeasured and depth is None:
            # "Never" and "we did not get to ask" are different answers, and
            # only one of them is evidence.
            print(f"  {incident_id}  not measured")
            continue
        print(f"  {incident_id}  {depth if depth is not None else 'never'}")

    needed = [d for d in depths.values() if d is not None]
    print()
    if needed and len(set(needed)) > 1:
        print(
            f"required depth varies from {min(needed)} to {max(needed)} across "
            f"{len(needed)} incidents"
        )
    never = [
        i for i, d in depths.items() if d is None and i not in unmeasured
    ]
    if never:
        print(
            f"routes at no depth in range: {', '.join(never)} - for these the "
            f"chain length is not a property you can fix in advance"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "cells": [cell.report() for cell in cells],
                    "first_routing_depth": depths,
                    "samples": args.samples,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
