"""Does memory layer three pay for itself?

Most teams build the scratchpad and stop. Chapter 9 argues the other two layers
are what you need to OPERATE a loop rather than run one - and the third layer,
recall across runs, is the one with an actual cost claim attached: the second
investigation of a similar incident should be cheaper than the first.

That claim is measurable and this script measures it, in three phases.

  --build      run the corpus with no memory, writing each run to a journal.
               This is memory layer two doing its job: the record outlives the
               process that made it.
  --calibrate  put the recall floor where Chapter 5 put the retrieval floor -
               above the best score any query the corpus CANNOT answer achieves.
               Needs no key and spends nothing, because a journal is just text.
  --compare    run the corpus again WITH the journal, and diff rounds, tokens,
               and routing against the no-memory arm.

The honest possibility is that it does not pay. Six incidents with distinct
causes is a corpus where prior runs have little to say to each other, and a null
result here is a finding rather than a failed experiment. The more interesting
possibility is that it pays negatively: two of these incidents share a service,
and a confident summary of the wrong one is a prior the loop can spend rounds
confirming. Both outcomes are read off the same table.

Usage:

    python scripts/agent_memory.py --build --samples 2      # needs a key
    python scripts/agent_memory.py --calibrate              # free
    python scripts/agent_memory.py --compare --samples 3    # needs a key
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from escalation_ladder.agent import (
    RECALL_FLOOR,
    Budget,
    Journal,
    pursue,
    recall,
)
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.fixtures.retrieval_labels import UNANSWERABLE
from escalation_ladder.llm import AnthropicCompleter
from escalation_ladder.retrieve import Passage, build_index, search

INCIDENTS = ("INC-1042", "INC-1043", "INC-1044", "INC-1045", "INC-1046")
DEFAULT_JOURNAL = Path("runs") / "journal.jsonl"
ABORT_AFTER = 5


@dataclass
class Arm:
    """One arm of the comparison: with memory, or without."""

    name: str
    rounds: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    routed: int = 0

    @property
    def runs(self) -> int:
        return len(self.rounds)

    def report(self) -> dict:
        if not self.runs:
            return {"arm": self.name, "runs": 0, "cost": "not measured"}
        return {
            "arm": self.name,
            "runs": self.runs,
            "routed": f"{self.routed}/{self.runs}",
            "mean_rounds": round(statistics.fmean(self.rounds), 2),
            "mean_tokens": round(statistics.fmean(self.tokens)),
        }


def _run_corpus(
    arm: Arm,
    samples: int,
    budget: Budget,
    journal: Journal | None,
    floor: float,
    *,
    write_to: Journal | None = None,
) -> Arm:
    incidents = {i.incident_id: i for i in load_incidents()}
    completer = AnthropicCompleter()
    consecutive = 0

    for incident_id in INCIDENTS:
        for sample in range(samples):
            result = pursue(
                incidents[incident_id],
                completer,
                budget=budget,
                journal=journal,
                recall_floor=floor,
            )
            why = result.decision.decided_by or ""
            if not result.routed and why.startswith("api error"):
                consecutive += 1
                print(f"  API FAILURE {why[:100]}", flush=True)
                if consecutive >= ABORT_AFTER:
                    raise SystemExit(
                        f"aborted after {consecutive} consecutive API failures; "
                        f"nothing is being concluded. Last: {why}"
                    )
                continue
            consecutive = 0

            arm.rounds.append(result.rounds)
            arm.tokens.append(
                result.ledger.total_input_tokens + result.ledger.total_output_tokens
            )
            if result.routed:
                arm.routed += 1
            if write_to is not None:
                write_to.append(
                    result.entry(
                        f"{incident_id}-{sample}",
                        report=incidents[incident_id].report,
                    )
                )
            print(
                json.dumps(
                    {
                        "arm": arm.name,
                        "incident": incident_id,
                        "rounds": result.rounds,
                        "routed": result.routed,
                        "recalled": [e.incident_id for e in result.recalled],
                    }
                ),
                flush=True,
            )
    return arm


def calibrate(journal: Journal) -> float | None:
    """Chapter 5's method, applied to a corpus of investigations.

    BM25 scores are not portable across corpora - Chapter 5 says so in prose and
    the manifest repeats it - so the runbook floor of 6.0 means nothing here. The
    procedure is the portable part: score queries the corpus cannot answer, and
    put the line above the best of them.
    """
    entries = journal.entries()
    if not entries:
        print("journal is empty; run --build first")
        return None
    passages = tuple(
        Passage(f"run:{e.run_id}", e.incident_id, e.searchable()) for e in entries
    )
    index = build_index(passages)

    best = 0.0
    print(f"scoring {len(UNANSWERABLE)} off-topic queries against "
          f"{len(passages)} journal entries")
    for query in UNANSWERABLE:
        hits = search(index, query, limit=1, floor=0.0)
        top = hits[0].score if hits else 0.0
        best = max(best, top)
        print(f"  {top:6.2f}  {query}")

    incidents = {i.incident_id: i for i in load_incidents()}
    print("\nand the real reports, for contrast")
    for incident_id in INCIDENTS:
        hits = search(index, incidents[incident_id].report, limit=1, floor=0.0)
        top = hits[0].score if hits else 0.0
        source = hits[0].passage.source if hits else "-"
        print(f"  {top:6.2f}  {incident_id} -> best match {source}")

    print(f"\nbest off-topic score: {best:.2f}")
    print(f"shipped RECALL_FLOOR:  {RECALL_FLOOR:.2f}")
    if RECALL_FLOOR <= best:
        print(
            "  THE SHIPPED FLOOR IS TOO LOW for this journal. A question about "
            "the coffee machine would retrieve an investigation."
        )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=Budget().iterations)
    parser.add_argument("--floor", type=float, default=RECALL_FLOOR)
    parser.add_argument("--journal", type=str, default=str(DEFAULT_JOURNAL))
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    journal = Journal(path=Path(args.journal))
    budget = Budget(iterations=args.iterations)
    report: dict = {}

    if args.build:
        print(f"building {journal.path} - no memory read during this pass")
        arm = _run_corpus(
            Arm("no memory"), args.samples, budget, None, args.floor,
            write_to=journal,
        )
        report["no_memory"] = arm.report()
        print(json.dumps(arm.report()))

    if args.calibrate:
        report["best_off_topic_score"] = calibrate(journal)

    if args.compare:
        entries = journal.entries()
        if not entries:
            raise SystemExit("journal is empty; run --build first")
        print(f"\ncomparing against {len(entries)} journal entries")
        without = _run_corpus(Arm("no memory"), args.samples, budget, None, args.floor)
        with_memory = _run_corpus(
            Arm("with memory"), args.samples, budget, journal, args.floor
        )
        report["no_memory"] = without.report()
        report["with_memory"] = with_memory.report()

        print()
        print(json.dumps(without.report()))
        print(json.dumps(with_memory.report()))
        if without.runs and with_memory.runs:
            rounds = statistics.fmean(with_memory.rounds) - statistics.fmean(
                without.rounds
            )
            tokens = statistics.fmean(with_memory.tokens) - statistics.fmean(
                without.tokens
            )
            print()
            print(f"memory changed mean rounds by {rounds:+.2f}")
            print(f"memory changed mean tokens by {tokens:+.0f}")
            print(
                f"routing: {without.routed}/{without.runs} without, "
                f"{with_memory.routed}/{with_memory.runs} with"
            )
            print(
                "\nA null result is a result. So is a negative one: a confident "
                "summary of a different incident on the same service is a prior "
                "the loop can spend rounds confirming."
            )
        else:
            print("\none arm is unmeasured; no comparison is available")

    if not (args.build or args.calibrate or args.compare):
        parser.error("choose at least one of --build, --calibrate, --compare")

    if args.json and report:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
