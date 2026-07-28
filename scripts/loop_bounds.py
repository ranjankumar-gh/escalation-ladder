"""What does a run cost, grouped by how it ended?

Chapter 9's central measurement. Every rung below Level 6 charged roughly the
same for a yes and a no: Level 0 refused in microseconds, Level 4 refused after
the one round it was always going to take, and Level 5 refused after at most
`2 * len(STEPS)` requests. A loop cannot do that. A run that succeeds exits as
soon as it can cite the cause; a run that cannot succeed keeps finding plausible
next reads until something outside it says stop.

So the interesting number is not the mean. It is the mean of each group, and the
ratio between them - the Refusal Premium. This script computes it by tagging
every run with how it ended:

    routed      the agent cited a cause and a rota was chosen
    gave-up     the agent set `unreachable` and stopped itself
    bounded     a limit in `Budget` fired, named individually
    failed      the vendor failed; not a measurement, counted separately

The second question is in the same data. `unreachable` is the only exit that is
neither a success nor a circuit breaker, and it is the cheapest possible answer
to an incident telemetry cannot solve. Whether a model actually uses it is a
measurement rather than a design decision, and the gave-up count is where to
read it.

Usage:

    python scripts/loop_bounds.py --replay            # free: no key, no spend
    python scripts/loop_bounds.py                     # live, 3 samples/incident
    python scripts/loop_bounds.py --samples 10
    python scripts/loop_bounds.py --iterations 12
    python scripts/loop_bounds.py --json out.json

`--replay` runs the shipped recording of the chapter's own measurement pass, so
the outcome of every run and the ROUND counts behind the premium regenerate for
nothing. Token and latency figures do not: the fake bills a fixed usage and never
touches a network, so those two columns are honestly absent rather than wrong.
Level 6 is the most expensive rung in this book to measure live, which is
precisely why the free path exists.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from escalation_ladder.agent import Budget, DryRunToolbox, Pursuit, pursue
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import AnthropicCompleter

INCIDENTS = ("INC-1042", "INC-1043", "INC-1044", "INC-1045", "INC-1046")

# Stop rather than keep paying for a session the vendor is refusing. Third
# chapter running with this guard, and it was earned three separate times - see
# the measurement-honesty note in the book's manifest.
ABORT_AFTER = 5


def outcome_of(result: Pursuit) -> str:
    """One of four buckets. The bound's own name is kept, not flattened."""
    if result.routed:
        return "routed"
    if result.gave_up is not None:
        return "gave-up"
    if result.stopped_by is not None:
        return f"bounded ({result.stopped_by.split(':')[0]})"
    return "stopped"


@dataclass
class Group:
    """Runs that ended the same way, and what each of them cost.

    `runs` counts SUCCESSFUL API interactions only. A run that died on a vendor
    error tells you about the vendor, and averaging it into a cost row is the
    defect this book has now found in its own tooling three times.
    """

    name: str
    rounds: list[int] = field(default_factory=list)
    requests: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.rounds)

    def report(self) -> dict:
        if not self.runs:
            return {"outcome": self.name, "runs": 0, "cost": "not measured"}
        return {
            "outcome": self.name,
            "runs": self.runs,
            "mean_rounds": round(statistics.fmean(self.rounds), 2),
            "mean_api_requests": round(statistics.fmean(self.requests), 2),
            "mean_tokens": round(statistics.fmean(self.tokens)),
            "p50_latency_s": round(statistics.median(self.latency_ms) / 1000.0, 1),
            "max_latency_s": round(max(self.latency_ms) / 1000.0, 1),
        }


def sweep(
    samples: int,
    budget: Budget,
    *,
    incidents_to_run: tuple[str, ...] = INCIDENTS,
    honor_unreachable: bool = True,
    allow_writes: bool = False,
) -> tuple[dict[str, Group], dict, int, list[str]]:
    incidents = {i.incident_id: i for i in load_incidents()}
    completer = AnthropicCompleter()
    proposed_writes: list[str] = []
    groups: dict[str, Group] = {}
    per_incident: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failures = 0
    consecutive = 0

    for incident_id in incidents_to_run:
        for _ in range(samples):
            # A dry run offers the write tool and performs nothing, so what it
            # records is what a write-capable agent would have done to
            # production. See `DryRunToolbox` - the result it returns does not
            # admit to being a dry run, on purpose.
            box = (
                DryRunToolbox(incident=incidents[incident_id], allow_writes=True)
                if allow_writes
                else None
            )
            result = pursue(
                incidents[incident_id],
                completer,
                budget=budget,
                honor_unreachable=honor_unreachable,
                allow_writes=allow_writes,
                box=box,
            )
            for call in result.proposed:
                proposed_writes.append(f"round-unknown {call.render()}")
                print(f"  PROPOSED WRITE {call.render()}", flush=True)
            why = result.decision.decided_by or ""
            if not result.routed and why.startswith("api error"):
                failures += 1
                consecutive += 1
                print(f"  API FAILURE {why[:100]}", flush=True)
                if consecutive >= ABORT_AFTER:
                    raise SystemExit(
                        f"aborted after {consecutive} consecutive API failures; "
                        f"nothing is being concluded. Last: {why}"
                    )
                continue
            consecutive = 0

            name = outcome_of(result)
            group = groups.setdefault(name, Group(name=name))
            group.rounds.append(result.rounds)
            group.requests.append(result.api_requests)
            group.tokens.append(
                result.ledger.total_input_tokens + result.ledger.total_output_tokens
            )
            group.latency_ms.append(result.ledger.total_latency_ms)
            per_incident[incident_id][name] += 1
            print(
                json.dumps(
                    {
                        "incident": incident_id,
                        "outcome": name,
                        "rounds": result.rounds,
                        "requests": result.api_requests,
                        "tokens": group.tokens[-1],
                        # Kept verbatim rather than counted. A run that ended
                        # without routing, without giving up, and without a
                        # bound ended because the seam reported something, and
                        # WHICH something is the whole finding.
                        "why": why[:120],
                    }
                ),
                flush=True,
            )
    return groups, dict(per_incident), failures, proposed_writes


def replay(budget: Budget) -> tuple[dict[str, Group], dict]:
    """Re-run the chapter's measurement from its recording. No key, no spend.

    The tools execute for real against the deterministic fixtures - Chapter 7's
    reason for recording only the model's half - so a replayed run makes the same
    choices, reads the same values, and ends the same way. What it cannot
    reproduce is what those rounds cost, because the fake bills a constant.
    """
    from escalation_ladder.llm import RecordedCompleter

    path = (
        Path(__file__).resolve().parent.parent
        / "tests" / "recordings" / "ch09_agent.json"
    )
    if not path.exists():
        raise SystemExit(f"no recording at {path}")
    recordings = json.loads(path.read_text(encoding="utf-8"))

    incidents = {i.incident_id: i for i in load_incidents()}
    groups: dict[str, Group] = {}
    per_incident: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for incident_id in INCIDENTS:
        result = pursue(
            incidents[incident_id],
            RecordedCompleter(recordings=recordings),
            budget=budget,
        )
        why = result.decision.decided_by or ""
        if "no recording for" in why:
            print(f"  {incident_id}: not in the recording, skipped", flush=True)
            continue
        name = outcome_of(result)
        group = groups.setdefault(name, Group(name=name))
        group.rounds.append(result.rounds)
        group.requests.append(result.api_requests)
        group.tokens.append(0)
        group.latency_ms.append(0.0)
        per_incident[incident_id][name] += 1
        print(
            json.dumps(
                {"incident": incident_id, "outcome": name, "rounds": result.rounds}
            ),
            flush=True,
        )
    return groups, dict(per_incident)


def premium(groups: dict[str, Group]) -> float | None:
    """Mean tokens of every run that did not answer, over runs that did.

    Above 1.0 means the system spends more on the incidents it cannot help with
    than on the ones it can. There is no rung below this one where that number
    can exceed 1.0, because no rung below can choose to keep going.
    """
    answered = groups.get("routed")
    if answered is None or not answered.runs:
        return None
    unanswered = [
        token
        for name, group in groups.items()
        if name != "routed"
        for token in group.tokens
    ]
    base = statistics.fmean(answered.tokens) if answered.tokens else 0.0
    if not unanswered or base == 0.0:
        # Replay bills a constant, so the token ratio is not available. The
        # rounds ratio still is, and it is reported instead rather than a
        # number computed from zeros - the third time this book has had to say
        # that an unmeasured arm is not a zero.
        return None
    return statistics.fmean(unanswered) / base


def premium_in_rounds(groups: dict[str, Group]) -> float | None:
    """The same ratio, counted in rounds. Regenerates from a replay for free."""
    answered = groups.get("routed")
    if answered is None or not answered.rounds:
        return None
    unanswered = [
        count
        for name, group in groups.items()
        if name != "routed"
        for count in group.rounds
    ]
    if not unanswered:
        return None
    return statistics.fmean(unanswered) / statistics.fmean(answered.rounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=Budget().iterations)
    parser.add_argument("--tokens", type=int, default=Budget().tokens)
    parser.add_argument("--seconds", type=float, default=Budget().seconds)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--no-quit",
        action="store_true",
        help="ignore the model's `unreachable`, so only a bound can stop a run",
    )
    parser.add_argument("--only", type=str, default="")
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="offer the write tool in a dry run and record what it proposes",
    )
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    budget = Budget(
        iterations=args.iterations, tokens=args.tokens, seconds=args.seconds
    )
    targets = (
        tuple(p.strip() for p in args.only.split(",") if p.strip())
        if args.only
        else INCIDENTS
    )
    if args.replay:
        groups, per_incident = replay(budget)
        failures, proposed = 0, []
    else:
        groups, per_incident, failures, proposed = sweep(
            args.samples,
            budget,
            incidents_to_run=targets,
            honor_unreachable=not args.no_quit,
            allow_writes=args.allow_writes,
        )

    measured = sum(group.runs for group in groups.values())
    if not measured:
        print("\nNOTHING MEASURED. No conclusion is available from this run.")
        print(f"api failures: {failures}")
        return

    print()
    print(f"budget: {budget}")
    print(f"measured runs: {measured}   api failures: {failures}")
    print()
    for group in sorted(groups.values(), key=lambda g: -g.runs):
        print(json.dumps(group.report()))

    ratio = premium(groups)
    rounds_ratio = premium_in_rounds(groups)
    print()
    if ratio is None:
        print("refusal premium (tokens): not measured")
    else:
        print(f"refusal premium (tokens): {ratio:.2f}x")
    if rounds_ratio is None:
        print("refusal premium (rounds): not measured")
    else:
        print(f"refusal premium (rounds): {rounds_ratio:.2f}x")
    print(
        "  cost of runs that did NOT answer, over runs that did. Above 1.0 "
        "means the spend concentrates on the incidents the system could not "
        "help with - which no rung below this one can do."
    )

    gave_up = groups.get("gave-up")
    print()
    if args.allow_writes:
        print()
        print(
            f"proposed writes: {len(proposed)} across {measured} runs - see "
            f"the Failure Receipt. Nothing was executed."
        )

    print(
        f"the agent stopped itself on {gave_up.runs if gave_up else 0} of "
        f"{measured} runs"
    )
    print(
        "  `unreachable` is the cheapest correct answer to an incident this "
        "rung cannot solve. How often it is used is a measurement, not a design."
    )

    print()
    print("outcome by incident")
    for incident_id in INCIDENTS:
        counts = per_incident.get(incident_id, {})
        if not counts:
            print(f"  {incident_id}  not measured")
            continue
        rendered = ", ".join(f"{name} {n}" for name, n in sorted(counts.items()))
        print(f"  {incident_id}  {rendered}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "budget": {
                        "iterations": budget.iterations,
                        "tokens": budget.tokens,
                        "seconds": budget.seconds,
                    },
                    "groups": [g.report() for g in groups.values()],
                    "per_incident": per_incident,
                    "refusal_premium": ratio,
                    "api_failures": failures,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
