"""What does a checkpointer save when a loop dies mid-run?

Chapter 8 concluded that LangGraph was not yet earned at Level 5 and named the
exact receipt that would earn it: resumption, measured against the cost of a cold
re-run. This is that measurement, and Chapter 9 is where it was always going to
be collected - a three-step chain has almost nothing worth resuming, while a loop
that has spent six rounds reading telemetry has six rounds of state that a crash
otherwise deletes.

The experiment kills a run at round K by raising out of the node, then re-enters
with the same thread id and lets it finish. Two numbers come out:

    saved       rounds the resumed run did NOT repeat
    cost        what those rounds cost the first time, in tokens and seconds

Multiply the second by how often your loops actually die and you have the value
of the checkpointer. Compare it against 22 packages and 26 MB - Chapter 8's
measured footprint - and the decision is arithmetic rather than taste.

This runs entirely on a scripted completer: no key, no network, no spend. The
thing under test is the framework's state handling, not the model's, and paying
a vendor to demonstrate a `for` loop restarting would measure nothing extra.

ONE HONEST LIMIT, stated here because the chapter states it too. `InMemorySaver`
is what `langgraph-checkpoint` ships, and it lives in the process that is about
to die. Surviving an actual process kill needs a durable checkpointer, which is a
further package that this book does not install - so what is measured below is
resumption after an exception, and the more expensive failure is one dependency
further away than the framework's own documentation makes it look.

Usage:

    python scripts/resume.py
    python scripts/resume.py --rounds 8 --die-at 5
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace

from escalation_ladder.agent import AgentState, Attempt, Budget, Verdict, pursue
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import ToolCall, ToolRun, Usage
from escalation_ladder.orchestration import LangGraphLoop, Node, WhileLoop

INCIDENT = "INC-1044"
PER_ROUND = Usage(input_tokens=5_200, output_tokens=550)


@dataclass
class Scripted:
    """A completer that settles on the ABSOLUTE round, counting what it served.

    The distinction is the instrument. An earlier version settled on its own
    call count, so a resumed run - which gets a fresh completer, exactly as a
    restarted process would - needed the full six calls again no matter how much
    state had been restored, and the experiment reported that the checkpointer
    saved nothing. It was measuring the fake.

    So the round number is read out of the working set the agent was handed,
    which is the same place the model would read it from. `served` then counts
    only the rounds this process actually paid for.
    """

    settle_at: int
    served: int = 0
    usage: Usage = field(default_factory=lambda: PER_ROUND)

    def parse(self, *, system, user, schema, effort="low"):
        raise AssertionError("this rung does not use parse")

    def invoke(self, *, system, user, tools, execute, schema, effort="low"):
        self.served += 1
        # Rounds already completed, according to the restored state rather than
        # according to this object.
        completed = user.count("- round ")
        call = ToolCall(
            f"toolu_{completed + 1}_{self.served}",
            "search_logs",
            {"source": "service-mesh", "pattern": "inventory", "minutes": 60},
        )
        results = (execute(call),)
        settled = completed + 1 >= self.settle_at
        verdict = Verdict(
            service="checkout-api",
            severity="SEV1",
            cause="The mesh certificate expired.",
            evidence_tool="search_logs",
            evidence_value="x509 certificate has expired",
            next_step=(
                "Renew the certificate."
                if settled
                else "Read the next thing."
            ),
            needs_human=not settled,
            unreachable=False,
        )
        return ToolRun(verdict, self.usage, (call,), results)


def measure(loop_name: str, settle_at: int, die_at: int) -> dict:
    """One arm: die at `die_at`, then resume, and count what was re-served."""
    incidents = {i.incident_id: i for i in load_incidents()}
    incident = incidents[INCIDENT]
    budget = Budget(iterations=settle_at + 4)

    cold = Scripted(settle_at=settle_at)
    baseline = pursue(incident, cold, budget=budget, loop=WhileLoop())

    if loop_name == "none":
        # The no-framework arm. There is no state to resume from, so a crash
        # costs the whole run - which is the number the checkpointer competes
        # against rather than a strawman.
        restarted = Scripted(settle_at=settle_at)
        after = pursue(incident, restarted, budget=budget, loop=WhileLoop())
        return {
            "arm": "cold restart (no checkpointer)",
            "rounds_to_settle": baseline.rounds,
            "rounds_re_served_after_crash": after.rounds,
            "rounds_saved": 0,
        }

    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    loop = LangGraphLoop(checkpointer=saver, thread=INCIDENT)
    dying = Scripted(settle_at=settle_at)
    real_invoke = dying.invoke

    def crashing(**kwargs):
        if dying.served >= die_at:
            raise RuntimeError(f"process died after round {dying.served}")
        return real_invoke(**kwargs)

    dying.invoke = crashing  # type: ignore[method-assign]
    try:
        pursue(incident, dying, budget=budget, loop=loop)
    except RuntimeError:
        pass
    died_after = dying.served

    resumed = Scripted(settle_at=settle_at)
    # `resume=True` invokes with None so the saved state stands. Without it the
    # same call re-runs the whole investigation and returns the same answer -
    # see the note in `LangGraphLoop`.
    pursue(
        incident,
        resumed,
        budget=budget,
        loop=LangGraphLoop(checkpointer=saver, thread=INCIDENT, resume=True),
    )

    saved = max(0, baseline.rounds - resumed.served)
    return {
        "arm": "langgraph InMemorySaver",
        "rounds_to_settle": baseline.rounds,
        "died_after_round": died_after,
        "rounds_re_served_on_resume": resumed.served,
        "rounds_saved": saved,
        "tokens_not_repurchased": saved
        * (PER_ROUND.input_tokens + PER_ROUND.output_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--die-at", type=int, default=4)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    if args.die_at >= args.rounds:
        parser.error("--die-at must be less than --rounds, or nothing crashes")

    print(f"a {args.rounds}-round investigation, killed after round {args.die_at}\n")

    arms = [measure("none", args.rounds, args.die_at)]
    try:
        arms.append(measure("langgraph", args.rounds, args.die_at))
    except ModuleNotFoundError:
        print("langgraph extra not installed; only the cold-restart arm ran")

    for arm in arms:
        print(json.dumps(arm, sort_keys=True))

    print()
    print(
        "The checkpointer is worth installing when rounds_saved times your "
        "mid-run death rate exceeds what 22 packages cost you to carry. Note "
        "that InMemorySaver does not survive the process - see this script's "
        "docstring."
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"arms": arms}, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
