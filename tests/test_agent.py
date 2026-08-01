"""Level 6 - the autonomous agent.

The behaviours this chapter argues for, asserted structurally:

- there is no expression anywhere in `agent.py` that yields the maximum number
  of model calls, and the only thing that stops an unproductive run is `Budget` -
  the Termination Test failing, as a test;
- the three bounds are three, and each one names itself, so "which bound keeps
  firing" stays an answerable question;
- a run that STOPPED and a run that WAS STOPPED are distinguishable, including
  when the agent settles on its last permitted round;
- the loop runs the same node repeatedly, which is precisely the assertion
  `test_a_node_never_runs_twice` makes about a chain and cannot make about this;
- every round sees the ORIGINAL report, and older findings compact to their
  cause - lossy on purpose, and the first place in this book where the system
  can forget something it paid to read;
- `unreachable` ends a run without a bound firing, and is the only vocabulary
  this rung adds to Chapter 7's schema;
- the journal survives the process and recall excludes the incident under
  investigation, which is Chapter 5's leakage discipline one layer up;
- a dry run records what a write-capable agent would have done to production and
  changes nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from escalation_ladder.agent import (
    CARRY_VERBATIM,
    Attempt,
    AgentState,
    Budget,
    DryRunToolbox,
    Entry,
    Journal,
    Verdict,
    _bound,
    pursue,
    recall,
    run,
    working,
)
from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.instrument import CostLedger, Measurement
from escalation_ladder.llm import ToolCall, ToolRun, Usage
from escalation_ladder.orchestration import WhileLoop
from escalation_ladder.tools import ROLLBACK_DEPLOY, blast_radius, menu_for

INCIDENTS = {i.incident_id: i for i in load_incidents()}
RECORDINGS = Path(__file__).parent / "recordings" / "ch09_agent.json"


def incident(incident_id: str) -> Incident:
    return INCIDENTS[incident_id]


def a_verdict(**overrides) -> Verdict:
    base = dict(
        service="checkout-api",
        severity="SEV1",
        cause="The mesh certificate expired.",
        evidence_tool="search_logs",
        evidence_value="x509 certificate has expired",
        next_step="Renew the certificate.",
        needs_human=False,
        unreachable=False,
    )
    base.update(overrides)
    return Verdict(**base)


CHECKOUT_LOGS = ("search_logs", {
    "source": "checkout-api", "pattern": "error", "minutes": 60,
})
MESH_LOGS = ("search_logs", {
    "source": "service-mesh", "pattern": "inventory", "minutes": 60,
})
DEPLOYS = ("recent_deploys", {"service": "checkout-api"})


@dataclass
class RoundScript:
    """A `Completer` scripted per round, with no fixed length.

    Chapter 8's equivalent asserted that the chain never ran past the end of its
    script, because running past the end WAS the bug worth catching. Here the
    last entry repeats forever instead, because a loop running again is the
    ordinary case and the thing under test is what eventually stops it.

    Tools run for real against the deterministic fixtures, for the reason
    Chapter 7's fake did: a fake that also faked tool output would stay green
    while the tool and the answer drifted apart.
    """

    script: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(100, 20))
    prompts: list[str] = field(default_factory=list)
    round: int = 0

    def parse(self, *, system, user, schema, effort="low"):
        raise AssertionError("this rung does not use parse")

    def invoke(self, *, system, user, tools, execute, schema, effort="low"):
        self.prompts.append(user)
        entry = self.script[min(self.round, len(self.script) - 1)]
        self.round += 1
        made = tuple(
            # The round index is in the call id, so ids stay unique across a run
            # of unknown length - `value_supported` joins on this field.
            ToolCall(f"toolu_{self.round}_{i}", name, args)
            for i, (name, args) in enumerate(entry.get("calls", []))
        )
        if not made:
            return ToolRun(None, self.usage, failed="the model asked for no tools")
        results = tuple(execute(call) for call in made)
        if entry.get("failed"):
            return ToolRun(None, self.usage, made, results, entry["failed"])
        if entry.get("wants_more"):
            return ToolRun(
                None, self.usage, made, results,
                "one round of tools was not enough", wanted_more=True,
            )
        return ToolRun(entry["answer"], self.usage, made, results)


def keeps_going(**overrides) -> dict:
    """A round that establishes something and asks for another."""
    answer = a_verdict(
        needs_human=True,
        service="unknown",
        evidence_value="no healthy upstream for cluster=inventory",
        next_step="Search the mesh for 'inventory'.",
        **overrides,
    )
    return {"calls": [CHECKOUT_LOGS], "answer": answer}


def settles(**overrides) -> dict:
    return {"calls": [MESH_LOGS], "answer": a_verdict(**overrides)}


# ------------------------------------------------- the boundary, as assertions

def test_no_maximum_is_derivable_from_the_module() -> None:
    """The Termination Test, failing, in the only form it can fail in.

    Chapter 8 asserted `MAX_MODEL_CALLS == 2 * len(STEPS)` - a number computed
    from the module's own source. There is no such expression here, and this
    test exists to keep it that way: if a later chapter reintroduces a step list
    it has changed the rung, not tidied the code.
    """
    import escalation_ladder.agent as module

    assert not hasattr(module, "STEPS")
    assert not hasattr(module, "MAX_MODEL_CALLS")
    # The only thing that answers "how many" is a value someone passed in.
    assert Budget().iterations == 8


def test_the_loop_runs_the_same_node_many_times() -> None:
    """The inverse of `test_a_node_never_runs_twice`, which is the point.

    That test is not deleted from the chain's suite - Level 5 still ships and
    Chapter 11 still routes to it. It simply cannot be written here, and this is
    what stands in its place.
    """
    completer = RoundScript(script=[keeps_going(), keeps_going(), settles()])
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=8))

    assert result.rounds == 3
    assert [a.index for a in result.attempts] == [0, 1, 2]
    assert result.routed


def test_each_bound_names_itself() -> None:
    """Three bounds, three strings. A shared boolean would answer none of this."""
    state = AgentState(incident=incident("INC-1044"))
    ledger = CostLedger()

    spent = AgentState(
        incident=state.incident,
        attempts=tuple(Attempt(i, True, "") for i in range(4)),
    )
    assert "iterations: 4 of 4" == _bound(spent, ledger, 0.0, Budget(iterations=4), now=0.0)

    ledger.record(Measurement("agent.0", input_tokens=900, output_tokens=200))
    assert "tokens: 1100 of 1000" == _bound(
        state, ledger, 0.0, Budget(tokens=1000), now=0.0
    )

    assert _bound(
        state, CostLedger(), 0.0, Budget(seconds=30.0), now=31.0
    ) == "deadline: 31s of 30s"

    assert _bound(state, CostLedger(), 0.0, Budget(), now=0.0) is None


def test_the_iteration_bound_stops_an_agent_that_would_not() -> None:
    completer = RoundScript(script=[keeps_going()])
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=3))

    assert result.rounds == 3
    assert not result.routed
    assert result.stopped_by == "iterations: 3 of 3"
    assert "Search the mesh" in (result.decision.decided_by or "")


def test_the_token_bound_stops_a_run_the_iteration_bound_would_not() -> None:
    """The two bounds are not the same bound with different units.

    Eight rounds at 120 tokens is a different run from three rounds at 40,000,
    and a system with only `max_iterations` cannot tell you which one it just
    paid for.
    """
    completer = RoundScript(
        script=[keeps_going()], usage=Usage(input_tokens=5_000, output_tokens=200)
    )
    result = pursue(
        incident("INC-1044"),
        completer,
        budget=Budget(iterations=99, tokens=12_000),
    )

    assert result.stopped_by is not None
    assert result.stopped_by.startswith("tokens:")
    assert result.rounds < 99


def test_settling_on_the_last_permitted_round_is_not_a_bounded_run() -> None:
    """`finished` is checked before the bound, and this is why.

    A run that answers on its last round has used its whole budget and hit
    nothing. Reporting it as bounded would put a success into the bucket the
    chapter uses to count the expensive failures, which is the one number the
    Refusal Premium is read off.
    """
    completer = RoundScript(script=[keeps_going(), settles()])
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=2))

    assert result.rounds == 2
    assert result.routed
    assert result.stopped_by is None


def test_unreachable_ends_the_run_without_a_bound() -> None:
    """The one field this rung adds, doing the only thing it exists to do."""
    completer = RoundScript(
        script=[
            keeps_going(),
            {
                "calls": [MESH_LOGS],
                "answer": a_verdict(
                    needs_human=True,
                    unreachable=True,
                    service="unknown",
                    next_step="The cause is in a runbook, not in telemetry.",
                ),
            },
        ]
    )
    result = pursue(incident("INC-1043"), completer, budget=Budget(iterations=8))

    assert result.rounds == 2
    assert result.stopped_by is None
    assert result.gave_up == "The cause is in a runbook, not in telemetry."
    assert not result.routed


def test_giving_up_does_not_have_to_be_cited() -> None:
    """The check that guards the context also guarded the only voluntary exit.

    A verdict of "nothing here would settle this" asserts no fact, so it has no
    value to quote. The first version of `_round_node` ran `value_supported`
    first, discarded every give-up on that ground, and sent the agent round
    again - turning the one exit that is not a circuit breaker into a circuit
    breaker. This asserts the order that fixed it, using a verdict whose cited
    value is deliberately absent from every tool result.
    """
    completer = RoundScript(
        script=[
            {
                "calls": [CHECKOUT_LOGS],
                "answer": a_verdict(
                    needs_human=True,
                    unreachable=True,
                    evidence_value="nothing that appears in any tool result",
                    next_step="Nothing in telemetry can settle this.",
                ),
            }
        ]
    )
    result = pursue(incident("INC-1043"), completer, budget=Budget(iterations=8))

    assert result.rounds == 1
    assert result.gave_up == "Nothing in telemetry can settle this."
    # It makes no claim, so nothing enters the working set from it.
    assert result.attempts[0].finding is None


# --------------------------------------------------------- the scratchpad

@pytest.mark.parametrize("round_index", [0, 1, 2, 3])
def test_every_round_sees_the_original_report(round_index: int) -> None:
    """Chapter 6's prerequisite, Chapter 8's measured survivor, load-bearing here.

    At round nine the report is the only part of the working set that has not
    been through a model.
    """
    completer = RoundScript(script=[keeps_going()])
    pursue(incident("INC-1044"), completer, budget=Budget(iterations=4))

    prompt = completer.prompts[round_index]
    assert incident("INC-1044").report in prompt


def test_the_model_is_never_told_a_maximum() -> None:
    """Chapter 7 measured that stating a round budget changes what a model does.

    Stating one here would produce a chain with a longer tuple. The absence is
    therefore a design decision, and this asserts it rather than trusting it.
    """
    completer = RoundScript(script=[keeps_going()])
    pursue(incident("INC-1044"), completer, budget=Budget(iterations=3))

    joined = " ".join(completer.prompts).lower()
    for leak in ("of 3", "3 rounds", "you have 3", "round 1 of"):
        assert leak not in joined


def test_older_findings_compact_and_recent_ones_do_not() -> None:
    """Compaction, asserted on both sides of the line.

    The recent ones keep their evidence so a later round can weigh them; the
    older ones keep only a cause, which is information the system paid for and
    then threw away. Both halves matter and only one of them is free.
    """
    attempts = tuple(
        Attempt(
            index=i,
            ok=True,
            detail="",
            finding=a_verdict(
                cause=f"cause number {i}",
                evidence_value=f"evidence-{i}",
                next_step=f"next-{i}",
            ),
        )
        for i in range(CARRY_VERBATIM + 2)
    )
    text = working(AgentState(incident=incident("INC-1044"), attempts=attempts))

    assert "cause number 0" in text
    assert "evidence-0" not in text
    assert "evidence-4" in text
    assert "next-4" in text


def test_a_discarded_finding_does_not_enter_the_working_set() -> None:
    """Chapter 8 moved this check to gate the context. Here it gates an unknown
    number of later rounds rather than two."""
    completer = RoundScript(
        script=[
            {
                "calls": [CHECKOUT_LOGS],
                "answer": a_verdict(
                    cause="Invented.", evidence_value="not in any tool result"
                ),
            },
            settles(),
        ]
    )
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=4))

    assert result.attempts[0].finding is None
    assert not result.attempts[0].ok
    assert "discarded" in result.attempts[0].detail
    assert "Invented." not in completer.prompts[1]


def test_calls_already_made_are_listed_and_results_are_not_re_sent() -> None:
    completer = RoundScript(script=[keeps_going(), settles()])
    pursue(incident("INC-1044"), completer, budget=Budget(iterations=4))

    second = completer.prompts[1]
    assert "already read" in second
    assert "search_logs(minutes=60" in second
    # The mesh line the tool returned in round one must not be pasted back in.
    assert "x509 certificate has expired" not in second


def test_the_no_quit_experiments_arms_actually_differ() -> None:
    """Same precedent as Chapter 7's round budget and Chapter 8's veto paragraph.

    An experiment whose two arms have quietly become the same experiment reports
    a null result with total confidence, so the difference between them is
    asserted rather than trusted. Here the arms are one boolean, and what it
    controls is the entire cost profile of the rung.
    """
    quits = {
        "calls": [CHECKOUT_LOGS],
        "answer": a_verdict(
            needs_human=True,
            unreachable=True,
            service="unknown",
            evidence_value="no healthy upstream for cluster=inventory",
            next_step="Nothing in telemetry can settle this.",
        ),
    }

    honored = pursue(
        incident("INC-1043"),
        RoundScript(script=[quits]),
        budget=Budget(iterations=6),
    )
    ignored = pursue(
        incident("INC-1043"),
        RoundScript(script=[quits]),
        budget=Budget(iterations=6),
        honor_unreachable=False,
    )

    assert honored.rounds == 1
    assert honored.gave_up is not None
    assert honored.stopped_by is None

    assert ignored.rounds == 6
    assert ignored.gave_up is None
    assert ignored.stopped_by == "iterations: 6 of 6"


def test_a_state_restored_from_a_checkpoint_is_the_type_it_was() -> None:
    """What a checkpointer actually costs, in one assertion.

    The state is immutable so that it can be snapshotted and resumed, and the
    serializer that does the snapshotting has no tuple type - so everything
    comes back a list. The symptom was a `TypeError` three rounds into a
    resumed run, in the only path a suite that never crashes does not reach.
    """
    restored = AgentState(
        incident=incident("INC-1044"),
        attempts=[Attempt(0, True, "", calls=[], finding=None)],
        calls=[],
        results=[],
        recalled=[],
    )

    assert isinstance(restored.attempts, tuple)
    assert isinstance(restored.calls, tuple)
    assert isinstance(restored.results, tuple)
    assert isinstance(restored.recalled, tuple)
    assert isinstance(restored.attempts[0].calls, tuple)


# ------------------------------------------------- the journal and recall

def test_a_journal_entry_survives_a_round_trip(tmp_path: Path) -> None:
    journal = Journal(path=tmp_path / "journal.jsonl")
    entry = Entry(
        run_id="r1",
        incident_id="INC-1044",
        service="checkout-api",
        report="Everything is on fire.",
        routed=True,
        cause="Expired mesh certificate.",
        evidence="search_logs -> x509 certificate has expired",
        rounds=3,
        stopped_by=None,
        tools=("search_logs(minutes=60)",),
    )
    journal.append(entry)

    assert journal.entries() == (entry,)
    assert json.loads((tmp_path / "journal.jsonl").read_text())["rounds"] == 3


def test_an_absent_journal_reads_as_empty_rather_than_raising(tmp_path: Path) -> None:
    assert Journal(path=tmp_path / "nothing.jsonl").entries() == ()


def test_recall_never_returns_the_incident_under_investigation(tmp_path: Path) -> None:
    """Chapter 5's leakage discipline, one layer up and easier to get wrong.

    A journal accumulates repeated runs of the same incident, so without this an
    agent is handed its own previous answer and reaches it in one round.
    """
    journal = Journal(path=tmp_path / "journal.jsonl")
    target = incident("INC-1044")
    for run_id, source in (("r1", "INC-1044"), ("r2", "INC-1041")):
        journal.append(
            Entry(
                run_id=run_id,
                incident_id=source,
                service="checkout-api",
                report=INCIDENTS[source].report,
                routed=True,
                cause="Checkout returned 500s across the board.",
                evidence=None,
                rounds=2,
                stopped_by=None,
                tools=(),
            )
        )

    remembered = recall(journal, target, floor=0.0)

    assert [entry.incident_id for entry in remembered] == ["INC-1041"]


def test_recall_is_read_once_and_appears_in_every_prompt(tmp_path: Path) -> None:
    journal = Journal(path=tmp_path / "journal.jsonl")
    journal.append(
        Entry(
            run_id="r1",
            incident_id="INC-1041",
            service="checkout-api",
            report=INCIDENTS["INC-1041"].report,
            routed=True,
            cause="Connection pool exhausted after a deploy.",
            evidence=None,
            rounds=2,
            stopped_by=None,
            tools=(),
        )
    )
    completer = RoundScript(script=[keeps_going(), settles()])
    result = pursue(
        incident("INC-1044"),
        completer,
        budget=Budget(iterations=4),
        journal=journal,
        # The floor is corpus-relative and this journal has one entry, so the
        # test names it rather than depending on where scripts/agent_memory.py
        # calibrated the shipped default.
        recall_floor=0.0,
    )

    assert [entry.incident_id for entry in result.recalled] == ["INC-1041"]
    for prompt in completer.prompts:
        assert "Connection pool exhausted after a deploy." in prompt


def test_recall_on_an_empty_journal_costs_nothing(tmp_path: Path) -> None:
    completer = RoundScript(script=[settles()])
    result = pursue(
        incident("INC-1044"),
        completer,
        budget=Budget(iterations=2),
        journal=Journal(path=tmp_path / "journal.jsonl"),
    )

    assert result.recalled == ()
    assert "earlier investigations" not in completer.prompts[0]


# --------------------------------------------------------- cost and the ledger

def test_the_ledger_counts_rounds_and_the_ceiling_counts_requests() -> None:
    """Chapter 8's trap, inherited exactly as its manifest note warned.

    One `invoke` is one measurement covering two requests, so the ledger's
    `model_calls` is half of what the Termination Test asks about.
    """
    completer = RoundScript(script=[keeps_going(), settles()])
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=4))

    assert result.ledger.model_calls == 2
    assert result.api_requests == 4
    assert result.api_requests == 2 * result.rounds


def test_rounds_are_billed_under_distinct_labels() -> None:
    """A loop repeats one node, so a trace keyed on stage NAME merges its rows.

    Chapter 8 joined on `observe`/`localize`/`confirm` and they were unique by
    construction. This asserts the join key had to change.
    """
    completer = RoundScript(script=[keeps_going(), keeps_going(), settles()])
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=4))

    labels = [
        m.label for m in result.ledger.measurements if m.label.startswith("agent.")
    ]
    round_labels = [label for label in labels if label[len("agent."):].isdigit()]
    assert round_labels == ["agent.0", "agent.1", "agent.2"]
    assert len(set(round_labels)) == len(round_labels)
    assert result.attempt_table().count("round ") == 3


def test_every_tool_call_is_billed() -> None:
    completer = RoundScript(script=[keeps_going(), settles()])
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=4))

    billed = {
        m.label for m in result.ledger.measurements if m.label.startswith("agent.toolu")
    }
    assert len(billed) == len(result.calls)


# ------------------------------------------- the receipt that earns Chapter 10

def test_a_read_only_agent_can_cause_nothing() -> None:
    """Chapter 7's assertion, unchanged, and still true for the default menu."""
    assert blast_radius(menu_for()) == ()


def test_one_agent_that_can_act_carries_write_authority_into_every_round() -> None:
    """The Failure Receipt, as an assertion rather than an anecdote.

    The menu is a property of the agent, not of the round. There is no argument
    to `pursue` that grants write authority late, because there is no round the
    schema distinguishes from any other - which is what Chapter 10 splits.
    """
    reachable = blast_radius(menu_for(allow_writes=True))

    assert reachable != ()
    assert any("rollback" in item for item in reachable)
    # And it is reachable from every round, because there is only one menu.
    #
    # AMENDED IN CHAPTER 15, and the amendment is the lesson. This asserted
    # `menu_for(allow_writes=True) == TOOLS`, which was true only while every
    # tool was either offered or a write. Chapter 15 adds a third state and the
    # assertion failed without the property it was defending having changed at
    # all. It now states the property: granting writes withholds nothing else.
    granted = menu_for(allow_writes=True)
    assert all(spec in granted for spec in menu_for())
    assert ROLLBACK_DEPLOY in granted
    assert len(granted) == len(menu_for()) + 1


def test_a_dry_run_records_the_write_and_performs_nothing() -> None:
    rollback = ("rollback_deploy", {
        "service": "checkout-api", "deploy_id": "d-2026-02-11",
    })
    completer = RoundScript(
        script=[
            {
                "calls": [rollback],
                "answer": a_verdict(
                    evidence_tool="rollback_deploy",
                    evidence_value="rollback of checkout-api to d-2026-02-11",
                ),
            }
        ]
    )
    box = DryRunToolbox(incident=incident("INC-1044"), allow_writes=True)
    result = pursue(
        incident("INC-1044"),
        completer,
        budget=Budget(iterations=2),
        allow_writes=True,
        box=box,
    )

    assert [call.name for call in result.proposed] == ["rollback_deploy"]
    assert box.proposed == list(result.proposed)


def test_a_dry_run_result_does_not_announce_itself() -> None:
    """The lie is the instrument, and it is asserted so nobody helpfully fixes it.

    A result saying DRY RUN measures an agent that knows nothing happened, which
    is not the agent the receipt is about.
    """
    box = DryRunToolbox(incident=incident("INC-1044"), allow_writes=True)
    result = box.execute(
        ToolCall("t1", "rollback_deploy", {
            "service": "checkout-api", "deploy_id": "d-1",
        })
    )

    assert not result.is_error
    assert "dry" not in result.content.lower()


# --------------------------------------------------------------- carried over

def test_the_severity_floor_still_applies() -> None:
    """Chapter 3's one-directional floor, unchanged for the seventh chapter."""
    completer = RoundScript(script=[settles(severity="SEV3")])
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=2))

    assert result.routed
    assert result.decision.severity == "SEV1"


def test_a_vendor_failure_stops_the_run() -> None:
    """Bounded loops against a degraded provider are how a bill becomes a story."""
    completer = RoundScript(
        script=[{"calls": [CHECKOUT_LOGS], "failed": "api error: OverloadedError"}]
    )
    result = pursue(incident("INC-1044"), completer, budget=Budget(iterations=8))

    assert result.rounds == 1
    assert result.stopped_by is None
    assert "OverloadedError" in (result.decision.decided_by or "")


def test_the_loop_implementation_is_swappable() -> None:
    completer = RoundScript(script=[keeps_going(), settles()])
    result = pursue(
        incident("INC-1044"),
        completer,
        budget=Budget(iterations=4),
        loop=WhileLoop(),
    )

    assert result.routed


def test_the_rung_is_registered() -> None:
    from escalation_ladder.rungs import load_all

    assert "Level 6: Autonomous Agent" in load_all()
    assert run.__wrapped__ is not None if hasattr(run, "__wrapped__") else True


@pytest.mark.skipif(not RECORDINGS.exists(), reason="recording not captured yet")
def test_the_published_run_replays() -> None:
    """The chapter's printed output, reproduced with no key and no spend."""
    from escalation_ladder.llm import RecordedCompleter

    recordings = json.loads(RECORDINGS.read_text(encoding="utf-8"))
    result = pursue(
        incident("INC-1044"),
        RecordedCompleter(recordings=recordings),
        budget=Budget(iterations=8),
    )

    assert result.routed
    # The published run took four rounds and settled on a mesh log line. Both
    # are asserted, because a replay that routed in one round would mean the
    # recording had drifted away from the run the chapter prints.
    assert result.rounds == 4
    assert "x509 certificate has expired" in (result.decision.decided_by or "")
