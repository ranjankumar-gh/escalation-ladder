"""Level 5 - multi-step reasoning.

The behaviours this chapter argues for, asserted structurally:

- the maximum number of model calls is derived from `STEPS` and nothing else,
  which is the Termination Test answered in code rather than in prose;
- every step is shown the ORIGINAL report, and earlier findings are labelled as
  claims - Chapter 6's Downstream Veto, carried into a chain;
- tool RESULTS are never re-sent, which is the difference between a linear and a
  quadratic Context Ratchet;
- a finding whose cited value is not in any tool result never enters the carried
  state, so the citation check gates the context rather than only the answer;
- `wanted_more` - Chapter 7's Failure Receipt - is the ordinary case here, and a
  step that treated it as an error would refuse the incidents this rung exists
  to solve;
- a settled finding stops the walk, so a shorter chain costs less and never
  costs more.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.llm import (
    RecordedCompleter,
    ToolCall,
    ToolRun,
    Usage,
)
from escalation_ladder.reasoning import (
    MAX_MODEL_CALLS,
    STEPS,
    ChainState,
    Reasoning,
    Step,
    StepRecord,
    carried,
    reason,
    run,
)
from escalation_ladder.tools import Finding

INCIDENTS = {i.incident_id: i for i in load_incidents()}
RECORDINGS = Path(__file__).parent / "recordings" / "ch08_reasoning.json"


def incident(incident_id: str) -> Incident:
    return INCIDENTS[incident_id]


def a_finding(**overrides) -> Finding:
    base = dict(
        service="checkout-api",
        severity="SEV1",
        cause="The mesh certificate expired.",
        evidence_tool="search_logs",
        evidence_value="x509 certificate has expired",
        next_step="Renew the certificate.",
        needs_human=False,
    )
    base.update(overrides)
    return Finding(**base)


@dataclass
class StepScript:
    """A `Completer` whose behaviour is scripted per step of the chain.

    One entry per step: the tools that step calls, and what it answers. Tools run
    for real against the deterministic fixtures, for the reason Chapter 7's fake
    did - a fake that also faked the tool output would stay green while the tool
    and the answer drifted apart, and this rung's whole subject is one step
    reading another step's output.
    """

    script: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(100, 20))
    prompts: list[str] = field(default_factory=list)
    step: int = 0

    def parse(self, *, system, user, schema, effort="low"):
        raise AssertionError("this rung does not use parse")

    def invoke(self, *, system, user, tools, execute, schema, effort="low"):
        self.prompts.append(user)
        if self.step >= len(self.script):
            raise AssertionError(
                f"step {self.step + 1} ran, but the script has "
                f"{len(self.script)} entries - the chain ran too long"
            )
        entry = self.script[self.step]
        self.step += 1
        made = tuple(
            # The step index is in the call id on purpose. Ids that restarted at
            # zero every step would collide across the chain, and `value_supported`
            # joins findings to results on exactly this field.
            ToolCall(f"toolu_{self.step}_{i}", name, args)
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


CHECKOUT_LOGS = ("search_logs", {
    "source": "checkout-api", "pattern": "error", "minutes": 60,
})
MESH_LOGS = ("search_logs", {
    "source": "service-mesh", "pattern": "inventory", "minutes": 60,
})


# ---------------------------------------------------------------- the boundary

def test_the_maximum_call_count_is_derived_from_the_step_list() -> None:
    """The Termination Test, answered by arithmetic on a tuple.

    Asserted against the tuple rather than against the literal 6, so adding a
    step cannot leave a stale number in the book's own source. The literal is
    checked too, because the chapter prints it.
    """
    assert MAX_MODEL_CALLS == 2 * len(STEPS)
    assert (len(STEPS), MAX_MODEL_CALLS) == (3, 6)


def test_a_settled_finding_stops_the_chain_early() -> None:
    """Early exit shortens a walk and can never lengthen one.

    The script holds ONE entry. A chain that ran a second step would raise from
    `StepScript`, which is the assertion.
    """
    model = StepScript(script=[{"calls": [CHECKOUT_LOGS], "answer": a_finding(
        evidence_value="no healthy upstream for cluster=inventory",
    )}])
    result = reason(incident("INC-1044"), model)
    assert result.routed
    assert len(result.steps) == 1
    assert model.step == 1


def test_a_shorter_chain_is_the_same_code_not_a_capped_loop() -> None:
    """`steps=` slices the architecture; it does not cap an iteration count."""
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "answer": a_finding(needs_human=True,
                                                       next_step="Read the mesh.")},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:1])
    assert not result.routed
    assert result.exhausted
    assert [record.name for record in result.steps] == ["observe"]


# ------------------------------------------------------- the carried context

@pytest.mark.parametrize("index", range(len(STEPS)))
def test_every_step_sees_the_original_report(index: int) -> None:
    """The Downstream Veto's first prerequisite, at every position in the chain."""
    case = incident("INC-1044")
    state = ChainState(
        incident=case,
        records=(StepRecord("observe", True, "x", finding=a_finding()),),
    )
    message = carried(state, STEPS[index], index, len(STEPS))
    assert case.report in message


def test_earlier_findings_are_labelled_as_claims() -> None:
    """The second prerequisite. A step told nothing has no reason to disagree."""
    state = ChainState(
        incident=incident("INC-1044"),
        records=(
            StepRecord(
                "observe", True, "x",
                finding=a_finding(cause="Payments is timing out."),
            ),
        ),
    )
    message = carried(state, STEPS[1], 1, len(STEPS))
    assert "these may be wrong" in message
    assert "Payments is timing out." in message
    assert "step observe:" in message


def test_a_discarded_step_does_not_borrow_another_steps_name() -> None:
    """Why the finding lives on the record rather than in a parallel tuple.

    Step one is discarded, step two survives. Pairing two lists by position would
    caption step two's claim with step one's name and hand the next step a
    confident, correctly-formatted lie about which step read what.
    """
    state = ChainState(
        incident=incident("INC-1044"),
        records=(
            StepRecord("observe", False, "discarded: invented"),
            StepRecord(
                "localize", True, "x",
                finding=a_finding(cause="The mesh certificate expired."),
            ),
        ),
    )
    message = carried(state, STEPS[2], 2, len(STEPS))
    assert "step localize: The mesh certificate expired." in message
    assert "step observe: The mesh" not in message


def test_tool_results_are_never_re_sent() -> None:
    """The ratchet's slope.

    The calls already made are listed so the model does not repeat them; their
    RESULTS are not, because re-sending every result at every step is what turns
    linear growth into quadratic growth.
    """
    state = ChainState(
        incident=incident("INC-1044"),
        calls=(ToolCall("toolu_1_0", "search_logs",
                        {"source": "checkout-api", "pattern": "error",
                         "minutes": 60}),),
    )
    message = carried(state, STEPS[1], 1, len(STEPS))
    assert "search_logs(minutes=60, pattern='error', source='checkout-api')" in message
    assert "no healthy upstream" not in message


def test_the_step_number_is_told_to_the_model() -> None:
    """Chapter 7's position, carried up: telling a model its budget is telling
    the truth about the system rather than tuning it."""
    state = ChainState(incident=incident("INC-1044"))
    assert "step 1 of 3" in carried(state, STEPS[0], 0, 3)
    assert "step 3 of 3" in carried(state, STEPS[2], 2, 3)


# ------------------------------------------------------ what enters the state

def test_an_unsupported_value_never_enters_the_carried_state() -> None:
    """The citation check, gating the context rather than the answer.

    Step 1 invents a value. Step 2 must not be shown it as a claim, because a
    step that reads an invented value as prior fact spends its round confirming
    a fiction.
    """
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "answer": a_finding(
            needs_human=True,
            evidence_value="disk full on shard 7",
            next_step="Check the disks.",
        )},
        {"calls": [MESH_LOGS], "answer": a_finding(
            evidence_value="x509 certificate has expired",
        )},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:2])

    assert result.routed
    assert not result.steps[0].ok
    assert "discarded" in result.steps[0].detail
    assert "disk full on shard 7" not in model.prompts[1]


def test_a_supported_value_does_enter_the_carried_state() -> None:
    """The other half. A gate that rejects everything is not a gate."""
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "answer": a_finding(
            needs_human=True,
            cause="Checkout cannot reach an upstream called inventory.",
            evidence_value="no healthy upstream for cluster=inventory",
            next_step="Search the mesh for inventory.",
        )},
        {"calls": [MESH_LOGS], "answer": a_finding(
            evidence_value="x509 certificate has expired",
        )},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:2])

    assert result.steps[0].ok
    assert "Checkout cannot reach an upstream called inventory." in model.prompts[1]


def test_a_step_may_cite_an_earlier_steps_tool_result() -> None:
    """The capability the rung exists for, from the checking side.

    Step 2 quotes a line step 1 read. `value_supported` is given every result so
    far, so the citation resolves - and a chain that only checked the current
    step's results would refuse its own best answers.
    """
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "answer": a_finding(
            needs_human=True, evidence_value="no healthy upstream",
            next_step="Read the mesh.")},
        {"calls": [MESH_LOGS], "answer": a_finding(
            evidence_tool="search_logs",
            evidence_value="no healthy upstream for cluster=inventory")},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:2])
    assert result.routed


# --------------------------------------------------------- signals and faults

def test_wanting_more_tools_continues_the_chain() -> None:
    """Chapter 7's Failure Receipt is this rung's ordinary case.

    A step that copied Level 4's refusal branch would return `needs_human` here.
    The chain gives the model the round it asked for instead.
    """
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "wants_more": True},
        {"calls": [MESH_LOGS], "answer": a_finding()},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:2])

    assert result.routed
    assert result.steps[0].ok
    assert "the next step is it" in result.steps[0].detail


def test_a_vendor_failure_stops_the_walk() -> None:
    """Two more steps against a degraded provider buy the same refusal twice.

    The script holds two entries and only one may run, so a chain that carried on
    would raise from `StepScript`.
    """
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "failed": "api error: OverloadedError"},
        {"calls": [MESH_LOGS], "answer": a_finding()},
    ])
    result = reason(incident("INC-1044"), model)

    assert not result.routed
    assert not result.exhausted
    assert "OverloadedError" in result.decision.decided_by
    assert model.step == 1


def test_running_out_of_steps_records_what_it_still_wanted() -> None:
    """The Failure Receipt's shape: a chain that stopped with a specific,
    executable next check still on the table."""
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "answer": a_finding(
            needs_human=True,
            evidence_value="no healthy upstream for cluster=inventory",
            next_step="Search service-mesh for inventory.")},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:1])

    assert result.exhausted
    assert "Search service-mesh for inventory." in result.decision.decided_by


def test_a_discarded_finding_does_not_supply_the_refusal_reason() -> None:
    """The gate holds at the end of the chain too.

    A finding whose value was invented is not shown to the next step, and it does
    not get to name the next check on the way out either - a next step derived
    from a value nobody read is a plausible instruction with nothing behind it,
    handed to whoever gets paged.
    """
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "answer": a_finding(
            needs_human=True,
            evidence_value="disk full on shard 7",
            next_step="Replace the disk on shard 7.")},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:1])

    assert result.exhausted
    assert "shard 7" not in result.decision.decided_by
    assert result.decision.decided_by == "the telemetry does not explain this incident"


def test_the_severity_floor_still_applies() -> None:
    """Chapter 3's one-directional floor, unchanged for the sixth chapter."""
    model = StepScript(script=[{"calls": [CHECKOUT_LOGS], "answer": a_finding(
        severity="SEV3", evidence_value="no healthy upstream for cluster=inventory",
    )}])
    result = reason(incident("INC-1044"), model)
    assert result.decision.severity == "SEV1"


# ----------------------------------------------------------------- the ledger

def test_every_step_and_every_tool_call_is_billed() -> None:
    """The per-step split is the number this rung can report and Level 4 cannot."""
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "answer": a_finding(
            needs_human=True, evidence_value="no healthy upstream",
            next_step="Read the mesh.")},
        {"calls": [MESH_LOGS], "answer": a_finding(
            evidence_value="x509 certificate has expired")},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:2])

    labels = [m.label for m in result.ledger.measurements]
    assert "reasoning.observe" in labels
    assert "reasoning.localize" in labels
    assert result.ledger.model_calls == 2
    assert sum(1 for label in labels if label.startswith("reasoning.toolu_")) == 2


def test_the_ledger_counts_steps_and_the_ceiling_counts_requests() -> None:
    """Two units that look like one number, pinned apart.

    `MeteredCompleter` wraps the seam, and one `invoke` is one measurement
    covering two requests. So the ledger counts steps while the Termination Test
    asks about requests. Quoting the ledger as the answer would publish a
    maximum of three against a real ceiling of six.
    """
    model = StepScript(script=[
        {"calls": [CHECKOUT_LOGS], "answer": a_finding(
            needs_human=True, evidence_value="no healthy upstream",
            next_step="Read the mesh.")},
        {"calls": [MESH_LOGS], "answer": a_finding(
            evidence_value="x509 certificate has expired")},
    ])
    result = reason(incident("INC-1044"), model, steps=STEPS[:2])

    assert len(result.steps) == 2
    assert result.ledger.model_calls == 2
    assert result.api_requests == 4
    assert result.api_requests <= MAX_MODEL_CALLS


def test_an_unmeasured_arm_reports_not_measured_rather_than_zero() -> None:
    """Zero out of zero is not zero percent.

    The third time this book has hit the same defect. Chapter 4's gate found
    scripts/measure_costs.py averaging an all-failed rung to zeros and printing
    a Level 1 row identical to Level 0's genuinely free one. Chapter 7 discarded
    a 125-run measurement that had averaged 41 API failures into its cost row.
    Chapter 8's depth sweep then printed the Failure Receipt's conclusion -
    "routes at no depth in range" - from forty-five runs where every call had
    been refused for a dead credit balance.

    It keeps recurring because reporting honestly costs extra code every time
    and reporting dishonestly costs none. So it is pinned here.
    """
    from scripts.chain_depth import Cell
    from scripts.drift import Arm

    empty_cell = Cell(incident_id="INC-1044", depth=3, failed=3)
    assert empty_cell.report()["routed"] == "not measured"

    empty_arm = Arm(label="claims marked fallible", failed=3)
    assert empty_arm.report()["corrected"] == "not measured"

    # And the other direction: a real measurement still reports a rate.
    real = Cell(incident_id="INC-1044", depth=3, runs=3, routed=2)
    assert real.report()["routed"] == "2/3"


def test_the_drift_experiments_arms_actually_differ() -> None:
    """Guard the cut in scripts/drift.py against prompt drift.

    Chapter 7 established this precedent for scripts/round_budget.py. A derived
    arm built with `str.replace` degrades silently into a comparison of the
    shipped prompt with itself the moment the paragraph it cuts is reworded, and
    the experiment would then report no difference for the most reassuring
    possible wrong reason.
    """
    from scripts.drift import VETO_PARAGRAPH, WITHHELD

    from escalation_ladder.reasoning import SYSTEM

    assert VETO_PARAGRAPH in SYSTEM
    assert WITHHELD != SYSTEM
    # Each half asserted against the SHIPPED prompt too, so neither can pass by
    # being vacuously true of any string.
    assert "CLAIMS, not facts" in SYSTEM
    assert "CLAIMS, not facts" not in WITHHELD
    assert "Build on them." not in SYSTEM
    assert "Build on them." in WITHHELD
    assert len(WITHHELD) < len(SYSTEM)


def test_the_rung_is_registered() -> None:
    from escalation_ladder.rungs import load_all

    assert load_all()["Level 5: Multi-Step Reasoning"] is run


# --------------------------------------------------------------- the recording

@pytest.mark.skipif(not RECORDINGS.exists(), reason="recording not captured yet")
def test_the_published_run_replays() -> None:
    """The chapter's printed INC-1044 output, reproduced with no key and no spend."""
    recorded = json.loads(RECORDINGS.read_text(encoding="utf-8"))
    model = RecordedCompleter(recordings=recorded)
    result = reason(incident("INC-1044"), model)

    assert result.routed
    assert result.service == "checkout-api"
    assert result.decision.rota == "payments-oncall"
    assert result.decision.severity == "SEV1"
    assert len(result.steps) >= 2
