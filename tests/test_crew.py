"""The behaviours this chapter argues for, asserted structurally.

Chapter 10 makes one claim that does not need an experiment and several that do.
This file is entirely about the first kind. Every test here is a property of the
tool menus, the handoffs, or the control flow - things that are true before any
model is called and stay true when a model behaves badly.

The authority boundary is the load-bearing one. `test_only_the_remediator_can_change_anything`
is the executable form of Chapter 9's receipt, and if it goes red the chapter's
central claim is false rather than merely unproven.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from escalation_ladder.agent import Verdict, pursue
from escalation_ladder.crew import (
    MENUS,
    ROLES,
    Handoff,
    Remediation,
    Resolution,
    Review,
    _settled_verdict,
    coordination_characters,
    menu_for_role,
    quote_supported,
    remediate_user,
    resolve,
    review_user,
)
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import Completion, ToolCall, ToolRun, ToolSpec, Usage
from escalation_ladder.rungs import RUNGS, load_all
from escalation_ladder.tools import TOOLS, blast_radius

RECORDINGS = Path(__file__).resolve().parent / "recordings"
INCIDENTS = {incident.incident_id: incident for incident in load_incidents()}


# --------------------------------------------------------------------------
# The authority boundary. This is the chapter.
# --------------------------------------------------------------------------


def test_only_the_remediator_can_change_anything() -> None:
    """Chapter 9's receipt, executable.

    Three roles, three menus, and exactly one of them has a non-empty blast
    radius. A single agent could not produce this table however it was prompted,
    because it had one menu.
    """
    radii = {role: blast_radius(menu_for_role(role)) for role in ROLES}
    assert radii["investigator"] == ()
    assert radii["reviewer"] == ()
    assert radii["remediator"] == (
        "rollback_deploy: roll a service back to a previous deploy",
    )


def test_the_remediator_cannot_read_anything() -> None:
    """The half of the boundary nobody builds, and the more important half.

    An actor that can gather its own evidence can talk itself into acting. This
    one cannot: every tool it holds is a write, so the only facts it has are the
    ones a different role handed it, and declining is always available.
    """
    assert all(spec.consequence == "write" for spec in menu_for_role("remediator"))
    assert not [spec for spec in menu_for_role("remediator") if spec.consequence == "read"]


def test_the_reviewer_holds_no_tools_at_all() -> None:
    """A reviewer that can read is a second investigator."""
    assert menu_for_role("reviewer") == ()


def test_the_menus_partition_the_tool_list() -> None:
    """Every tool lands in exactly one role, because the split is by consequence.

    Written as a partition rather than as three literal lists on purpose. A tool
    added in a later chapter is routed by what it DOES, so the failure mode where
    a new write tool is quietly offered to the investigator cannot happen by
    omission - it would have to be given a `consequence` of "read", which is a
    lie a reviewer can see in the diff.
    """
    assigned = [spec for role in ROLES for spec in menu_for_role(role)]
    assert sorted(spec.name for spec in assigned) == sorted(spec.name for spec in TOOLS)
    assert len(assigned) == len(set(spec.name for spec in assigned))


def test_a_new_write_tool_reaches_only_the_remediator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test for the partition, run rather than asserted in prose."""
    invented = ToolSpec(
        name="scale_service",
        description="Change a service's replica count.",
        parameters={"type": "object", "properties": {}, "required": []},
        consequence="write",
    )
    import escalation_ladder.crew as crew

    widened = TOOLS + (invented,)
    monkeypatch.setattr(crew, "TOOLS", widened)
    rebuilt = {
        "investigator": tuple(s for s in widened if s.consequence == "read"),
        "reviewer": (),
        "remediator": tuple(s for s in widened if s.consequence == "write"),
    }
    assert invented in rebuilt["remediator"]
    assert invented not in rebuilt["investigator"]
    assert len(blast_radius(rebuilt["remediator"])) == 2


# --------------------------------------------------------------------------
# The handoffs, and what they cost.
# --------------------------------------------------------------------------


@dataclass
class Scripted:
    """A completer that plays each role in turn. No key, no network, no spend."""

    endorse: bool = True
    quote: str | None = None
    rollback: bool = True
    invented_quote: bool = False
    parses: int = 0
    invokes: int = 0

    def parse(self, *, system, user, schema, effort="low"):  # type: ignore[no-untyped-def]
        self.parses += 1
        quoted = "a line that was never sent" if self.invented_quote else (
            self.quote if self.quote is not None else user.strip().splitlines()[0]
        )
        return Completion(
            parsed=schema(endorsed=self.endorse, reason="scripted", quoted=quoted),
            usage=Usage(1_200, 90),
            failed=None,
        )

    def invoke(self, *, system, user, tools, execute, schema, effort="low"):  # type: ignore[no-untyped-def]
        self.invokes += 1
        if schema is Remediation:
            proposed = ()
            if self.rollback:
                call = ToolCall(
                    "toolu_r", "rollback_deploy",
                    {"service": "checkout-api", "deploy_id": "dep-1"},
                )
                execute(call)
                proposed = (call,)
            return ToolRun(
                parsed=Remediation(
                    action="rollback" if self.rollback else "none",
                    deploy_id="dep-1" if self.rollback else "",
                    reason="scripted",
                ),
                usage=Usage(600, 60),
                calls=proposed,
                results=(),
            )
        call = ToolCall(
            "toolu_1", "search_logs",
            {"source": "service-mesh", "pattern": "inventory", "minutes": 60},
        )
        results = (execute(call),)
        return ToolRun(
            parsed=Verdict(
                service="checkout-api",
                severity="SEV1",
                cause="The mesh certificate expired.",
                evidence_tool="search_logs",
                evidence_value=results[0].content.split("\n")[0][:40],
                next_step="rotate the certificate",
                needs_human=False,
                unreachable=False,
            ),
            usage=Usage(5_200, 550),
            calls=(call,),
            results=results,
        )


def test_a_full_run_visits_three_roles_and_two_handoffs() -> None:
    outcome = resolve(INCIDENTS["INC-1044"], Scripted())
    assert outcome.routed
    assert outcome.roles_run == ("investigator", "reviewer", "remediator")
    assert [(h.sender, h.receiver) for h in outcome.handoffs] == [
        ("investigator", "reviewer"),
        ("reviewer", "remediator"),
    ]
    assert outcome.blocked_at is None


def test_the_reviewer_can_stop_the_page_and_the_remediator_never_runs() -> None:
    """A rejection is a refusal, not a downgrade, and it names the role."""
    outcome = resolve(INCIDENTS["INC-1044"], Scripted(endorse=False))
    assert not outcome.routed
    assert outcome.blocked_at == "reviewer"
    assert outcome.remediation is None
    assert outcome.proposed == ()
    assert outcome.decision.decided_by.startswith("reviewer: ")


def test_an_invented_quote_is_caught_before_the_page() -> None:
    """The fourth outing for the citation check, guarding a judgment this time.

    A role with no tools can only manufacture evidence, because manufacturing is
    the only way it can produce any. This is the check that notices.
    """
    outcome = resolve(INCIDENTS["INC-1044"], Scripted(invented_quote=True))
    assert not outcome.routed
    assert outcome.blocked_at == "reviewer"
    assert "not in what it was sent" in outcome.decision.decided_by


def test_a_refusing_investigator_produces_no_handoff() -> None:
    """No conclusion, no coordination cost. The cheap outcome is genuinely cheap."""

    @dataclass
    class Silent:
        def parse(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("the reviewer must never be reached")

        def invoke(self, *, system, user, tools, execute, schema, effort="low"):  # type: ignore[no-untyped-def]
            return ToolRun(parsed=None, usage=Usage(400, 10), failed="vendor said no")

    outcome = resolve(INCIDENTS["INC-1044"], Silent())
    assert outcome.blocked_at == "investigator"
    assert outcome.handoffs == ()
    assert coordination_characters(outcome) == 0


def test_api_requests_counts_requests_not_ledger_calls() -> None:
    """Chapter 8's trap, inherited twice.

    One `invoke` is one measurement covering two vendor requests. A crew that
    quoted its ledger would publish a number roughly half the truth, and this
    rung is asked about that number more than any other.
    """
    outcome = resolve(INCIDENTS["INC-1044"], Scripted())
    investigator = outcome.investigation.api_requests
    assert outcome.api_requests == investigator + 1 + 2
    assert outcome.api_requests > outcome.ledger.model_calls


def test_carrying_the_claim_alone_is_cheaper_and_shows_the_reviewer_less() -> None:
    """The knob that changes the rung's cost without changing its behaviour.

    Both arms are real designs. The cheap one hands the reviewer the
    investigator's conclusion and the investigator's own selection of evidence,
    which is a consistency check wearing a veto's clothes - so the saving is real
    and so is what it buys.
    """
    full = resolve(INCIDENTS["INC-1044"], Scripted(), carry="results")
    thin = resolve(INCIDENTS["INC-1044"], Scripted(), carry="claim")
    assert coordination_characters(thin) < coordination_characters(full)
    assert "<tool-output>" in full.handoffs[0].carried
    assert "<tool-output>" not in thin.handoffs[0].carried
    # Both still show the reviewer the ORIGINAL report. Chapter 8 measured that
    # this is the only half of the Downstream Veto that carries weight, so it is
    # the one thing a token optimization must not be allowed to remove.
    assert "<report>" in thin.handoffs[0].carried


def test_the_reviewer_always_sees_the_original_report() -> None:
    incident = INCIDENTS["INC-1044"]
    outcome = resolve(incident, Scripted())
    assert incident.report in outcome.handoffs[0].carried


# --------------------------------------------------------------------------
# The published numbers. These break before the prose goes stale.
# --------------------------------------------------------------------------


def _replayed_rows() -> dict[str, int]:
    """Coordination characters per incident, replayed from Chapter 9's recordings."""
    from escalation_ladder.llm import RecordedCompleter

    recordings = json.loads((RECORDINGS / "ch09_agent.json").read_text())
    completer = RecordedCompleter(recordings=recordings)
    from escalation_ladder.crew import REMEDIATE_SYSTEM, REVIEW_SYSTEM

    specimen = Review(endorsed=True, reason="scripted", quoted="placeholder")
    rows: dict[str, int] = {}
    for incident in load_incidents():
        investigation = pursue(incident, completer)
        verdict = _settled_verdict(investigation)
        if verdict is None:
            continue
        first = len(REVIEW_SYSTEM) + len(
            review_user(incident, verdict, investigation, carry="results")
        )
        second = len(REMEDIATE_SYSTEM) + len(remediate_user(verdict, specimen))
        rows[incident.incident_id] = first + second
    return rows


def test_the_replayed_corpus_reaches_four_handoffs() -> None:
    """Chapter 9's split, reproduced for free: INC-1043 gives up and never hands over."""
    rows = _replayed_rows()
    assert sorted(rows) == ["INC-1042", "INC-1044", "INC-1045", "INC-1046"]
    assert "INC-1043" not in rows


def test_the_published_coordination_figure_still_holds() -> None:
    """The chapter prints a mean of 6,305 characters. Break here, not in the prose."""
    rows = _replayed_rows()
    mean = sum(rows.values()) / len(rows)
    assert mean == pytest.approx(6_305, rel=0.02)


def test_the_crew_adds_three_requests_per_incident() -> None:
    """Exact, and the number the cost table's request column is built from."""
    rows = _replayed_rows()
    from escalation_ladder.llm import RecordedCompleter

    recordings = json.loads((RECORDINGS / "ch09_agent.json").read_text())
    completer = RecordedCompleter(recordings=recordings)
    agent_requests = []
    for incident_id in rows:
        investigation = pursue(INCIDENTS[incident_id], completer)
        agent_requests.append(investigation.api_requests)
    mean_agent = sum(agent_requests) / len(agent_requests)
    assert mean_agent == pytest.approx(5.50, rel=0.01)
    assert (mean_agent + 3) / mean_agent == pytest.approx(1.55, rel=0.02)


def test_chapter_nine_figures_used_by_the_break_even_test_are_the_published_ones() -> None:
    """`crew_cost.py` hard-codes Chapter 9's row. Pin it so a drift is loud."""
    import scripts.crew_cost as crew_cost

    assert crew_cost.AGENT_TOKENS == 20_158
    assert crew_cost.AGENT_ROUTED == 12
    assert crew_cost.AGENT_RUNS == 15


# --------------------------------------------------------------------------
# The rung registry.
# --------------------------------------------------------------------------


def test_the_top_rung_registers_itself() -> None:
    load_all()
    assert "Level 7: Multi-Agent System" in RUNGS


def test_the_ledger_is_the_same_unit_as_every_other_rung() -> None:
    """Eight architectures, one unit, one comparison. This is the last row."""
    outcome = resolve(INCIDENTS["INC-1044"], Scripted())
    assert outcome.ledger.model_calls >= 1
    assert outcome.ledger.total_input_tokens > 0
    labels = {m.label for m in outcome.ledger.measurements}
    assert any(label.startswith("crew.investigator") for label in labels)
    assert "crew.reviewer" in labels
    assert "crew.remediator" in labels


# --------------------------------------------------------------------------
# Measurement honesty. Fourth occurrence, now pinned.
# --------------------------------------------------------------------------


def test_an_unreplayable_role_raises_rather_than_billing_zero() -> None:
    """The crew must not hand back a ledger assembled from part of a crew.

    Chapter 4's gate found `measure_costs.py` averaging an all-failed rung to
    zeros. Chapter 7 discarded a 125-run measurement that averaged 41 API
    failures into its cost row. Chapter 8's first depth sweep printed a
    conclusion from runs where every call was refused. This is the fourth, and it
    is the subtlest: the crew replays its investigator perfectly and has no
    recordings for its other two roles, so the Level 7 row came out
    byte-identical to Level 6's.
    """
    from escalation_ladder.crew import run as crew_run
    from escalation_ladder.llm import RecordedCompleter

    recordings = json.loads((RECORDINGS / "ch09_agent.json").read_text())
    completer = RecordedCompleter(recordings=recordings)
    with pytest.raises(RuntimeError, match="reviewer never answered"):
        crew_run(INCIDENTS["INC-1044"], completer)


def test_a_rung_that_failed_anywhere_is_reported_unmeasured() -> None:
    """The table refuses to average the incidents that happened to be cheap.

    The runs that fail at this rung are the ones that got FURTHEST, so the
    survivors are the cheap incidents by construction. Averaging them publishes
    Level 7 as less expensive than Level 6.
    """
    import scripts.measure_costs as measure_costs

    empty = measure_costs.summarise_ledgers("Level 9: Nothing", [])
    assert empty["Model calls"] == "not measured"

    table = measure_costs.render_markdown_table(
        [{"Rung": "Level 7: X (unmeasured on 4 of 6)", **dict.fromkeys(
            measure_costs.COLUMNS, "not measured")}]
    )
    assert "not measured" in table


def test_a_reviewer_that_rejects_is_measured_and_a_reviewer_that_dies_is_not() -> None:
    """The distinction the raise depends on: a judgment is data, silence is not."""
    rejected = resolve(INCIDENTS["INC-1044"], Scripted(endorse=False))
    assert rejected.review is not None
    assert rejected.blocked_at == "reviewer"

    from escalation_ladder.crew import run as crew_run

    # A rejection is a real, cheap, measurable outcome, so `run` returns.
    ledger = crew_run(INCIDENTS["INC-1044"], Scripted(endorse=False))
    assert ledger.model_calls >= 1


def test_the_published_break_even_ratio_still_holds() -> None:
    """The chapter prints R = 1.189x and a headroom ceiling of 0.841.

    Both are derived from Chapter 9's recordings plus the committed price
    sidecar, with no key and no request, so this asserts the whole cost argument
    rather than a fixture detail. A corpus change breaks it here first.
    """
    import scripts.crew_cost as crew_cost

    added = crew_cost._derived("results")
    assert added is not None
    ratio = (crew_cost.AGENT_TOKENS + added) / crew_cost.AGENT_TOKENS
    assert ratio == pytest.approx(1.189, rel=0.01)
    assert 1.0 / ratio == pytest.approx(0.841, rel=0.01)

    accuracy = crew_cost.AGENT_ROUTED / crew_cost.AGENT_RUNS
    assert accuracy * ratio == pytest.approx(0.952, rel=0.01)


def test_the_cheap_handoff_is_cheaper_and_the_chapter_says_by_how_much() -> None:
    """Carrying the claim alone roughly halves the coordination cost."""
    import scripts.crew_cost as crew_cost

    full = crew_cost._derived("results")
    thin = crew_cost._derived("claim")
    assert full is not None and thin is not None
    assert full == pytest.approx(3_819, rel=0.02)
    assert thin == pytest.approx(1_583, rel=0.02)
    assert thin < full / 2


def test_the_calibration_is_measured_rather_than_the_folk_ratio() -> None:
    """Four characters per token is wrong here by more than a factor of two.

    Asserted because a later reader tempted to "simplify" the derivation back to
    a constant would silently change every number in the cost section.
    """
    import scripts.crew_cost as crew_cost
    from escalation_ladder.llm import RecordedCompleter

    recordings = json.loads((RECORDINGS / "ch09_agent.json").read_text())
    priced = json.loads((RECORDINGS / "usage.json").read_text())
    spy = crew_cost.Spy(RecordedCompleter(recordings=recordings))
    pursue(INCIDENTS["INC-1044"], spy)
    ratio = crew_cost.calibrate(spy.seen, priced)
    assert ratio is not None
    assert 1.4 < ratio < 1.9
    assert ratio < 4.0 / 2


def test_breaking_even_on_this_corpus_demands_a_perfect_score() -> None:
    """The chapter's sharpest number, and it comes from the granularity.

    A required accuracy of 0.952 over fifteen runs is 14.28 runs, and there is no
    such thing as 0.28 of a run. Fourteen is 0.933, which is short - so the crew
    must go fifteen for fifteen. Asserted because the prose says "perfect" and a
    corpus change could quietly make that false.
    """
    import math

    import scripts.crew_cost as crew_cost

    added = crew_cost._derived("results")
    assert added is not None
    ratio = (crew_cost.AGENT_TOKENS + added) / crew_cost.AGENT_TOKENS
    required = (crew_cost.AGENT_ROUTED / crew_cost.AGENT_RUNS) * ratio

    needed = math.ceil(required * crew_cost.AGENT_RUNS - 1e-9)
    assert needed == crew_cost.AGENT_RUNS == 15
    assert needed - crew_cost.AGENT_ROUTED == 3
    # Fourteen of fifteen is genuinely short, which is the whole point.
    assert 14 / 15 < required


def test_the_top_rung_spends_five_requests_on_an_incident_level_zero_answers() -> None:
    """Chapter 10's Failure Receipt, executable.

    INC-1046 is a machine-generated alert line. `rules.py` routes it correctly for
    zero tokens, and the crew reaches the same page through three roles and five
    vendor requests. That gap is the argument for Chapter 11's router, so it is
    asserted here rather than left as a sentence.
    """
    from escalation_ladder.llm import RecordedCompleter
    from escalation_ladder.rules import route

    floor = route(INCIDENTS["INC-1046"])
    assert floor.routed
    assert floor.rota == "search-oncall"

    recordings = json.loads((RECORDINGS / "ch09_agent.json").read_text())
    investigation = pursue(INCIDENTS["INC-1046"], RecordedCompleter(recordings=recordings))
    assert investigation.rounds == 1
    assert investigation.api_requests == 2
    # Plus one parse for the reviewer and two for the remediator's invoke.
    assert investigation.api_requests + 3 == 5
