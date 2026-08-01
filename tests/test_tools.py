"""Level 4 - the tool-using LLM.

The behaviours this chapter argues for, asserted structurally:

- the model's menu is read-only by default, and that is a property of the type
  rather than of the prompt (the Blast Radius Schema);
- no tool takes a timestamp, so the window is not something a model can get
  wrong;
- a tool error names what was available, so a wrong argument is correctable
  inside the round it happened in;
- a reported value that is not in a tool result is refused, however plausible;
- one round means one round: a model that asks again is recorded, not obeyed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.llm import (
    Completion,
    ToolCall,
    ToolResult,
    ToolRun,
    RecordedCompleter,
    ToolSpec,
    Usage,
    _wire,
)
from escalation_ladder.tools import (
    ROLLBACK_DEPLOY,
    SEARCH_RUNBOOKS,
    TOOLS,
    WINDOWS,
    Finding,
    Investigation,
    Toolbox,
    blast_radius,
    investigate,
    menu_for,
    value_supported,
)

INCIDENTS = {i.incident_id: i for i in load_incidents()}


def incident(incident_id: str) -> Incident:
    return INCIDENTS[incident_id]


@dataclass
class ScriptedCompleter:
    """A `Completer` whose tool choices and answer are fixed by the test.

    Executes the tools for real, because they are deterministic fixtures - which
    is the same reason `RecordedCompleter.invoke` does. A fake that also faked
    the tool output would pass while the tool and the answer drifted apart.
    """

    calls: list[tuple[str, dict]] = field(default_factory=list)
    answer: Finding | None = None
    wants_more: bool = False
    usage: Usage = field(default_factory=lambda: Usage(100, 20))
    seen: list[ToolCall] = field(default_factory=list)

    def parse(self, *, system, user, schema, effort="low"):
        raise AssertionError("this rung does not use parse")

    def invoke(self, *, system, user, tools, execute, schema, effort="low"):
        made = tuple(
            ToolCall(f"toolu_{i}", name, args)
            for i, (name, args) in enumerate(self.calls)
        )
        if not made:
            return ToolRun(None, self.usage, failed="the model asked for no tools")
        results = tuple(execute(call) for call in made)
        self.seen = list(made)
        if self.wants_more:
            return ToolRun(
                None, self.usage, made, results,
                "one round of tools was not enough", wanted_more=True,
            )
        return ToolRun(self.answer, self.usage, made, results)


def a_finding(**overrides) -> Finding:
    base = dict(
        service="search-api",
        severity="SEV3",
        cause="Reindex saturated disk IO and queries timed out.",
        evidence_tool="query_metric",
        evidence_value="0.1603",
        next_step="Throttle the reindex.",
        needs_human=False,
    )
    base.update(overrides)
    return Finding(**base)


# --------------------------------------------------------------------------
# the Blast Radius Schema
# --------------------------------------------------------------------------

def test_the_default_menu_contains_no_write_tools():
    """The load-bearing assertion of the whole chapter.

    Mutation-tested: flipping `menu_for`'s default to True turns this red.
    """
    assert all(spec.consequence == "read" for spec in menu_for())
    assert ROLLBACK_DEPLOY not in menu_for()


def test_the_write_tool_exists_and_is_only_reachable_on_purpose():
    assert ROLLBACK_DEPLOY in TOOLS
    assert ROLLBACK_DEPLOY in menu_for(allow_writes=True)


def test_blast_radius_of_the_shipped_menu_is_empty():
    # The number a design review should be given. It is computed from the menu,
    # not from a claim about the prompt.
    assert blast_radius(menu_for()) == ()
    assert len(blast_radius(menu_for(allow_writes=True))) == 1


# Chapter 15's addition. Five assertions, and the third is the one that made
# the change safe to make at all.
def test_the_runbook_tool_is_defined_and_not_offered():
    assert SEARCH_RUNBOOKS in TOOLS
    assert SEARCH_RUNBOOKS not in menu_for()
    assert SEARCH_RUNBOOKS in menu_for(allow_runbooks=True)


def test_a_read_capability_moves_no_blast_radius():
    """The cheap kind of capability addition, and why it is the cheap kind.

    A new tool changes what the system can find. Only `consequence` changes
    what it can break, and this one does not touch it - so the number a design
    review is handed is identical before and after.
    """
    assert blast_radius(menu_for(allow_runbooks=True)) == ()
    assert len(blast_radius(menu_for(allow_writes=True, allow_runbooks=True))) == 1


def test_the_default_menu_is_unchanged_from_chapter_sevens():
    """Why every prior test and every recorded figure survived the addition.

    The default menu is part of the prompt. Holding it fixed is what lets a
    capability be added in one commit and evaluated in another, instead of
    invalidating every measurement in the book on the way past.
    """
    assert [spec.name for spec in menu_for()] == [
        "query_metric",
        "recent_deploys",
        "search_logs",
    ]


def test_the_runbook_tool_refuses_below_the_retrieval_floor():
    """Chapter 5's empty tuple, arriving in a tool result unchanged."""
    box = Toolbox(incident=incident("INC-1046"))
    result = box.execute(
        ToolCall("x", "search_runbooks", {"query": "quarterly revenue forecast"})
    )
    assert result.is_error
    assert "above the floor" in result.content


def test_the_runbook_tool_returns_citable_passages():
    """The query INC-1046 needed and no Level 4 menu could express.

    Note what it takes to clear the floor: the words a report contains, not the
    word a human would type. That is Chapter 5's lexical result reaching Level
    4 unchanged, and it is why this tool is a hypothesis rather than a fix.
    """
    box = Toolbox(incident=incident("INC-1046"))
    result = box.execute(
        ToolCall(
            "x",
            "search_runbooks",
            {"query": "search-api reindex job saturating disk io"},
        )
    )
    assert not result.is_error
    assert "search-api-reindex" in result.content


def test_the_corpus_a_tool_reads_excludes_the_incident_it_is_reading_for():
    """Chapter 5's `exclude_incident`, which a new caller could have skipped.

    A tool built on `default_index` inherits the exclusion. One built on
    `build_corpus()` with no argument would retrieve the incident's own root
    cause and score perfectly while measuring nothing. The same query proves
    both halves: INC-1044's own passage is retrievable for a different
    incident and unreachable for itself.
    """
    query = {"query": "service mesh certificate expiry"}
    other = Toolbox(incident=incident("INC-1046")).execute(
        ToolCall("x", "search_runbooks", query)
    )
    itself = Toolbox(incident=incident("INC-1044")).execute(
        ToolCall("x", "search_runbooks", query)
    )

    assert "INC-1044" in other.content
    assert "INC-1044" not in itself.content


def test_every_tool_declares_a_consequence():
    # `consequence` has no default, so this cannot silently regress - but a
    # future tool could still be declared "read" while writing. The test that
    # catches THAT is a reviewer, and this one makes the field visible in a diff.
    assert all(spec.consequence in {"read", "write"} for spec in TOOLS)


def test_the_toolbox_refuses_a_write_even_if_one_reaches_it():
    box = Toolbox(incident=incident("INC-1046"))
    result = box.execute(ToolCall("x", "rollback_deploy", {"service": "search-api"}))
    assert result.is_error
    assert "not permitted" in result.content


def test_no_tool_schema_accepts_a_timestamp():
    """The model chooses what to read and never when."""
    for spec in TOOLS:
        fields = set(spec.parameters["properties"])
        assert not fields & {"at", "until", "ending_at", "start", "end", "since"}


def test_every_tool_schema_is_strict_shaped():
    for spec in TOOLS:
        wire = _wire(spec)
        assert wire["strict"] is True
        assert wire["input_schema"]["additionalProperties"] is False
        assert set(wire["input_schema"]["required"]) == set(
            wire["input_schema"]["properties"]
        )
    assert "consequence" not in _wire(TOOLS[0])


def test_the_window_is_an_enum_not_a_free_integer():
    schema = dict(TOOLS[0].parameters["properties"])["minutes"]
    assert schema["enum"] == list(WINDOWS)


# --------------------------------------------------------------------------
# legible errors
# --------------------------------------------------------------------------

def test_an_unknown_metric_error_names_the_metrics_that_exist():
    box = Toolbox(incident=incident("INC-1046"))
    result = box.execute(
        ToolCall("x", "query_metric",
                 {"service": "search-api", "metric": "messages_sent",
                  "minutes": 60})
    )
    assert result.is_error
    assert "http_5xx_rate" in result.content
    assert "p99_latency_ms" in result.content


def test_an_unknown_log_source_error_names_the_sources_that_exist():
    box = Toolbox(incident=incident("INC-1046"))
    result = box.execute(
        ToolCall("x", "search_logs",
                 {"source": "kafka", "pattern": "lag", "minutes": 60})
    )
    assert result.is_error
    assert "service-mesh" in result.content


def test_a_log_search_with_no_matches_is_not_an_error():
    # An empty result is an answer. Reporting it as a failure would push a model
    # into treating "nothing is wrong here" as something to work around.
    box = Toolbox(incident=incident("INC-1046"))
    result = box.execute(
        ToolCall("x", "search_logs",
                 {"source": "payment-gateway", "pattern": "kubernetes",
                  "minutes": 60})
    )
    assert not result.is_error
    assert "no lines matching" in result.content


# --------------------------------------------------------------------------
# the window is bound to the incident
# --------------------------------------------------------------------------

def test_the_window_ends_at_the_incident_not_at_wall_clock_now():
    assert Toolbox(incident=incident("INC-1044")).until == "2026-02-11T22:37:00Z"


def test_a_live_incident_window_is_clamped_to_now():
    assert Toolbox(incident=incident("INC-1046")).until == "2026-03-21T05:00:00Z"


def test_the_live_incident_reads_as_degraded():
    box = Toolbox(incident=incident("INC-1046"))
    result = box.execute(
        ToolCall("x", "query_metric",
                 {"service": "search-api", "metric": "http_5xx_rate",
                  "minutes": 60})
    )
    assert not result.is_error
    assert "max 0.18" in result.content


# --------------------------------------------------------------------------
# the citation check
# --------------------------------------------------------------------------

def test_a_value_copied_from_the_cited_tool_is_supported():
    box = Toolbox(incident=incident("INC-1046"))
    call = ToolCall("c1", "query_metric",
                    {"service": "search-api", "metric": "http_5xx_rate",
                     "minutes": 60})
    result = box.execute(call)
    value = result.content.split("max ")[1].split()[0]
    assert value_supported(a_finding(evidence_value=value), (call,), (result,))


def test_a_rounded_value_is_not_supported():
    """Catches arithmetic rather than invention, which is new at this rung."""
    box = Toolbox(incident=incident("INC-1046"))
    call = ToolCall("c1", "query_metric",
                    {"service": "search-api", "metric": "http_5xx_rate",
                     "minutes": 60})
    result = box.execute(call)
    assert not value_supported(
        a_finding(evidence_value="roughly 18 percent"), (call,), (result,)
    )


def test_a_value_from_a_different_tool_is_not_supported():
    box = Toolbox(incident=incident("INC-1046"))
    metric_call = ToolCall("c1", "query_metric",
                           {"service": "search-api", "metric": "http_5xx_rate",
                            "minutes": 60})
    deploy_call = ToolCall("c2", "recent_deploys", {"service": "search-api"})
    metric_result = box.execute(metric_call)
    deploy_result = box.execute(deploy_call)
    finding = a_finding(evidence_tool="recent_deploys", evidence_value="max 0.18")
    assert not value_supported(
        finding, (metric_call, deploy_call), (metric_result, deploy_result)
    )


def test_an_empty_value_is_not_supported():
    assert not value_supported(a_finding(evidence_value="   "), (), ())


# --------------------------------------------------------------------------
# investigate
# --------------------------------------------------------------------------

def test_a_grounded_finding_routes():
    box = Toolbox(incident=incident("INC-1046"))
    call = ToolCall("toolu_0", "query_metric",
                    {"service": "search-api", "metric": "http_5xx_rate",
                     "minutes": 60})
    value = box.execute(call).content.split("max ")[1].split()[0]

    completer = ScriptedCompleter(
        calls=[("query_metric",
                {"service": "search-api", "metric": "http_5xx_rate",
                 "minutes": 60})],
        answer=a_finding(evidence_value=value),
    )
    result = investigate(incident("INC-1046"), completer)
    assert result.routed
    assert result.decision.rota == "search-oncall"
    assert result.cause is not None
    assert value in result.decision.decided_by


def test_an_ungrounded_finding_is_refused():
    completer = ScriptedCompleter(
        calls=[("recent_deploys", {"service": "search-api"})],
        answer=a_finding(evidence_tool="query_metric", evidence_value="0.9999"),
    )
    result = investigate(incident("INC-1046"), completer)
    assert not result.routed
    assert "evidence not found" in result.decision.decided_by


def test_asking_for_no_tools_is_refused_rather_than_answered():
    """A finding with no tool result behind it is a Level 1 answer at Level 4 cost."""
    result = investigate(incident("INC-1046"), ScriptedCompleter(calls=[]))
    assert not result.routed
    assert "no tools" in result.decision.decided_by


def test_wanting_a_second_round_refuses_and_records_it():
    completer = ScriptedCompleter(
        calls=[("search_logs",
                {"source": "checkout-api", "pattern": "error", "minutes": 60})],
        wants_more=True,
    )
    result = investigate(incident("INC-1044"), completer)
    assert not result.routed
    assert result.wanted_more
    assert "not enough" in result.decision.decided_by


def test_the_severity_floor_still_applies_five_rungs_on():
    box = Toolbox(incident=incident("INC-1046"))
    call = ToolCall("toolu_0", "recent_deploys", {"service": "checkout-api"})
    value = box.execute(call).content.splitlines()[1].split()[0]
    completer = ScriptedCompleter(
        calls=[("recent_deploys", {"service": "checkout-api"})],
        answer=a_finding(
            service="checkout-api", severity="SEV3",
            evidence_tool="recent_deploys", evidence_value=value,
        ),
    )
    result = investigate(incident("INC-1046"), completer)
    # checkout-api is a tier SEV1 service; a model reading SEV3 cannot lower it.
    assert result.decision.severity == "SEV1"


def test_every_tool_call_is_measured_separately():
    completer = ScriptedCompleter(
        calls=[
            ("query_metric", {"service": "search-api",
                              "metric": "http_5xx_rate", "minutes": 60}),
            ("recent_deploys", {"service": "search-api"}),
        ],
        answer=a_finding(needs_human=True),
    )
    result = investigate(incident("INC-1046"), completer)
    labels = [m.label for m in result.ledger.measurements]
    assert "tools.model" in labels
    assert sum(1 for label in labels if label.startswith("tools.toolu_")) == 2


def test_the_model_call_is_the_only_billed_tokens():
    completer = ScriptedCompleter(
        calls=[("recent_deploys", {"service": "search-api"})],
        answer=a_finding(needs_human=True),
    )
    result = investigate(incident("INC-1046"), completer)
    assert result.ledger.model_calls == 1
    assert result.ledger.total_input_tokens == 100


def test_run_returns_the_ledger_from_the_shipped_path():
    """`run` must not re-walk the rung; it measures what `investigate` runs."""
    from escalation_ladder.tools import run

    completer = ScriptedCompleter(
        calls=[("recent_deploys", {"service": "search-api"})],
        answer=a_finding(needs_human=True),
    )
    ledger = run(incident("INC-1046"), completer)
    assert [m.label for m in ledger.measurements][-1] == "tools.model"


def test_the_rung_is_registered_under_its_level():
    from escalation_ladder.rungs import load_all

    assert "Level 4: Tool-Using LLM" in load_all()


def test_the_round_budget_experiment_still_derives_from_the_shipped_prompt():
    """Guards scripts/round_budget.py against prompt drift.

    Its withheld arm is built by cutting the round-budget paragraph out of the
    real SYSTEM. If that paragraph is ever reworded, the cut stops matching and
    the experiment would silently compare the shipped prompt against itself,
    reporting no difference and quietly retiring one of the chapter's claims.
    """
    from scripts.round_budget import WITHHELD

    from escalation_ladder import tools as rung

    assert WITHHELD != rung.SYSTEM
    assert "one round" not in WITHHELD
    assert "several at once" in WITHHELD


# --------------------------------------------------------------------------
# replaying the chapter's published run, with no key and no spend
#
# The recordings hold one round per incident - `prompt_key` is the same for
# every sample of an incident, so the last one wins. That makes the replay the
# COMMON case rather than the best one: INC-1044 routed on 2 runs of 15 and the
# chapter prints both outcomes, while this replays the refusal.
# --------------------------------------------------------------------------

RECORDINGS = Path(__file__).parent / "recordings" / "ch07_tools.json"

needs_recordings = pytest.mark.skipif(
    not RECORDINGS.exists(), reason="recorded run not present"
)


def replay() -> RecordedCompleter:
    return RecordedCompleter(recordings=json.loads(RECORDINGS.read_text("utf-8")))


@needs_recordings
def test_the_worked_example_replays_exactly_as_the_chapter_prints_it():
    result = investigate(incident("INC-1046"), replay())
    assert result.routed
    assert result.decision.rota == "search-oncall"
    assert result.decision.severity == "SEV2"
    # The cited value is copied out of a tool result, so this also re-runs the
    # tools: a fixture change that moved the log line breaks it here.
    assert "query timeout after 3000ms during reindex" in result.decision.decided_by


@needs_recordings
def test_the_receipt_incident_replays_as_a_refusal():
    """INC-1044, the 13-in-15 case the Failure Receipt is built on."""
    result = investigate(incident("INC-1044"), replay())
    assert not result.routed
    assert not result.wanted_more
    assert result.decision.decided_by == "the telemetry does not explain this incident"


@needs_recordings
def test_the_untooled_incident_refuses_with_correct_tool_choices():
    """INC-1043: nothing in the menu can segment latency by tenant.

    The refusal is the right answer and the investigation was correct, which is
    the chapter's argument for scoring tool choice separately from the answer.
    """
    result = investigate(incident("INC-1043"), replay())
    assert not result.routed
    assert {c.name for c in result.calls} <= {
        "query_metric", "recent_deploys", "search_logs"
    }
    assert all(c.arguments.get("service", "search-api") == "search-api"
               for c in result.calls if c.name != "search_logs")
