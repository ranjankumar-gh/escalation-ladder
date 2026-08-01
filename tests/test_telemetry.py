"""What the record must not do.

Chapter 14's rules, asserted rather than described: a counter may only be
grouped by a bounded dimension, a cost ledger cannot express a failure, a loop
always records why it stopped, and the published Instrumentation Level is a
measurement that breaks when the record regresses.

The three seam levels are asserted directly, so a change that quietly drops a
field breaks a test rather than a chapter.
"""
from __future__ import annotations

import pytest

from escalation_ladder import telemetry as tel
from escalation_ladder.instrument import CostLedger, Measurement
from escalation_ladder.rules import route
from scripts.observe_rungs import composite_traces, ledger_traces, observe, sweep


@pytest.fixture(scope="module")
def swept() -> tuple[dict[int, list[tel.Trace]], dict[int, list[CostLedger]]]:
    return sweep()


def test_every_bounded_dimension_is_groupable() -> None:
    event = tel.Event("INC-1", 4, "tool.search_logs", "refused", "no_evidence")
    assert event.key(*tel.DIMENSIONS) == (4, "tool.search_logs", "refused",
                                          "no_evidence", 0)


def test_grouping_by_an_unbounded_field_is_refused() -> None:
    """The cardinality guard, and the only one that would cost real money.

    `reason` is free text produced per request. A counter grouped on it grows a
    label set with traffic, which is how a metrics bill overtakes a model bill.
    """
    event = tel.Event("INC-1", 4, "tool.search_logs", "refused", "no_evidence")
    with pytest.raises(ValueError, match="not a bounded dimension"):
        event.key("reason")
    with pytest.raises(ValueError, match="not a bounded dimension"):
        event.key("kind", "fields")


def test_kinds_are_closed_over_the_whole_corpus(swept: tuple) -> None:
    """No observed reason string escapes the declared set."""
    traces, _ = swept
    seen = {
        event.kind
        for batch in traces.values()
        for trace in batch
        for event in trace.events
    }
    assert seen <= set(tel.KINDS), seen - set(tel.KINDS)


def test_reasons_outnumber_kinds_by_an_order_of_magnitude(swept: tuple) -> None:
    """Chapter 14's cardinality figures, asserted so prose cannot drift."""
    traces, _ = swept
    events = [e for b in traces.values() for t in b for e in t.events]
    reasons = {e.reason for e in events if e.reason}
    kinds = {e.kind for e in events}
    assert len(reasons) >= 100
    assert len(kinds) <= 25
    assert len(reasons) > 5 * len(kinds)


def test_an_unclassified_reason_lands_in_narrative_not_in_a_new_kind() -> None:
    assert tel.kind_of("the acquirer's SDK started returning 502s after lunch") == (
        "narrative"
    )
    assert tel.kind_of("") == tel.UNKNOWN


def test_a_cost_ledger_cannot_express_a_failure() -> None:
    """The eleven-chapter seam, and why its Instrumentation Level is none.

    Not a straw man and not a criticism of `CostLedger`, which measures cost
    correctly. There is simply no field in a `Measurement` that can hold a
    refusal, so a record built from one reports every decision as `unknown`.
    """
    ledger = CostLedger([Measurement("classify", 900, 120, 41.0, 1)])
    trace = tel.from_ledger(ledger, incident_id="INC-1042", level=1)
    assert [e.outcome for e in trace.events] == ["answered"]
    assert all(e.kind in ("billed", tel.UNKNOWN) for e in trace.events)
    assert all(e.reason == "" for e in trace.events)


def test_level_zero_is_invisible_to_a_completer_wrapper() -> None:
    """The Zero Row cannot be seen by vendor observability, by construction."""
    trace = tel.witness(route(next_incident := _incident("INC-1042")), level=0)
    assert trace.events and trace.model_calls == 0
    assert trace.events[0].kind in tel.KINDS
    assert next_incident.incident_id == trace.incident_id


def test_a_loop_always_records_why_it_stopped(swept: tuple) -> None:
    """The regression that took the measured level from 5 to 7.

    The first adapter emitted a stop event only when a bound fired or the agent
    gave up - every path the loop takes deliberately, and none of the paths it
    takes because something underneath it broke. Deleting the unconditional arm
    reproduces an Instrumentation Level of 5.
    """
    traces, _ = swept
    for trace in traces[6]:
        rounds = trace.where(site="round").events
        stops = trace.where(site="stop").events
        assert not rounds or len(stops) == 1, trace.incident_id
        assert all(stop.kind in tel.KINDS for stop in stops)


def test_tool_arguments_survive_into_the_record(swept: tuple) -> None:
    """INC-1046's whole story is one argument, so the record must carry it."""
    traces, _ = swept
    trace = next(t for t in traces[4] if t.incident_id == "INC-1046")
    searches = [
        dict(event.fields)
        for event in trace.sites("tool.search_logs")
    ]
    assert {"source": "search-api", "pattern": "error", "minutes": "240"} in searches
    assert not any(argument.get("pattern") == "reindex" for argument in searches)


@pytest.mark.parametrize("level", [4, 5, 6])
def test_requests_outrun_model_calls_at_every_tool_using_rung(
    swept: tuple, level: int
) -> None:
    """Chapter 8's trap, as a counter. One metered call can be two requests.

    A rate limit is consumed at close to twice the rate the ledger reports at
    every rung that offers tools, and the ledger is not wrong - it is counting
    a different thing correctly.
    """
    traces, _ = swept
    requests = sum(e.requests for t in traces[level] for e in t.events)
    calls = sum(e.model_calls for t in traces[level] for e in t.events)
    assert calls > 0
    assert requests > calls
    assert 1.5 <= requests / calls <= 2.0


def test_every_owed_counter_computes(swept: tuple) -> None:
    """Twenty-two counters from ten chapters, none of them an instrument."""
    traces, _ = swept
    every = [t for batch in traces.values() for t in batch] + composite_traces()
    assert len(tel.OWED) == 22
    for owed in tel.OWED:
        tallied = owed.of(every)
        assert isinstance(tallied, dict), owed.name
        if owed.chapter != 6:  # no request degraded on this corpus
            assert tallied, owed.name


def test_the_last_rung_counter_needs_the_ladder() -> None:
    """The counter that was trivially true, and the reason it was.

    A cascade stops at the first rung that answers, so comparing `level_reached`
    against the last rung ATTEMPTED marks every answered request as a last-rung
    request. The ladder has to come from the configuration. Without it the
    counter reports `unknown` rather than a plausible wrong answer.
    """
    from escalation_ladder.router import LADDERS

    kinds = {
        (trace.ask, event.kind)
        for trace in composite_traces()
        for event in trace.where(site="composite").events
    }
    assert ("diagnose", "last_rung") not in kinds
    assert ("page", "last_rung") in kinds

    from escalation_ladder import evaluate as ev
    from escalation_ladder.router import handle

    blind = tel.witness(handle(_incident("INC-1041"), "page", ev.replay()), level=-1)
    assert [e.kind for e in blind.where(site="composite").events] == [tel.UNKNOWN]
    assert LADDERS["diagnose"][-1] == 6


def test_the_published_instrumentation_levels(swept: tuple) -> None:
    """Chapter 14's headline table, asserted.

    A change that drops a field from the record moves one of these, which is
    the point of publishing an integer rather than a claim about coverage.
    """
    traces, ledgers = swept
    assert tel.instrumentation_level(traces, traces) == 7
    assert tel.instrumentation_level(ledger_traces(ledgers), traces) is None
    assert (
        tel.instrumentation_level(ledger_traces(ledgers, billed_only=True), traces)
        is None
    )


def test_a_seam_is_graded_on_the_population_it_did_not_choose(swept: tuple) -> None:
    """The experimental design, guarded.

    Letting each seam nominate its own failing requests would let the poorest
    record report that it had nothing to explain. The population is fixed once
    from the richest record and applied to every seam.
    """
    traces, ledgers = swept
    poorest = ledger_traces(ledgers)
    for question in tel.QUESTIONS:
        assert not tel.exercised(poorest, question), question.level
        if tel.exercised(traces, question):
            assert not tel.localizes(poorest, question, tel.exercised(traces, question))


def test_no_rung_module_imports_telemetry() -> None:
    """The accretion constraint, mechanically.

    Chapter 14 added a record to eight rungs without editing any of them. If a
    rung ever imports this module the claim stops being true, and the chapter
    that says so is wrong rather than merely out of date.
    """
    import re
    from pathlib import Path

    # Matched on the import statement, not on the word: several rungs use
    # "telemetry" in prose, including the Level 4 refusal this chapter opens on.
    imports = re.compile(r"^\s*(from|import)\s+escalation_ladder\.telemetry", re.M)
    package = Path(tel.__file__).parent
    offenders = [
        path.name
        for path in sorted(package.glob("*.py"))
        if path.name != "telemetry.py" and imports.search(path.read_text())
    ]
    assert offenders == []


def _incident(incident_id: str):
    from escalation_ladder.fixtures.incidents import load_incidents

    return next(i for i in load_incidents() if i.incident_id == incident_id)
