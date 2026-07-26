"""Chapter 6 - the Level 3 LLM workflow.

These tests pin the behaviours the chapter argues for. They split the same way
Chapters 4 and 5 do: a deterministic half needing no model, and a replay half
needing no key.

The deterministic half is larger again, for the reason the chapter gives - three
of the five stages are ordinary code, so three fifths of this rung is unit
testable in the Chapter 3 sense. That is the payoff being claimed, so it is the
thing the test file has to demonstrate rather than assert.

The load-bearing guards, each of which turns red under a specific mutation:

  * stage 2 ADDS the classification to the report rather than replacing it.
    Change `extract` to return `service + summary` and
    `test_replacing_the_report_loses_an_incident_that_augmenting_keeps` fails,
    because INC-1044's answer spans stop being retrieved.
  * blame attribution distinguishes a stage breaking from a stage refusing its
    input. Flip either `blames_input=` argument and one of the `test_origin_*`
    tests fails.
  * `origin_of` walks THROUGH deterministic stages and STOPS at model stages.
    Mark `extract` as a model stage in `STAGES` and the walk stops one hop early.
  * an empty retrieval degrades to the Level 1 answer instead of refusing, and
    makes no second call. `ExplodingCompleter` proves the second half.
  * the pipeline's ledger carries one measurement per stage that ran, which is
    what makes the per-stage cost split in the chapter regenerable.
"""

import json
from pathlib import Path

import pytest

from escalation_ladder.classify import SYSTEM as CLASSIFY_SYSTEM
from escalation_ladder.classify import Classification
from escalation_ladder.classify import build_user as classify_user
from escalation_ladder.classify import classify
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.fixtures.retrieval_labels import LABELS
from escalation_ladder.llm import (
    Completion,
    RecordedCompleter,
    Usage,
    prompt_key,
)
from escalation_ladder.retrieve import (
    FLOOR,
    LIMIT,
    Grounded,
    build_index,
    default_index,
    search,
)
from escalation_ladder.rules import SERVICES, route
from escalation_ladder.rungs import register_rung  # noqa: F401  (import side effect)
from escalation_ladder.workflow import (
    DRAFT_SYSTEM,
    STAGE_INPUT,
    STAGE_KIND,
    STAGES,
    MeteredCompleter,
    StageRecord,
    TracedAdvice,
    advise,
    build_user,
    extract,
    origin_of,
    run,
)

RECORDINGS = Path(__file__).parent / "recordings" / "ch06_workflow.json"
CH04_RECORDINGS = Path(__file__).parent / "recordings" / "ch04_classify.json"


def incident(incident_id: str):
    return next(i for i in load_incidents() if i.incident_id == incident_id)


needs_recordings = pytest.mark.skipif(
    not RECORDINGS.exists(), reason="Chapter 6 recordings not yet captured"
)


def recordings() -> dict[str, str]:
    """Chapter 6's recorded prompts. The rewritten-query block is not a prompt."""
    raw = json.loads(RECORDINGS.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def replay() -> RecordedCompleter:
    """Both model stages, replayed. Needs the Chapter 6 recordings."""
    return RecordedCompleter(recordings=recordings())


def replay_classify_only() -> RecordedCompleter:
    """Stage 1 only, from Chapter 4's recordings.

    Enough to drive the pipeline as far as retrieval, which is all the fallback
    tests need - and it is the same file Chapter 4 shipped, so these tests keep
    working even before this chapter's own run exists.
    """
    raw = json.loads(CH04_RECORDINGS.read_text(encoding="utf-8"))
    return RecordedCompleter(recordings=raw)


class ExplodingCompleter:
    """A completer that fails the test if anything asks it for a completion."""

    def parse(self, *, system, user, schema, effort="low"):
        raise AssertionError("a model call was made when none was expected")


class CountingCompleter:
    """Wraps a completer and counts how many calls were made through it."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def parse(self, *, system, user, schema, effort="low"):
        self.calls += 1
        return self.inner.parse(system=system, user=user, schema=schema, effort=effort)


class FailingCompleter:
    """Every call fails, the way a degraded vendor fails."""

    def __init__(self, reason: str = "api error: APIStatusError") -> None:
        self.reason = reason

    def parse(self, *, system, user, schema, effort="low"):
        return Completion(None, Usage(), self.reason)


def classified(incident_id: str) -> Classification:
    """Chapter 4's real recorded classification for one report."""
    raw = json.loads(CH04_RECORDINGS.read_text(encoding="utf-8"))
    claim = classify(incident(incident_id), RecordedCompleter(recordings=raw))
    assert not claim.needs_human, "the Chapter 4 recording should classify this"
    return claim


# --------------------------------------------------------------------------
# the sequence itself
# --------------------------------------------------------------------------


def test_the_sequence_is_knowable_before_the_run():
    """Chapter 2's Level 3 gate: the steps can be drawn. Level 5 cannot do this."""
    assert [name for name, _ in STAGES] == [
        "classify",
        "extract",
        "retrieve",
        "draft",
        "route",
    ]


def test_most_stages_are_not_model_calls():
    """The chapter's arithmetic depends on this ratio, so it is pinned."""
    kinds = [kind for _, kind in STAGES]
    assert kinds.count("model") == 2
    assert kinds.count("code") == 3


def test_every_stage_declares_where_its_input_comes_from():
    assert set(STAGE_INPUT) == set(STAGE_KIND)
    assert STAGE_INPUT["classify"] is None, "stage 1 reads the request itself"
    for name, _ in STAGES[1:]:
        assert STAGE_INPUT[name] in STAGE_KIND


# --------------------------------------------------------------------------
# stage 2 - the stage that answers Chapter 5's receipt, and has no model in it
# --------------------------------------------------------------------------


def test_extract_prepends_the_service_and_keeps_the_report_verbatim():
    claim = classified("INC-1045")
    query = extract(claim, incident("INC-1045"))
    assert query.startswith("notification-worker ")
    assert incident("INC-1045").report in query


def test_extract_is_fully_specified_by_its_inputs():
    """The Floor Test on stage 2: a unit test specifies the output exactly."""
    claim = classified("INC-1043")
    target = incident("INC-1043")
    assert extract(claim, target) == f"{claim.service} {target.report}"


def test_the_receipt_incident_now_retrieves_the_runbook_it_always_needed():
    """Chapter 5 refused INC-1045 here. This is the whole capability of the rung."""
    target = incident("INC-1045")
    assert search(default_index(target), target.report) == ()

    hits = search(default_index(target), extract(classified("INC-1045"), target))
    assert hits, "stage 2 should make the corpus reachable"
    assert "notification-worker-duplicates" in {h.passage.passage_id for h in hits}


def test_replacing_the_report_loses_an_incident_that_augmenting_keeps():
    """The measured reason stage 2 augments rather than replaces.

    Chapter 5's Failure Receipt demonstrated `service + summary`, which fixes
    INC-1045. Shipping it would have broken INC-1044, whose answer lives in a
    runbook the classifier's summary does not use the words for.
    """
    target = incident("INC-1044")
    spans = LABELS["INC-1044"].answers
    claim = classified("INC-1044")

    def reached(query: str) -> int:
        hits = search(default_index(target), query, limit=LIMIT, floor=FLOOR)
        context = " ".join(h.passage.text for h in hits)
        return sum(1 for span in spans if span in context)

    assert reached(f"{claim.service} {claim.summary}") == 0
    assert reached(extract(claim, target)) == len(spans)


# --------------------------------------------------------------------------
# the blame sink
# --------------------------------------------------------------------------


def trace_of(*names: str) -> tuple[StageRecord, ...]:
    return tuple(
        StageRecord(name, STAGE_KIND[name], True, "") for name in names
    )


def test_origin_is_the_reporting_stage_when_the_stage_itself_broke():
    trace = trace_of("classify", "extract", "retrieve", "draft")
    assert origin_of(trace, "draft", blames_input=False) == "draft"


def test_origin_walks_through_deterministic_stages_to_the_model_that_fed_them():
    """The drafter refusing its passages is a classification fault, one hop back."""
    trace = trace_of("classify", "extract", "retrieve", "draft")
    assert origin_of(trace, "draft", blames_input=True) == "classify"


def test_origin_of_an_empty_retrieval_is_the_classification_not_the_retriever():
    trace = trace_of("classify", "extract", "retrieve")
    assert origin_of(trace, "retrieve", blames_input=True) == "classify"


def test_the_first_stage_has_nobody_upstream_to_blame():
    trace = trace_of("classify")
    assert origin_of(trace, "classify", blames_input=True) == "classify"


def test_a_vendor_failure_is_reported_and_originated_by_the_same_stage():
    """The counter that would otherwise send a team to tune the wrong prompt."""
    target = incident("INC-1042")
    result = advise(target, FailingCompleter(), default_index(target))
    assert not result.routed
    assert result.reported_by == "classify"
    assert result.origin == "classify"


# --------------------------------------------------------------------------
# the fallback: degrade to the rung below rather than refuse
# --------------------------------------------------------------------------


def test_an_empty_retrieval_degrades_to_the_level_one_answer():
    """The classification was already paid for, so refusing here wastes it.

    An index over an empty corpus is the empty-retrieval case by construction -
    no floor to tune, no query to contrive.
    """
    target = incident("INC-1045")
    counting = CountingCompleter(replay_classify_only())
    result = advise(target, counting, build_index(()))

    assert result.routed, result.summary()
    assert result.degraded
    assert result.decision.rota == SERVICES["notification-worker"].rota
    assert counting.calls == 1, "the drafting call must not be made"


def test_the_degraded_answer_carries_no_citation():
    """A page with no runbook behind it must not look like a grounded one."""
    target = incident("INC-1045")
    result = advise(target, replay_classify_only(), build_index(()))

    assert result.degraded
    assert result.advice.citation is None
    assert result.advice.next_step is None
    assert not result.grounded
    assert "degraded" in result.summary()


def test_a_degraded_answer_names_the_stage_that_caused_it():
    """It routed, and it still has a cause worth counting separately.

    `retrieve` noticed; `classify` chose the query that found nothing. The
    degrade path is the one path with no downstream stage to overrule stage 1,
    so it is the one that most needs an origin recorded against it.
    """
    result = advise(incident("INC-1045"), replay_classify_only(), build_index(()))
    assert result.degraded and result.routed
    assert result.reported_by == "retrieve"
    assert result.origin == "classify"


def test_the_pipeline_refuses_when_even_the_fallback_cannot_route():
    """Degrading is not defaulting. A stage 1 refusal still leaves as a refusal."""
    target = incident("INC-1042")
    result = advise(target, FailingCompleter(), build_index(()))
    assert not result.routed
    assert not result.degraded
    assert result.decision.needs_human


def test_the_rung_registers_itself():
    from escalation_ladder.rungs import RUNGS, load_all

    load_all()
    assert "Level 3: LLM Workflow" in RUNGS


# --------------------------------------------------------------------------
# cost, measured inside the pipeline
# --------------------------------------------------------------------------


@needs_recordings
def test_the_ledger_records_one_measurement_per_stage_that_ran():
    target = incident("INC-1042")
    result = advise(target, replay(), default_index(target))
    labels = [m.label for m in result.ledger.measurements]
    assert labels, "the pipeline must measure itself"
    for label in labels:
        assert label.startswith("workflow.")
        assert label.removeprefix("workflow.") in STAGE_KIND


@needs_recordings
def test_deterministic_stages_record_zero_model_calls():
    """What makes the per-stage split honest rather than apportioned."""
    target = incident("INC-1042")
    result = advise(target, replay(), default_index(target))
    for measurement in result.ledger.measurements:
        stage = measurement.label.removeprefix("workflow.")
        if STAGE_KIND[stage] == "code":
            assert measurement.model_calls == 0
            assert measurement.input_tokens == 0


@needs_recordings
def test_the_metered_completer_bills_the_stage_it_is_set_to():
    from escalation_ladder.instrument import CostLedger

    ledger = CostLedger()
    metered = MeteredCompleter(inner=replay(), ledger=ledger, stage="draft")
    metered.parse(system="s", user="u", schema=Grounded)
    assert [m.label for m in ledger.measurements] == ["workflow.draft"]


@needs_recordings
def test_run_returns_the_ledger_from_the_shipped_path():
    """`run` must not re-walk the sequence; it measures what `advise` runs."""
    ledger = run(incident("INC-1042"), replay())
    assert [m.label for m in ledger.measurements][0] == "workflow.classify"


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------


def test_the_draft_prompt_tells_the_model_the_classification_may_be_wrong():
    """Without this the pipeline can only compound; it can never self-correct."""
    assert "IT MAY BE WRONG" in DRAFT_SYSTEM


def test_the_draft_user_turn_labels_the_claim_as_an_earlier_stage_not_as_fact():
    target = incident("INC-1042")
    claim = classified("INC-1042")
    hits = search(default_index(target), extract(claim, target))
    user = build_user(target, claim, hits)
    assert "An earlier stage classified this report as:" in user
    assert user.index("Retrieved passages:") < user.index("An earlier stage")


def test_the_draft_prompt_offers_every_service_the_routing_table_owns():
    for service in SERVICES:
        assert service in DRAFT_SYSTEM


def test_a_prompt_edit_invalidates_every_recording():
    """Chapter 4's property, still true two rungs up: evidence is prompt-keyed."""
    target = incident("INC-1042")
    claim = classified("INC-1042")
    hits = search(default_index(target), extract(claim, target))
    live = prompt_key(system=DRAFT_SYSTEM, user=build_user(target, claim, hits))
    edited = prompt_key(
        system=DRAFT_SYSTEM + "\n- Be concise.",
        user=build_user(target, claim, hits),
    )
    assert live != edited


# --------------------------------------------------------------------------
# the published run, replayed
# --------------------------------------------------------------------------


@needs_recordings
def test_level_zero_still_refuses_exactly_the_reports_this_rung_handles():
    refused = [i.incident_id for i in load_incidents() if not route(i).routed]
    assert refused == ["INC-1042", "INC-1043", "INC-1044", "INC-1045"]


@needs_recordings
def test_the_receipt_incident_routes_end_to_end():
    """INC-1045: refused by Level 0, refused by Level 2, routed here."""
    target = incident("INC-1045")
    result = advise(target, replay(), default_index(target))
    assert result.routed, result.summary()
    assert result.decision.rota == SERVICES["notification-worker"].rota
