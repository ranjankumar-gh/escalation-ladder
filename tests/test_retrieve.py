"""Chapter 5 - the Level 2 retrieval-augmented rung.

These tests pin the behaviours the chapter argues for, and they split the same
way Chapter 4's do: a deterministic half that needs no model, and a replay half
that needs no key.

The deterministic half is larger here than at any previous rung, and that is the
point rather than an accident. Retrieval is ordinary code with no model in it, so
retrieval quality is unit-testable in a way generation quality is not - which is
exactly why the chapter insists the two be measured separately.

The load-bearing guards, each of which turns red under a specific mutation:

  * the relevance floor rejects every query the corpus cannot answer. Drop the
    floor to zero and `test_the_floor_refuses_every_unanswerable_query` fails
    with six confabulations.
  * `context_recall` returns the exact numbers Chapter 5 publishes. Edit a
    runbook, a report, or the chunking, and the number moves before the prose
    silently goes stale.
  * a citation is only supported when the passage was retrieved AND contains the
    quote. Weaken either half and `test_citation_*` fails.
  * an empty retrieval makes no model call at all. `ExplodingCompleter` proves
    it rather than the test trusting a comment.
"""

import json
from pathlib import Path

import pytest

from escalation_ladder.classify import normalized
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.fixtures.retrieval_labels import LABELS, UNANSWERABLE
from escalation_ladder.fixtures.runbooks_index import runbook_paths
from escalation_ladder.llm import Completion, RecordedCompleter, Usage, prompt_key
from escalation_ladder.retrieve import (
    FLOOR,
    LIMIT,
    SYSTEM,
    Grounded,
    _at_least,
    advise,
    build_corpus,
    build_index,
    build_user,
    chunk_runbook,
    citation_supported,
    context_recall,
    default_index,
    run,
    search,
    terms,
)
from escalation_ladder.rules import SERVICES
from escalation_ladder.rungs import register_rung  # noqa: F401  (import side effect)


def incident(incident_id: str):
    return next(i for i in load_incidents() if i.incident_id == incident_id)


class ExplodingCompleter:
    """A completer that fails the test if anything asks it for a completion."""

    def parse(self, *, system, user, schema, effort="low"):
        raise AssertionError("a model call was made when none was expected")


# --------------------------------------------------------------------------
# the corpus
# --------------------------------------------------------------------------


def test_one_passage_per_runbook_not_one_per_section():
    corpus = build_corpus()
    runbooks = [p for p in corpus if not p.passage_id.startswith("incident:")]
    assert len(runbooks) == len(runbook_paths())


def test_every_runbook_passage_carries_its_title():
    for path in runbook_paths():
        passage = chunk_runbook(path.read_text(encoding="utf-8"), path.stem)
        assert passage.text.startswith("#")
        assert passage.source in passage.text


def test_the_live_incident_is_excluded_from_its_own_index():
    """The eval-leakage guard. Without it every answer is retrievable verbatim."""
    for case in load_incidents():
        ids = {p.passage_id for p in build_corpus(exclude_incident=case.incident_id)}
        assert f"incident:{case.incident_id}" not in ids
        # every OTHER incident is still there - the exclusion is surgical
        others = {
            f"incident:{i.incident_id}"
            for i in load_incidents()
            if i.incident_id != case.incident_id
        }
        assert others <= ids


def test_labels_are_verbatim_spans_of_the_corpus():
    """A golden set that has drifted from the corpus measures nothing."""
    corpus = normalized(
        " ".join(p.read_text(encoding="utf-8") for p in runbook_paths())
    )
    for incident_id, label in LABELS.items():
        for span in label.answers:
            assert normalized(span) in corpus, f"{incident_id}: {span!r}"


def test_labelled_services_agree_with_the_routing_table():
    for label in LABELS.values():
        assert label.service in SERVICES


# --------------------------------------------------------------------------
# the retriever
# --------------------------------------------------------------------------


def test_terms_keeps_metric_names_whole():
    """`connection_pool_in_use` is one term, not four. Splitting it would make
    the rarest and most diagnostic token in the corpus disappear."""
    assert "connection_pool_in_use" in terms("check connection_pool_in_use now")


def test_search_is_deterministic():
    index = default_index(incident("INC-1042"))
    first = search(index, incident("INC-1042").report)
    second = search(index, incident("INC-1042").report)
    assert [h.passage.passage_id for h in first] == [
        h.passage.passage_id for h in second
    ]
    assert [h.score for h in first] == [h.score for h in second]


def test_search_never_returns_more_than_the_limit():
    index = default_index(incident("INC-1044"))
    assert len(search(index, incident("INC-1044").report, limit=2)) <= 2


def test_the_floor_refuses_every_unanswerable_query():
    """The chapter's central claim about the floor, as an assertion.

    With `floor=0.0` every one of these returns three passages and the model is
    asked to triage a question about the coffee machine.
    """
    index = build_index(build_corpus())
    for query in UNANSWERABLE:
        assert search(index, query) == (), query
        assert search(index, query, floor=0.0) != ()


def test_hits_say_which_words_earned_them():
    index = default_index(incident("INC-1043"))
    hits = search(index, incident("INC-1043").report)
    assert hits
    assert "search" in hits[0].matched


# --------------------------------------------------------------------------
# retrieval quality, measured with no model in the loop
# --------------------------------------------------------------------------


def test_context_recall_matches_the_published_numbers():
    """Chapter 5 publishes these. If a fixture moves, this breaks first."""
    scores = context_recall()
    assert scores == {
        "INC-1041": 0.5,
        "INC-1042": 1.0,
        "INC-1043": 1.0,
        "INC-1044": 1.0,
        "INC-1045": 0.0,
        "INC-1046": 1.0,
    }
    assert sum(scores.values()) / len(scores) == 0.75


def test_the_receipt_incident_retrieves_nothing_at_all():
    """INC-1045 is the Failure Receipt. The refusal is the measured behaviour."""
    case = incident("INC-1045")
    assert search(default_index(case), case.report) == ()


def test_substituting_the_query_reaches_the_answer_that_was_always_there():
    """The Substitution Test, applied to the query rather than the documents.

    One stage earlier - a classification - and the same corpus, the same
    retriever, and the same floor find the passage. That gap is what the receipt
    hands to Chapter 6, and this test is the proof it is a query problem rather
    than a corpus problem.
    """
    case = incident("INC-1045")
    index = default_index(case)
    label = LABELS["INC-1045"]

    assert search(index, case.report) == ()

    hits = search(index, f"{label.service} {case.report}")
    assert hits
    assert hits[0].passage.passage_id == "notification-worker-duplicates"
    context = normalized(" ".join(h.passage.text for h in hits))
    assert all(normalized(span) in context for span in label.answers)


def test_chapter_fours_own_output_is_the_query_this_rung_needed():
    """The receipt, tied to real prior-chapter output rather than a hand-written
    query. This is Chapter 4's recorded classification of INC-1045, verbatim from
    tests/recordings/ch04_classify.json - a service and a restatement of the
    symptom. Feeding it to the same retriever, over the same corpus, at the same
    floor, reaches the answer the raw report cannot.
    """
    case = incident("INC-1045")
    index = default_index(case)
    from_level_one = (
        "notification-worker Confirmation emails are being delivered "
        "multiple times (3-7 duplicates) to customers."
    )

    assert search(index, case.report) == ()

    hits = search(index, from_level_one)
    assert hits
    context = normalized(" ".join(h.passage.text for h in hits))
    assert all(
        normalized(span) in context for span in LABELS["INC-1045"].answers
    )


# --------------------------------------------------------------------------
# the schema and the citation check
# --------------------------------------------------------------------------


def test_service_enum_agrees_with_the_routing_table():
    """Chapter 4's guard, restated. A service added to one and not the other is
    a silent `unknown` in production rather than a failing test."""
    allowed = set(Grounded.model_fields["service"].annotation.__args__)
    assert allowed == set(SERVICES) | {"unknown"}


def claim(**overrides) -> Grounded:
    base = dict(
        service="search-api",
        severity="SEV3",
        passage_id="search-latency",
        quote="Verify the composite index on `tenant_id, created_at` exists.",
        next_step="Verify the composite index exists.",
        needs_human=False,
    )
    base.update(overrides)
    return Grounded(**base)


def retrieved_for(incident_id: str):
    case = incident(incident_id)
    return search(default_index(case), case.report)


def test_citation_accepts_a_real_quote_from_a_retrieved_passage():
    assert citation_supported(claim(), retrieved_for("INC-1043"))


def test_citation_forgives_rewrapping_but_not_rewording():
    hits = retrieved_for("INC-1043")
    rewrapped = claim(
        quote="Verify the   composite index\non `tenant_id, created_at`  exists."
    )
    assert citation_supported(rewrapped, hits)
    reworded = claim(quote="Verify that the composite index exists.")
    assert not citation_supported(reworded, hits)


def test_citation_rejects_a_passage_that_was_never_retrieved():
    """The invented-source failure."""
    assert not citation_supported(
        claim(passage_id="notification-worker-duplicates"),
        retrieved_for("INC-1043"),
    )


def test_citation_rejects_an_empty_quote():
    assert not citation_supported(claim(quote="   "), retrieved_for("INC-1043"))


# --------------------------------------------------------------------------
# the rung
# --------------------------------------------------------------------------


def test_an_empty_retrieval_costs_nothing_and_calls_nothing():
    """No passages means no call. The refusal is free, and the test proves it
    rather than trusting the ordering of two lines."""
    result = advise(incident("INC-1045"), ExplodingCompleter())
    assert result.decision.needs_human
    assert result.decision.decided_by == "no passage cleared the relevance floor"
    assert result.citation is None


def test_an_empty_retrieval_records_a_search_and_no_model_call():
    ledger = run(incident("INC-1045"), ExplodingCompleter())
    assert ledger.model_calls == 0
    assert [m.label for m in ledger.measurements] == ["retrieve.search"]


def recorded(incident_id: str, **fields) -> RecordedCompleter:
    """Build a completer that replays one hand-written claim for one incident."""
    case = incident(incident_id)
    hits = search(default_index(case), case.report)
    key = prompt_key(system=SYSTEM, user=build_user(case, hits))
    return RecordedCompleter(
        recordings={key: claim(**fields).model_dump_json()},
        usage=Usage(input_tokens=1000, output_tokens=100),
    )


def test_a_supported_citation_becomes_a_page_carrying_its_source():
    case = incident("INC-1043")
    result = advise(case, recorded("INC-1043"))
    assert result.grounded
    assert result.decision.rota == "search-oncall"
    assert result.decision.severity == "SEV3"
    assert result.decision.escalate_after_minutes == 60
    assert result.citation == "search-latency"
    assert result.decision.decided_by == "grounded in search-latency"
    assert "composite index" in (result.next_step or "")


def test_an_unsupported_citation_refuses_rather_than_paging():
    result = advise(
        incident("INC-1043"),
        recorded("INC-1043", quote="Restart the search cluster immediately."),
    )
    assert result.decision.needs_human
    assert result.decision.decided_by.startswith("citation not supported")


def test_a_declared_refusal_is_honoured_before_the_citation_is_checked():
    """Same ordering argument as Chapter 4: an honest 'these passages do not
    apply' must not be counted as an invented citation."""
    result = advise(
        incident("INC-1043"),
        recorded("INC-1043", needs_human=True, quote="not in any passage"),
    )
    assert result.decision.needs_human
    assert (
        result.decision.decided_by
        == "the retrieved passages do not address this incident"
    )


def test_a_vendor_failure_degrades_to_chapter_threes_shape():
    class Failing:
        def parse(self, *, system, user, schema, effort="low"):
            return Completion(None, Usage(), "api error: APIConnectionError")

    result = advise(incident("INC-1043"), Failing())
    assert result.decision.needs_human
    assert result.decision.decided_by == "api error: APIConnectionError"
    assert result.next_step is None


def test_the_severity_floor_still_only_raises():
    assert _at_least("SEV3", "SEV1") == "SEV1"
    assert _at_least("SEV1", "SEV3") == "SEV1"
    assert _at_least("SEV2", "SEV2") == "SEV2"


def test_the_floor_raises_a_model_under_read_on_a_tier_one_service():
    result = advise(
        incident("INC-1044"),
        recorded(
            "INC-1044",
            service="checkout-api",
            severity="SEV3",
            passage_id="service-mesh-certificates",
            quote="Read the mesh sidecar logs for handshake failures.",
        ),
    )
    assert result.decision.severity == "SEV1"


def test_the_prompt_leads_with_passages_so_the_cache_prefix_is_stable():
    case = incident("INC-1042")
    user = build_user(case, search(default_index(case), case.report))
    assert user.index("<passage") < user.index("<report>")


def test_the_rung_registers_itself():
    from escalation_ladder.rungs import RUNGS

    assert "Level 2: Retrieval-Augmented Generation" in RUNGS


@pytest.mark.parametrize("incident_id", ["INC-1042", "INC-1043", "INC-1044"])
def test_every_measured_incident_retrieves_something(incident_id):
    case = incident(incident_id)
    assert search(default_index(case), case.report), incident_id


# --------------------------------------------------------------------------
# the measured run, replayed
# --------------------------------------------------------------------------

RECORDINGS = json.loads(
    (Path(__file__).parent / "recordings" / "ch05_retrieve.json").read_text(
        encoding="utf-8"
    )
)

# The measured means from the run Chapter 5 publishes, so a replayed ledger
# reports the same order of magnitude the cost table does.
MEASURED = RecordedCompleter(
    recordings=RECORDINGS, usage=Usage(input_tokens=1719, output_tokens=163)
)


@pytest.mark.parametrize(
    "incident_id, rota, severity, citation",
    [
        ("INC-1042", "payments-oncall", "SEV2", "payment-gateway-timeouts"),
        ("INC-1043", "search-oncall", "SEV3", "search-latency"),
        ("INC-1044", "payments-oncall", "SEV1", "service-mesh-certificates"),
    ],
)
def test_the_published_run_replays(incident_id, rota, severity, citation):
    """Chapter 5 prints these three pages. This is them, with no key and no spend."""
    result = advise(incident(incident_id), MEASURED)
    assert result.grounded
    assert result.decision.rota == rota
    assert result.decision.severity == severity
    assert result.citation == citation


def test_every_published_quote_is_really_in_its_cited_runbook():
    """The citation check, run against real model output rather than a fixture.

    This is the assertion the chapter's whole citation argument rests on, so it
    is made against what the model actually returned on the measured run.
    """
    for incident_id in ("INC-1042", "INC-1043", "INC-1044"):
        case = incident(incident_id)
        hits = search(default_index(case), case.report)
        result = advise(case, MEASURED)
        cited = next(
            h.passage for h in hits if h.passage.passage_id == result.citation
        )
        assert normalized(result.quote or "") in normalized(cited.text)


def test_a_prompt_edit_invalidates_every_recording():
    """Chapter 4's property, still holding. Recordings are keyed on the exact
    prompt, so editing SYSTEM must make them all miss rather than quietly pass."""
    case = incident("INC-1043")
    hits = search(default_index(case), case.report)
    live_key = prompt_key(system=SYSTEM, user=build_user(case, hits))
    edited_key = prompt_key(
        system=SYSTEM + "\n- Prefer the shortest runbook.",
        user=build_user(case, hits),
    )
    assert live_key in RECORDINGS
    assert edited_key not in RECORDINGS
