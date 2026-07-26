"""Chapter 4 - the Level 1 prompt application.

These tests pin the behaviours the chapter argues for, and they are split the way
the chapter says a Level 1 test suite has to be split.

The deterministic half needs no model at all: the schema agrees with Chapter 3's
routing table, an invented quote is refused, a declared refusal keeps its own
reason, the severity floor only ever raises, and every failure path returns
Chapter 3's shape. That half is an ordinary unit test.

The half that does involve model output replays committed recordings from the
measured run in `tests/recordings/ch04_classify.json`, so the chapter's published
numbers are reproducible with no key and no spend. Those recordings are keyed on
the exact prompt: edit `SYSTEM` and they all miss, which is the intended cost of
editing a prompt production depends on.

What is deliberately NOT here is an assertion that a live call returns a
particular service. The chapter measured the same report returning two different
services across runs, so that assertion would be a coin flip. Accuracy over live
traffic is a pass rate over N, and Chapter 13 owns it.
"""

import json
from pathlib import Path

import pytest

from escalation_ladder.classify import (
    SYSTEM,
    Classification,
    _at_least,
    build_user,
    classify,
    decide,
    evidence_supported,
    run,
    triage,
)
from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.instrument import CostLedger
from escalation_ladder.llm import RecordedCompleter, Usage, prompt_key
from escalation_ladder.rules import SERVICES, SEVERITIES, route

RECORDINGS = Path(__file__).parent / "recordings" / "ch04_classify.json"

# The measured run these recordings came from. Chapter 4 publishes these; if the
# fixtures or the prompt change, this file goes red before the prose goes stale.
MEASURED_INPUT_TOKENS = 1098
MEASURED_OUTPUT_TOKENS = 128


@pytest.fixture(scope="module")
def recordings() -> dict[str, str]:
    return json.loads(RECORDINGS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def completer(recordings) -> RecordedCompleter:
    return RecordedCompleter(
        recordings=recordings,
        usage=Usage(MEASURED_INPUT_TOKENS, MEASURED_OUTPUT_TOKENS),
    )


@pytest.fixture(scope="module")
def incidents() -> dict[str, Incident]:
    return {i.incident_id: i for i in load_incidents()}


@pytest.fixture(scope="module")
def refused() -> list[Incident]:
    """The four reports Level 0 refuses - exactly Chapter 4's input set."""
    return [i for i in load_incidents() if not route(i).routed]


def _claim(**kwargs) -> Classification:
    base = dict(
        service="search-api",
        severity="SEV3",
        summary="s",
        evidence="",
        needs_human=False,
    )
    base.update(kwargs)
    return Classification(**base)


# --------------------------------------------------------------------------
# The schema and Chapter 3's table must not drift apart
# --------------------------------------------------------------------------


def test_schema_services_are_the_routing_table_plus_a_sentinel():
    """The drift guard the chapter promises.

    A service added to SERVICES without being added to the schema would become
    an `unknown` in production rather than a routing error, so it breaks here.
    """
    allowed = set(Classification.model_fields["service"].annotation.__args__)
    assert allowed == set(SERVICES) | {"unknown"}


def test_schema_severities_match_chapter_threes_severities():
    allowed = set(Classification.model_fields["severity"].annotation.__args__)
    assert allowed == set(SEVERITIES)


def test_the_service_catalog_in_the_prompt_names_every_routable_service():
    """Every enum member the model may choose must be explained to it.

    A service in the schema but absent from the catalog is one the model can
    emit and has no basis for choosing.
    """
    for name in SERVICES:
        assert name in SYSTEM, name


def test_the_report_is_fenced_as_data(incidents):
    user = build_user(incidents["INC-1042"])
    assert "<report>" in user and "</report>" in user
    assert incidents["INC-1042"].report in user


# --------------------------------------------------------------------------
# The evidence check: no model required
# --------------------------------------------------------------------------


def test_a_verbatim_quote_is_supported(incidents):
    incident = incidents["INC-1042"]
    claim = _claim(evidence="their card was charged but the order page spun forever")
    assert evidence_supported(claim, incident)


def test_rewrapped_and_recased_quotes_are_forgiven(incidents):
    incident = incidents["INC-1042"]
    claim = _claim(evidence="THEIR CARD\n   was    charged")
    assert evidence_supported(claim, incident)


def test_a_paraphrase_is_not_supported(incidents):
    claim = _claim(evidence="the customer's payment card was billed")
    assert not evidence_supported(claim, incidents["INC-1042"])


def test_an_invented_quote_is_not_supported(incidents):
    claim = _claim(evidence="the search cluster lost quorum")
    assert not evidence_supported(claim, incidents["INC-1042"])


def test_an_empty_quote_is_not_supported(incidents):
    assert not evidence_supported(_claim(evidence=""), incidents["INC-1042"])


def test_a_quote_from_another_report_is_not_supported(incidents):
    """Cross-contamination is invention, even though the text is real."""
    claim = _claim(evidence=incidents["INC-1044"].report[:40])
    assert not evidence_supported(claim, incidents["INC-1042"])


# --------------------------------------------------------------------------
# Refusal reasons stay separable - the counters the chapter asks for
# --------------------------------------------------------------------------


def test_an_invented_quote_is_refused_and_named(incidents):
    incident = incidents["INC-1042"]
    key = prompt_key(system=SYSTEM, user=build_user(incident))
    payload = json.dumps(
        {
            "service": "search-api",
            "severity": "SEV1",
            "summary": "s",
            "evidence": "the search cluster lost quorum",
            "needs_human": False,
        }
    )
    result = classify(incident, RecordedCompleter(recordings={key: payload}))
    assert result.needs_human
    assert result.service == "unknown"
    assert "not in the report" in result.summary


def test_a_declared_refusal_keeps_its_own_reason(incidents):
    """Ordering matters: the evidence check must not mask an honest refusal.

    A model that says "I cannot tell" has no quote to give. If the evidence check
    ran first, every honest refusal would be filed as invented evidence and the
    two counters would merge.
    """
    incident = incidents["INC-1045"]
    key = prompt_key(system=SYSTEM, user=build_user(incident))
    payload = json.dumps(
        {
            "service": "unknown",
            "severity": "SEV3",
            "summary": "cannot tell which service owns confirmation emails",
            "evidence": "",
            "needs_human": True,
        }
    )
    result = classify(incident, RecordedCompleter(recordings={key: payload}))
    assert result.needs_human
    assert "cannot tell which service" in result.summary
    assert "not in the report" not in result.summary


def test_a_transport_failure_is_refused_and_named(incidents):
    """An empty recording set stands in for an unreachable vendor."""
    result = classify(incidents["INC-1042"], RecordedCompleter(recordings={}))
    assert result.needs_human
    assert result.service == "unknown"
    assert "no recording" in result.summary


# --------------------------------------------------------------------------
# The severity floor, reused from Chapter 3 as a rule
# --------------------------------------------------------------------------


def test_the_floor_raises_an_under_read():
    assert _at_least("SEV3", "SEV2") == "SEV2"


def test_the_floor_never_lowers_an_over_read():
    assert _at_least("SEV1", "SEV2") == "SEV1"
    assert _at_least("SEV1", "SEV3") == "SEV1"


def test_the_floor_is_a_no_op_when_they_agree():
    for severity in SEVERITIES:
        assert _at_least(severity, severity) == severity


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_no_page_is_ever_less_severe_than_its_service_tier(service, incidents):
    """The property, over every service and every claim the model could make."""
    tier = SERVICES[service].tier
    for claimed in SEVERITIES:
        decision = decide(incidents["INC-1042"], _claim(service=service, severity=claimed))
        assert SEVERITIES.index(decision.severity) <= SEVERITIES.index(tier)


# --------------------------------------------------------------------------
# decide() reuses Chapter 3's tables rather than reimplementing them
# --------------------------------------------------------------------------


def test_the_rota_and_sla_come_from_chapter_threes_tables(incidents):
    decision = decide(incidents["INC-1043"], _claim(service="search-api", severity="SEV3"))
    assert decision.rota == SERVICES["search-api"].rota
    assert decision.escalate_after_minutes == 60
    assert decision.routed


def test_a_page_carries_the_quote_that_caused_it(incidents):
    claim = _claim(service="search-api", evidence="Search feels slow")
    decision = decide(incidents["INC-1043"], claim)
    assert "Search feels slow" in decision.decided_by


def test_level_one_never_claims_an_alert_it_does_not_have(incidents):
    """No alert line means no firing duration, so overdue must be an honest False."""
    decision = decide(incidents["INC-1043"], _claim(service="search-api"))
    assert decision.already_overdue is False
    assert decision.recent_deploy is None


def test_an_unknown_service_refuses_in_chapter_threes_shape(incidents):
    decision = decide(incidents["INC-1042"], _claim(service="unknown", needs_human=True))
    assert decision.needs_human
    assert not decision.routed
    assert decision.rota is None
    assert decision.severity is None
    assert decision.escalate_after_minutes is None
    assert decision.decided_by


# --------------------------------------------------------------------------
# The measured run, replayed
# --------------------------------------------------------------------------


def test_the_corpus_splits_four_refused_by_level_zero(refused):
    """Chapter 4's input set is exactly Chapter 3's refusal set."""
    assert [i.incident_id for i in refused] == [
        "INC-1042",
        "INC-1043",
        "INC-1044",
        "INC-1045",
    ]


def test_recordings_cover_every_refused_incident(refused, recordings):
    for incident in refused:
        assert prompt_key(system=SYSTEM, user=build_user(incident)) in recordings


def test_level_one_routes_four_of_four_on_the_recorded_run(refused, completer):
    """The chapter's published headline: 2/6 at Level 0 becomes 4/4 here."""
    decisions = [triage(i, completer) for i in refused]
    assert sum(d.routed for d in decisions) == 4


def test_every_recorded_quote_is_really_in_its_report(refused, completer):
    for incident in refused:
        claim = classify(incident, completer)
        assert not claim.needs_human, incident.incident_id
        assert evidence_supported(claim, incident), incident.incident_id


def test_the_recorded_run_gets_every_service_right(refused, completer):
    expected = {
        "INC-1042": "payment-gateway",
        "INC-1043": "search-api",
        "INC-1044": "checkout-api",
        "INC-1045": "notification-worker",
    }
    for incident in refused:
        claim = classify(incident, completer)
        assert claim.service == expected[incident.incident_id], incident.incident_id


def test_the_floor_fixes_one_severity_and_cannot_fix_the_other(refused, completer):
    """The chapter's sharpest measured claim, pinned so it cannot rot.

    Ground truth: INC-1042 SEV2, INC-1043 SEV3, INC-1044 SEV1, INC-1045 SEV2.
    The model got 1043 and 1044 right, under-read 1045 (the floor corrected it),
    and over-read 1042 (a one-directional floor structurally cannot correct it).
    """
    truth = {
        "INC-1042": "SEV2",
        "INC-1043": "SEV3",
        "INC-1044": "SEV1",
        "INC-1045": "SEV2",
    }
    model_right = 0
    paged_right = 0
    for incident in refused:
        claim = classify(incident, completer)
        decision = decide(incident, claim)
        model_right += claim.severity == truth[incident.incident_id]
        paged_right += decision.severity == truth[incident.incident_id]

    assert model_right == 2, "chapter publishes 2/4 severities correct from the model"
    assert paged_right == 3, "chapter publishes 3/4 correct once the floor applies"

    # The specific pair the argument rests on.
    under = classify(next(i for i in refused if i.incident_id == "INC-1045"), completer)
    assert under.severity == "SEV3"
    assert decide(
        next(i for i in refused if i.incident_id == "INC-1045"), under
    ).severity == "SEV2"

    over = classify(next(i for i in refused if i.incident_id == "INC-1042"), completer)
    assert over.severity == "SEV1"
    assert decide(
        next(i for i in refused if i.incident_id == "INC-1042"), over
    ).severity == "SEV1"


def test_editing_the_system_prompt_invalidates_every_recording(refused, recordings):
    """A prompt change must break the tests rather than quietly pass them."""
    fake = RecordedCompleter(recordings=recordings)
    for incident in refused:
        key = prompt_key(system=SYSTEM + "\n- one more rule", user=build_user(incident))
        assert key not in recordings


# --------------------------------------------------------------------------
# The rung is wired into the cost table
# --------------------------------------------------------------------------


def test_run_records_exactly_one_measured_model_call(incidents, completer):
    ledger = run(incidents["INC-1042"], completer=completer)
    assert isinstance(ledger, CostLedger)
    assert ledger.model_calls == 1
    assert ledger.total_input_tokens == MEASURED_INPUT_TOKENS
    assert ledger.total_output_tokens == MEASURED_OUTPUT_TOKENS
    assert ledger.total_latency_ms > 0


def test_the_rung_is_registered_under_its_display_name():
    from escalation_ladder.rungs import load_all

    assert "Level 1: Prompt Application" in load_all()


def test_level_zero_is_still_registered_and_still_free(incidents):
    """Accretion check: adding Level 1 must not disturb the Zero Row."""
    from escalation_ladder import rules

    ledger = rules.run(incidents["INC-1041"])
    assert ledger.model_calls == 0
    assert ledger.total_input_tokens == 0
