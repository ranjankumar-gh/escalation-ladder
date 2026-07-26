"""Chapter 3 - the Level 0 triage slice.

These tests pin the two behaviours the chapter argues for structurally: the
router refuses rather than defaulting, and a metric can escalate severity but
can never lower it below the service tier.

They are also the chapter's own claim about Level 0 evaluation - the correct
output is fully specifiable for every input, so the eval is a unit test and
nothing more.
"""

import pytest

from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.instrument import CostLedger
from escalation_ladder.rules import (
    ESCALATION_SLA,
    METRIC_SEV1_LINE,
    SERVICES,
    SEVERITIES,
    AlertSignal,
    RoutingDecision,
    escalation_after,
    owner_for,
    parse_alert,
    route,
    run,
    severity_for,
)


def _incident(report: str, incident_id: str = "INC-TEST") -> Incident:
    return Incident(
        incident_id=incident_id,
        opened_at="2026-01-01T00:00:00Z",
        service="checkout-api",
        severity="SEV1",
        title="test",
        report=report,
        root_cause="test",
        resolution="test",
    )


def test_parses_the_alert_grammar():
    signal = parse_alert("alert: http_5xx_rate > 0.4 for 5m on checkout-api")
    assert signal == AlertSignal(
        service="checkout-api",
        metric="http_5xx_rate",
        threshold=0.4,
        firing_minutes=5,
    )


def test_prose_does_not_parse():
    # The whole chapter turns on this returning None rather than a best guess.
    assert parse_alert("Checkout is returning 500s and I have no idea why") is None
    assert parse_alert("") is None


def test_the_pattern_is_anchored_at_both_ends():
    # An alert grammar embedded in a human sentence is not an alert. An
    # unanchored pattern would match the fragment and silently drop the rest.
    buried = "I think alert: http_5xx_rate > 0.4 for 5m on checkout-api is stale?"
    assert parse_alert(buried) is None


def test_owner_lookup_returns_none_for_an_unclaimed_service():
    assert owner_for("checkout-api") is SERVICES["checkout-api"]
    assert owner_for("ledger-api") is None


def test_metric_past_its_sev1_line_escalates_above_the_service_tier():
    signal = AlertSignal("search-api", "http_5xx_rate", 0.9, 5)
    assert severity_for(signal, SERVICES["search-api"]) == "SEV1"


def test_a_quiet_metric_never_lowers_the_service_tier():
    # checkout-api is tier SEV1; a mild metric must not talk it down.
    signal = AlertSignal("checkout-api", "http_5xx_rate", 0.01, 5)
    assert severity_for(signal, SERVICES["checkout-api"]) == "SEV1"


def test_an_unknown_metric_falls_back_to_the_service_tier():
    signal = AlertSignal("search-api", "goroutine_count", 99999.0, 5)
    assert severity_for(signal, SERVICES["search-api"]) == "SEV3"


def test_routes_a_machine_alert_and_names_the_rule():
    decision = route(_incident("alert: http_5xx_rate > 0.4 for 5m on checkout-api"))
    assert decision.routed
    assert decision.rota == "payments-oncall"
    assert decision.severity == "SEV1"
    assert decision.escalate_after_minutes == 5
    assert decision.already_overdue is True
    assert "checkout-api" in decision.decided_by


def test_refuses_prose_instead_of_defaulting():
    decision = route(_incident("Checkout is down and the dashboard is red"))
    assert decision.needs_human
    assert decision.rota is None
    assert decision.severity is None
    assert "prose" in decision.decided_by


def test_refuses_an_alert_for_a_service_no_rota_owns():
    decision = route(_incident("alert: cpu_utilization > 0.99 for 5m on ledger-api"))
    assert decision.needs_human
    assert "ledger-api" in decision.decided_by


def test_the_corpus_splits_two_routed_four_refused():
    # The chapter's Failure Receipt is this number. If the corpus or the rules
    # change, the receipt changes with them - and this test says so first.
    decisions = [route(incident) for incident in load_incidents()]
    routed = [d for d in decisions if d.routed]
    assert len(routed) == 2
    assert {d.incident_id for d in routed} == {"INC-1041", "INC-1046"}


def test_routed_severities_match_the_recorded_severity():
    by_id = {incident.incident_id: incident for incident in load_incidents()}
    for decision in (route(i) for i in by_id.values()):
        if decision.routed:
            assert decision.severity == by_id[decision.incident_id].severity


def test_summary_is_readable_for_both_outcomes():
    routed = route(_incident("alert: http_5xx_rate > 0.4 for 5m on checkout-api"))
    assert "page payments-oncall at SEV1" in routed.summary()
    assert "already past SLA" in routed.summary()
    refused = route(_incident("no idea what is happening"))
    assert "UNROUTABLE" in refused.summary()


def test_run_records_a_measurement_with_zero_model_calls():
    ledger = run(_incident("alert: http_5xx_rate > 0.4 for 5m on checkout-api"))
    assert isinstance(ledger, CostLedger)
    assert ledger.model_calls == 0
    assert ledger.total_input_tokens == 0
    assert ledger.total_output_tokens == 0
    assert ledger.total_latency_ms >= 0.0
    assert [m.label for m in ledger.measurements] == ["rules.route"]


def test_run_is_registered_as_a_rung():
    from escalation_ladder.rungs import load_all

    assert load_all()["Level 0: Deterministic Code"] is run


@pytest.mark.parametrize("severity", SEVERITIES)
def test_every_severity_has_an_escalation_sla(severity):
    assert escalation_after(severity) > 0


def test_slas_get_looser_as_severity_drops():
    assert ESCALATION_SLA["SEV1"] < ESCALATION_SLA["SEV2"] < ESCALATION_SLA["SEV3"]


def test_tables_agree_with_each_other():
    assert set(ESCALATION_SLA) == set(SEVERITIES)
    assert all(service.tier in SEVERITIES for service in SERVICES.values())
    assert all(line > 0 for line in METRIC_SEV1_LINE.values())
    assert isinstance(route(_incident("x")), RoutingDecision)
