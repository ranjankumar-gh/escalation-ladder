from pathlib import Path

from escalation_ladder.fixtures.deploys import Deploy, recent_deploys
from escalation_ladder.fixtures.incidents import (
    SEED_INCIDENTS,
    Incident,
    build_db,
    load_incidents,
)
from escalation_ladder.fixtures.metrics import query_metric
from escalation_ladder.fixtures.runbooks_index import runbook_paths


def test_seed_incidents_cover_every_rung_shape():
    # One incident per rung shape, so each chapter has a case it can and cannot handle.
    assert len(SEED_INCIDENTS) >= 6
    assert all(isinstance(i, Incident) for i in SEED_INCIDENTS)
    assert len({i.incident_id for i in SEED_INCIDENTS}) == len(SEED_INCIDENTS)


def test_severities_are_from_the_fixed_set():
    assert {i.severity for i in SEED_INCIDENTS} <= {"SEV1", "SEV2", "SEV3"}


def test_build_db_then_load_roundtrips_every_incident(tmp_path: Path):
    db = build_db(tmp_path / "incidents.db")
    assert db.exists()
    loaded = load_incidents(db)
    assert len(loaded) == len(SEED_INCIDENTS)
    assert {i.incident_id for i in loaded} == {i.incident_id for i in SEED_INCIDENTS}


def test_build_db_is_idempotent(tmp_path: Path):
    path = tmp_path / "incidents.db"
    build_db(path)
    build_db(path)
    assert len(load_incidents(path)) == len(SEED_INCIDENTS)


def test_query_metric_is_deterministic_across_calls():
    a = query_metric("checkout-api", "http_5xx_rate", minutes=30)
    b = query_metric("checkout-api", "http_5xx_rate", minutes=30)
    assert a == b
    assert len(a) == 30


def test_query_metric_differs_by_service_and_metric():
    assert query_metric("checkout-api", "http_5xx_rate", 10) != query_metric(
        "search-api", "http_5xx_rate", 10
    )
    assert query_metric("checkout-api", "http_5xx_rate", 10) != query_metric(
        "checkout-api", "p99_latency_ms", 10
    )


def test_query_metric_values_are_in_a_plausible_range():
    for _, value in query_metric("checkout-api", "http_5xx_rate", 60):
        assert 0.0 <= value <= 100.0


def test_rate_metrics_are_fractions_not_percentages():
    # Units must match the alert grammar rules.py parses. An alert reading
    # `http_5xx_rate > 0.4` and a sample of this series have to be directly
    # comparable, or Chapter 7's tools silently misreport every incident.
    for metric in ("http_5xx_rate", "cpu_utilization", "connection_pool_in_use"):
        for _, value in query_metric("checkout-api", metric, 60):
            assert 0.0 <= value <= 1.0, f"{metric} is not a fraction"


def test_latency_metric_is_still_milliseconds():
    values = [v for _, v in query_metric("checkout-api", "p99_latency_ms", 60)]
    assert max(values) > 100.0


def test_query_metric_of_zero_or_negative_minutes_is_empty():
    assert query_metric("checkout-api", "http_5xx_rate", 0) == []
    assert query_metric("checkout-api", "http_5xx_rate", -5) == []


def test_recent_deploys_returns_deploys_for_a_known_service():
    deploys = recent_deploys("checkout-api")
    assert deploys
    assert all(isinstance(d, Deploy) for d in deploys)
    assert all(d.service == "checkout-api" for d in deploys)


def test_recent_deploys_is_empty_for_an_unknown_service():
    assert recent_deploys("no-such-service") == []


def test_recent_deploys_is_newest_first():
    deploys = recent_deploys("search-api")
    stamps = [d.deployed_at for d in deploys]
    assert stamps == sorted(stamps, reverse=True)


def test_every_runbook_is_non_empty_markdown():
    paths = runbook_paths()
    assert len(paths) >= 4
    for p in paths:
        assert p.suffix == ".md"
        assert p.read_text(encoding="utf-8").strip()
