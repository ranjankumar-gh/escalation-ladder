import re
from pathlib import Path

from escalation_ladder.fixtures.deploys import Deploy, recent_deploys
from escalation_ladder.fixtures.incidents import (
    SEED_INCIDENTS,
    Incident,
    build_db,
    load_incidents,
)
from escalation_ladder.fixtures.metrics import _RANGES, query_metric
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


# --------------------------------------------------------------------------
# the metric series the runbooks promise, against the ones the fixture serves
#
# Added at Chapter 6's gate. Ch6's Failure Receipt lands on
# notification-worker-duplicates, whose FIRST check is a ratio between two
# series the fake metrics API does not have. Ch7 builds tools over that API, so
# the gap stops being cosmetic there: the tool cannot execute the next step the
# rung below hands it.
#
# The assertion is equality rather than a subset on purpose, so it fails in both
# directions. Add a runbook naming another unserved metric and it goes red. Add
# the two missing series in Ch7 and it ALSO goes red, forcing whoever does it to
# delete this list and close the thread rather than leaving a stale allowance
# behind.
# --------------------------------------------------------------------------

_METRIC_TOKEN = re.compile(r"`([a-z][a-z0-9_]{3,})`")
_QUERY_METRIC = re.compile(r'query_metric\("[^"]+",\s*"([^"]+)"')

# Backticked identifiers in the runbooks that are not metric series.
_NOT_METRICS = {
    "query_metric", "tenant_id", "created_at", "max_connections", "order_id",
}

# Owed by Chapter 7. Both come from notification-worker-duplicates: "Compare
# `messages_sent` against `orders_created` for the same window. A ratio above
# 1.1 means redelivery rather than a producer bug."
OWED_BY_CH07 = {"messages_sent", "orders_created"}


def referenced_metrics() -> set[str]:
    """Every metric series the shipped runbooks tell an engineer to read."""
    found: set[str] = set()
    for path in runbook_paths():
        text = path.read_text(encoding="utf-8")
        found |= {t for t in _METRIC_TOKEN.findall(text) if t not in _NOT_METRICS}
        found |= set(_QUERY_METRIC.findall(text))
    return found


def test_runbook_metrics_resolve_except_the_series_chapter_seven_owes():
    missing = referenced_metrics() - set(_RANGES)
    assert missing == OWED_BY_CH07, (
        "The runbooks and the metrics fixture have drifted apart. Either a new "
        "runbook names a series query_metric cannot serve, or Chapter 7 has "
        "added one of the owed series - in which case delete it from "
        "OWED_BY_CH07 and update the Ch6 open_threads entry in the book's "
        f"_book-manifest.yml. Unserved right now: {sorted(missing)}"
    )


def test_the_owed_series_are_the_ones_chapter_sixs_receipt_depends_on():
    """Pins the gap to its cause, so the list above cannot quietly become junk."""
    duplicates = next(
        p for p in runbook_paths() if p.stem == "notification-worker-duplicates"
    )
    text = duplicates.read_text(encoding="utf-8")
    for series in OWED_BY_CH07:
        assert f"`{series}`" in text
