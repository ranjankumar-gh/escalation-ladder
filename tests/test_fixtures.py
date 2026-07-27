import re
from pathlib import Path

from escalation_ladder.fixtures.deploys import Deploy, recent_deploys
from escalation_ladder.fixtures.incidents import (
    SEED_INCIDENTS,
    Incident,
    build_db,
    load_incidents,
)
from escalation_ladder.fixtures.metrics import (
    _BASELINE,
    EPISODES,
    degraded_at,
    known_metrics,
    query_metric,
)
from escalation_ladder.fixtures.metrics import _parse
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
# Opened at Chapter 6's gate as an allowance and CLOSED in Chapter 7, which
# added `messages_sent` and `orders_created`. The assertion is now that nothing
# a runbook names is unserved, in either direction: add a runbook that tells an
# engineer to read a series `query_metric` does not have and this goes red
# before the tool that cannot execute it ever ships.
# --------------------------------------------------------------------------

_METRIC_TOKEN = re.compile(r"`([a-z][a-z0-9_]{3,})`")
_QUERY_METRIC = re.compile(r'query_metric\("[^"]+",\s*"([^"]+)"')

# Backticked identifiers in the runbooks that are not metric series.
_NOT_METRICS = {
    "query_metric", "tenant_id", "created_at", "max_connections", "order_id",
}


def referenced_metrics() -> set[str]:
    """Every metric series the shipped runbooks tell an engineer to read."""
    found: set[str] = set()
    for path in runbook_paths():
        text = path.read_text(encoding="utf-8")
        found |= {t for t in _METRIC_TOKEN.findall(text) if t not in _NOT_METRICS}
        found |= set(_QUERY_METRIC.findall(text))
    return found


def test_every_metric_a_runbook_names_can_actually_be_read():
    missing = referenced_metrics() - set(_BASELINE)
    assert missing == set(), (
        "A runbook names a series query_metric cannot serve. A tool built on "
        f"this API cannot execute the step that runbook hands it. Unserved: {sorted(missing)}"
    )


def test_the_duplicates_runbook_ratio_is_now_servable():
    """The two series Chapter 7 owed, pinned to the runbook that needs them."""
    duplicates = next(
        p for p in runbook_paths() if p.stem == "notification-worker-duplicates"
    )
    text = duplicates.read_text(encoding="utf-8")
    for series in ("messages_sent", "orders_created"):
        assert f"`{series}`" in text
        assert series in known_metrics("notification-worker")


def test_metrics_are_quiet_outside_an_incident_window():
    # A baseline as wide as the plausible range is indistinguishable from an
    # incident, and every alert threshold would be crossed by ordinary noise.
    calm = [v for _, v in query_metric("checkout-api", "http_5xx_rate", 60)]
    assert max(calm) < 0.01


def test_metrics_depart_from_baseline_inside_an_incident_window():
    # INC-1046 is still degrading search-api at NOW, which is what makes it the
    # incident a live tool can read.
    live = [v for _, v in query_metric("search-api", "http_5xx_rate", 15)]
    assert min(live) > 0.05
    assert degraded_at(
        "search-api", "http_5xx_rate", _parse("2026-03-21T04:45:00Z")
    ) == "INC-1046"


def test_an_incident_only_moves_the_series_it_touched():
    # INC-1045's signal is the RATIO: messages_sent quadruples, orders_created
    # does not move. A fixture that moved every series during an incident would
    # make the runbook's ratio check meaningless.
    window = {"ending_at": "2026-03-03T16:50:00Z"}
    sent = [v for _, v in query_metric("notification-worker", "messages_sent", 15, **window)]
    orders = [v for _, v in query_metric("notification-worker", "orders_created", 15, **window)]
    assert min(sent) > 3000.0
    assert max(orders) < 1200.0
    assert sum(sent) / sum(orders) > 1.1


def test_overlapping_windows_agree_on_the_minutes_they_share():
    # The sample index is absolute, not relative to the start of the window.
    short = dict(query_metric("search-api", "p99_latency_ms", 15))
    long = dict(query_metric("search-api", "p99_latency_ms", 60))
    shared = set(short) & set(long)
    assert len(shared) == 15
    assert all(short[k] == long[k] for k in shared)


def test_every_episode_names_a_seeded_incident():
    ids = {i.incident_id for i in SEED_INCIDENTS}
    assert {e.incident_id for e in EPISODES} <= ids
