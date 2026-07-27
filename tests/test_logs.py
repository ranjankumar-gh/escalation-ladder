"""The seeded log store.

Two behaviours the chapter argues for structurally: a search that finds nothing
is distinguishable from a source that is silent, and the cause of INC-1044 is
reachable but not findable without first reading the checkout logs. The second
one is what makes Chapter 7's Failure Receipt honest rather than staged - the
answer is not hidden, it is merely not implied by anything the model has seen.
"""
from escalation_ladder.fixtures.logs import (
    EPISODES,
    LOG_SOURCES,
    ROUTINE,
    search_logs,
)


def test_every_source_emits_routine_lines_when_nothing_is_wrong():
    # Otherwise "no matches" and "this source is not logging" look identical,
    # and they are different production problems.
    for source in LOG_SOURCES:
        assert search_logs(source, "", 60), source


def test_search_is_case_insensitive_substring():
    hits = search_logs("search-api", "REINDEX", 60)
    assert hits
    assert all("reindex" in line.message.lower() for line in hits)


def test_the_level_is_searchable():
    """Regression. Matching the message alone made this return nothing while
    ERROR-level lines sat in the window, and the model that searched for
    "error" was told there were none."""
    window = {"ending_at": "2026-02-11T22:37:00Z"}
    hits = search_logs("service-mesh", "error", 240, **window)
    assert hits
    assert all(line.level == "ERROR" for line in hits)


def test_a_pattern_that_matches_nothing_returns_nothing():
    assert search_logs("search-api", "kubernetes", 60) == []


def test_an_unknown_source_returns_nothing_rather_than_raising():
    assert search_logs("no-such-service", "error", 60) == []


def test_zero_or_negative_minutes_is_empty():
    assert search_logs("search-api", "error", 0) == []
    assert search_logs("search-api", "error", -5) == []


def test_results_are_capped():
    assert len(search_logs("search-api", "", 1440, limit=5)) == 5


def test_the_live_incident_is_readable_at_now():
    """INC-1046 is still running, which is why one round of tools can solve it."""
    hits = search_logs("search-api", "reindex", 60)
    assert hits
    assert any("batch_size=5000" in line.message for line in hits)


def test_checkout_logs_show_the_symptom_and_never_the_cause():
    """The heart of Chapter 7's receipt.

    A service cannot see why its own sidecar refused a handshake, so no search
    over its logs - however well phrased - can find an expired certificate.
    """
    window = {"ending_at": "2026-02-11T22:37:00Z"}
    hits = search_logs("checkout-api", "", 60, **window)
    assert hits
    text = " ".join(line.render().lower() for line in hits)
    assert "upstream connect error" in text
    for absent in ("certificate", "x509", "handshake", "tls", "mesh"):
        assert absent not in text, absent


def test_the_cause_is_reachable_in_the_mesh_for_anyone_who_looks():
    """Nothing is hidden. It is simply in a source you have no reason to query
    until the checkout logs tell you the sidecar refused the connection."""
    window = {"ending_at": "2026-02-11T22:37:00Z"}
    hits = search_logs("service-mesh", "handshake", 60, **window)
    assert hits
    assert any("certificate has expired" in line.message for line in hits)


def test_lines_are_ordered_oldest_first():
    hits = search_logs("search-api", "", 240)
    assert [line.at for line in hits] == sorted(line.at for line in hits)


def test_episodes_only_name_known_sources():
    assert {e.source for e in EPISODES} <= set(LOG_SOURCES)
    assert set(ROUTINE) == set(LOG_SOURCES)


def test_the_store_is_deterministic():
    assert search_logs("search-api", "reindex", 60) == search_logs(
        "search-api", "reindex", 60
    )
