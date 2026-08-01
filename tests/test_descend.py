"""Chapter 16 - the Removal Receipt, and the order a descent is forced into.

Four behaviours the chapter argues for structurally, pinned here so a later
change breaks a test before it silently invalidates the prose.

1. A zero counter authorizes the REVERSIBLE stage and never the irreversible
   one. Every published verdict in this chapter is `route`; a future change that
   produced a `code` verdict on this corpus would be reporting that five
   requests had become sufficient evidence, which they cannot.
2. The four clauses refuse independently, and each refusal names which clause
   was missing. A receipt with no population and a receipt with a busy rung are
   opposite findings and must not collapse into one falsy value.
3. An import from a rung BLOCKS a deletion; an import from a consumer, a
   registry line, and a test file are work items inside it. Conflating them
   makes stage two unreachable for every rung in the book.
4. The shipped configuration is never mutated. `unroute` returns a new mapping
   and `handle` without `ladders` walks exactly what it walked in Chapter 11.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from escalation_ladder.descend import (
    BLOCKS,
    RUNG_AT,
    TOLERANCE,
    Holder,
    holds,
    importers,
    receipt,
    unroute,
    unused,
)
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import RecordedCompleter, Usage
from escalation_ladder.migrate import RUNG_LEVEL, upward_imports
from escalation_ladder.router import LADDERS, handle
from escalation_ladder.rungs import RUNG_MODULES

RECORDINGS = Path(__file__).parent / "recordings"

# The population Chapter 11 measured and Chapter 16 issues its receipt against:
# five diagnose requests, four answered below the top rung, one answered nowhere.
DIAGNOSE_REACHED: tuple[int | None, ...] = (5, None, 5, 4, 4)


@pytest.fixture(scope="module")
def replay() -> RecordedCompleter:
    recordings: dict[str, str] = {}
    for path in sorted(RECORDINGS.glob("ch*.json")):
        if path.name == "usage.json":
            continue
        recordings.update(json.loads(path.read_text(encoding="utf-8")))
    priced = {
        key: Usage(int(entry["input_tokens"]), int(entry["output_tokens"]))
        for key, entry in json.loads(
            (RECORDINGS / "usage.json").read_text(encoding="utf-8")
        ).items()
    }
    return RecordedCompleter(recordings=recordings, usage_by_key=priced)


@pytest.fixture(scope="module")
def incidents() -> dict[str, object]:
    return {incident.incident_id: incident for incident in load_incidents()}


def test_the_diagnose_ladder_answers_nothing_at_its_top_rung() -> None:
    assert 6 not in DIAGNOSE_REACHED
    assert sum(1 for rung in DIAGNOSE_REACHED if rung is not None) == 4


def test_a_zero_counter_authorizes_the_reversible_stage_only() -> None:
    scored = receipt(6, "diagnose", DIAGNOSE_REACHED, ladders=LADDERS)
    assert scored.answered == 0
    assert scored.authorized == "route"
    assert "cannot detect" in scored.decided_by


def test_five_requests_are_short_of_the_population_the_clause_needs() -> None:
    scored = receipt(6, "diagnose", DIAGNOSE_REACHED, ladders=LADDERS)
    assert scored.population == 5
    assert scored.needed == 906
    assert scored.population < scored.needed


def test_the_floor_sits_on_its_own_ceiling_at_this_sample_size() -> None:
    """A saturated floor is the arithmetic saying the question was never asked."""
    scored = receipt(6, "diagnose", DIAGNOSE_REACHED, ladders=LADDERS)
    assert scored.saturated
    assert scored.floor == pytest.approx(scored.ceiling, abs=1e-3)


def test_no_rung_on_any_shipped_ladder_authorizes_deleting_code() -> None:
    """The chapter's honest limit, pinned rather than stated.

    A change that turns any of these into `code` is either a much larger corpus
    or a bug, and the two must not be told apart by reading the prose.
    """
    for ask, ladder in LADDERS.items():
        for level in ladder:
            scored = receipt(level, ask, DIAGNOSE_REACHED, ladders=LADDERS)
            assert scored.authorized != "code", scored.render()


@pytest.mark.parametrize(
    "reached, ask, expected, fragment",
    [
        ((), "diagnose", None, "no population"),
        (DIAGNOSE_REACHED, "page", None, "does not route here"),
        ((6, 6, 4), "diagnose", None, "is not idle"),
        (DIAGNOSE_REACHED, "diagnose", "route", "cannot detect"),
    ],
)
def test_each_clause_refuses_for_its_own_stated_reason(
    reached: tuple[int | None, ...], ask: str, expected: str | None, fragment: str
) -> None:
    scored = receipt(6, ask, reached, ladders=LADDERS)
    assert scored.authorized == expected
    assert fragment in scored.decided_by


def test_a_rung_that_answers_is_not_a_removal_candidate_however_costly() -> None:
    """Cost is a different argument, and this instrument declines to make it."""
    scored = receipt(5, "diagnose", DIAGNOSE_REACHED, ladders=LADDERS)
    assert scored.answered == 2
    assert scored.authorized is None


def test_level_six_is_held_by_the_rung_above_it() -> None:
    """The descent order, measured. Level 7 is built on Level 6."""
    blocking = [h for h in holds(6, LADDERS) if h.blocks == "code"]
    assert [h.name for h in blocking] == ["crew"]
    assert "crew" in importers("agent")


def test_level_seven_is_held_by_nothing_that_would_break() -> None:
    """The only structurally deletable rung is the most expensive one."""
    assert [h for h in holds(7, LADDERS) if h.blocks == "code"] == []
    assert 7 in unused(LADDERS)


def test_an_unrouted_rung_emits_no_counter_to_read() -> None:
    """Which is why `unused` exists: a zero and a silence are different facts."""
    scored = receipt(7, "diagnose", DIAGNOSE_REACHED, ladders=LADDERS)
    assert scored.authorized is None
    assert "does not route here" in scored.decided_by


def test_a_registry_line_and_a_test_file_are_work_not_obstruction() -> None:
    grips = {h.grip: h.blocks for h in holds(6, LADDERS)}
    assert grips["registry"] is None
    assert grips["test"] is None
    assert grips["consumer"] is None
    assert grips["rung"] == "code"


def test_every_grip_kind_has_a_stated_effect() -> None:
    assert set(BLOCKS) == {"ladder", "rung", "consumer", "registry", "test"}
    assert Holder("x", "ladder").blocks == "route"


def test_unroute_returns_a_new_configuration_and_mutates_nothing() -> None:
    before = dict(LADDERS)
    after = unroute(LADDERS, "diagnose", 6)
    assert after["diagnose"] == (4, 5)
    assert after["page"] == LADDERS["page"]
    assert dict(LADDERS) == before


def test_unroute_refuses_an_ask_that_has_no_ladder() -> None:
    with pytest.raises(KeyError):
        unroute(LADDERS, "remediate", 7)


def test_holds_refuses_a_level_that_is_not_a_registered_rung() -> None:
    with pytest.raises(KeyError):
        holds(9, LADDERS)


def test_the_rung_map_is_derived_from_the_registry_not_restated() -> None:
    assert len(RUNG_AT) == len(RUNG_MODULES)
    assert RUNG_AT[0] == "rules" and RUNG_AT[7] == "crew"


def test_the_descent_order_is_forced_by_the_absence_of_upward_imports() -> None:
    """Nothing imports upward, so every blocking import points DOWN the ladder.

    Which means the only end of the ladder that is ever free is the top, and a
    descent has no choice about the order it runs in.
    """
    assert upward_imports() == ()
    for level, module in sorted(RUNG_AT.items()):
        for holder in holds(level, LADDERS):
            if holder.blocks == "code":
                assert RUNG_LEVEL[holder.name] > level, f"{holder.name} -> {module}"


def test_handle_without_ladders_walks_exactly_what_it_walked_before(
    replay: RecordedCompleter,
) -> None:
    """Chapters 11 to 15's figures must not move because of this change."""
    for incident in load_incidents():
        default = handle(incident, "page", replay)
        explicit = handle(incident, "page", replay, ladders=dict(LADDERS))
        assert default.level_reached == explicit.level_reached
        assert [a.level for a in default.attempts] == [
            a.level for a in explicit.attempts
        ]


def test_supplying_a_ladder_supplies_its_budget(
    replay: RecordedCompleter, incidents: dict[str, object]
) -> None:
    """A budget derived from the shipped ladder would truncate a longer one."""
    incident = incidents["INC-1042"]
    shipped = handle(incident, "page", replay)
    widened = handle(incident, "page", replay, ladders={**LADDERS, "page": (0, 1, 2, 3)})
    assert [a.level for a in shipped.attempts] == [0, 1]
    assert [a.level for a in widened.attempts][:3] == [0, 1, 2]


def test_unrouting_the_top_rung_answers_the_same_requests(
    replay: RecordedCompleter,
) -> None:
    """Stage one on Level 6, run rather than argued.

    The published claim is that removing it changes no outcome and removes a
    billed attempt. Both halves are asserted, because a cheaper ladder that
    answered less would be a different finding wearing the same headline.
    """
    after = unroute(LADDERS, "diagnose", 6)
    before_answers = after_answers = 0
    before_tokens = after_tokens = 0
    for incident in load_incidents():
        shipped = handle(incident, "diagnose", replay)
        unrouted = handle(incident, "diagnose", replay, ladders=after)
        if any("no recording" in a.decided_by for a in shipped.attempts):
            continue
        before_answers += int(shipped.answered)
        after_answers += int(unrouted.answered)
        before_tokens += shipped.tokens
        after_tokens += unrouted.tokens
        assert shipped.level_reached == unrouted.level_reached
    assert before_answers == after_answers == 4
    assert after_tokens < before_tokens


def test_the_tolerance_is_a_stated_constant_rather_than_a_literal() -> None:
    assert 0.0 < TOLERANCE < 1.0
