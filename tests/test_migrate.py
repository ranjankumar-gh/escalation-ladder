"""What a climb is allowed to cost.

Chapter 15's rules, asserted rather than described: no rung module imports a
rung above it, a change is classified by the role of the file it lands in
rather than by intent, and the published below-edit figures break when the
history they were read from changes.

The import invariant is the one worth breaking deliberately. It costs nothing
to violate and it is not the climb it damages - it is the descent, in Chapter
16, run by somebody who did not add the edge.
"""
from __future__ import annotations

import json

import pytest

from escalation_ladder import migrate as mg
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import RecordedCompleter
from escalation_ladder.router import LADDERS, handle
from escalation_ladder.tools import menu_for

# Chapter 11's target, and the incident whose whole point is that no rung on
# the diagnose ladder answers it.
TARGET = "INC-1043"

# The twelve climbs Chapter 15 published, pinned so the figures in the chapter
# break if the history they were read from is rewritten. Later tags are
# deliberately excluded: this is a closed measurement, not a running total.
THROUGH_CH14: tuple[str, ...] = (
    "ch02-escalation-ladder",
    "ch03-the-floor",
    "ch04-prompt-application",
    "ch05-retrieval-augmented-generation",
    "ch06-llm-workflow",
    "ch07-tool-using-llm",
    "ch08-multi-step-reasoning",
    "ch09-autonomous-agent",
    "ch10-multi-agent-system",
    "ch11-composite-architectures",
    "ch12-costing",
    "ch13-evaluating",
    "ch14-observability",
)


@pytest.fixture(scope="module")
def published() -> tuple[mg.Climb, ...]:
    measured = mg.climbs(THROUGH_CH14)
    if not all(climb.measured for climb in measured):
        pytest.skip("clone has no chapter tags; nothing to measure")
    return measured


def test_no_rung_module_imports_a_rung_above_it() -> None:
    """The static clause, and the only one answerable before N+1 exists.

    Level 6 importing Level 2 is fine and is what `agent.py` does. Level 2
    importing Level 6 would mean the cheap path cannot be kept without the
    expensive one, which is the sentence Chapter 16 depends on being false.
    """
    assert mg.upward_imports() == ()


def test_a_rung_may_import_below_itself() -> None:
    """The invariant is directional, not a ban on coupling.

    Asserted so a later reader does not "fix" the check by forbidding all
    intra-package imports, which would pass the test and delete the accretion
    the whole book is built on.
    """
    assert "retrieve" in mg.imports_of("agent")
    assert mg.RUNG_LEVEL["retrieve"] < mg.RUNG_LEVEL["agent"]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("escalation_ladder/llm.py", "seam"),
        ("escalation_ladder/orchestration.py", "seam"),
        ("escalation_ladder/fixtures/metrics.py", "fixture"),
        ("escalation_ladder/classify.py", "rung"),
        ("escalation_ladder/crew.py", "rung"),
        ("escalation_ladder/router.py", "consumer"),
        ("escalation_ladder/telemetry.py", "consumer"),
        ("tests/test_workflow.py", "rung"),
        ("tests/test_measure_costs.py", "consumer"),
    ],
)
def test_a_change_is_classified_by_where_it_landed(path: str, expected: str) -> None:
    """Mechanical, so the number cannot be argued with before it is read."""
    assert mg.role(path) == expected


def test_three_lines_in_twelve_climbs_landed_in_a_rung_module(
    published: tuple[mg.Climb, ...],
) -> None:
    """The chapter's headline, as an assertion.

    951 lines changed below across twelve climbs and 57 of them are in a rung
    module - six percent, in three files, each of which the chapter reads by
    hand because three is a number a human can read.
    """
    assert sum(climb.below() for climb in published) == 951
    assert sum(climb.rung_lines for climb in published) == 57
    touched = {
        change.path
        for climb in published
        for change in climb.changes
        if change.kind == "rung"
    }
    assert touched == {
        "escalation_ladder/rules.py",
        "escalation_ladder/classify.py",
        "escalation_ladder/workflow.py",
    }


def test_four_of_twelve_climbs_edited_nothing_below_at_all(
    published: tuple[mg.Climb, ...],
) -> None:
    """And they are the last four, which is the direction that matters."""
    clean = [climb.tag for climb in published if climb.below() == 0]
    assert clean[-4:] == [
        "ch11-composite-architectures",
        "ch12-costing",
        "ch13-evaluating",
        "ch14-observability",
    ]


def test_the_seams_absorbed_three_quarters_of_the_cost(
    published: tuple[mg.Climb, ...],
) -> None:
    """A seam is a file you expect to edit. That is the definition, and 717 of
    951 lines is what it is worth."""
    assert sum(climb.below("seam") for climb in published) == 717
    assert sum(climb.below("consumer") for climb in published) == 0


def test_an_unmeasurable_climb_refuses_rather_than_reporting_zero() -> None:
    """Refuse, never default - Chapter 3's rule, applied to a measurement.

    A climb with no edits below and a climb that was never measured produce the
    same integer and mean opposite things, which is the distinction this whole
    chapter is about. `measured` is the field that keeps them apart.
    """
    climb = mg.edits_below("ch03-the-floor", "ch99-does-not-exist")

    assert climb.measured is False
    assert climb.below() == 0
    assert mg.addition_test(climb).passed is False
    assert "not measured" in mg.addition_test(climb).decided_by


def test_a_replay_is_blind_to_the_tool_menu() -> None:
    """The reason Chapter 15 does not run its own experiment offline.

    The recordings are keyed on the system and user prompts and NOT on the tool
    menu, so widening the menu does not make them miss - it returns answers
    produced when the tool was never offered, and the two runs are identical.
    A harness that reports green here is reporting on the wrong system.

    This pins the blindness rather than fixing it. Adding the menu to the key
    would invalidate every recording and every published figure in the book, to
    guard a flag that is off by default.
    """
    recordings: dict[str, str] = {}
    for path in (mg.REPO / "tests" / "recordings").glob("ch*.json"):
        recordings.update(json.loads(path.read_text(encoding="utf-8")))
    incident = {i.incident_id: i for i in load_incidents()}[TARGET]
    widened = {
        level: {"tools": menu_for(allow_runbooks=True)}
        for level in LADDERS["diagnose"]
    }

    plain = handle(incident, "diagnose", RecordedCompleter(recordings=recordings))
    wide = handle(
        incident,
        "diagnose",
        RecordedCompleter(recordings=recordings),
        options=widened,
    )

    assert plain.level_reached == wide.level_reached
    assert [a.decided_by for a in plain.attempts] == [
        a.decided_by for a in wide.attempts
    ]


def test_the_static_clause_decides_before_any_history_exists() -> None:
    """Applied to level N, not to level N+1. There is no climb to pass in."""
    verdict = mg.addition_test()

    assert verdict.passed is True
    assert verdict.decided_by == "no rung imports a rung above it"
