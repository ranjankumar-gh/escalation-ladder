"""Chapter 2 - the Escalation Ladder diagnostic.

These tests pin the two behaviours the chapter argues for structurally: the Floor
Test short-circuits ahead of everything else, and an unanswered question refuses
rather than guessing.
"""

from escalation_ladder.diagnose import (
    BASE,
    FLOOR,
    LADDER,
    Recommendation,
    TaskProfile,
    floor_test,
    recommend_rung,
    termination_test,
    unanswered,
)


def _fully_answered(**overrides: object) -> TaskProfile:
    """A profile with every question answered "no", so one override isolates one gate."""
    base = dict(
        request_type="test",
        output_fully_specifiable=False,
        needs_external_knowledge=False,
        needs_fixed_multi_step=False,
        needs_live_data_or_action=False,
        next_step_depends_on_results=False,
        max_model_calls_knowable=True,
        specialization_measured_to_win=False,
    )
    base.update(overrides)
    return TaskProfile(**base)  # type: ignore[arg-type]


def test_floor_test_short_circuits_without_any_other_answer():
    # The chapter's central structural claim: Level 0 is a guard clause, not the
    # bottom of the loop. A passing Floor Test decides on its own.
    profile = TaskProfile(request_type="route an alert", output_fully_specifiable=True)
    rec = recommend_rung(profile)
    assert rec.level == 0
    assert rec.architecture == "Deterministic Code"
    assert rec.resolved


def test_unanswered_floor_test_blocks_everything():
    rec = recommend_rung(TaskProfile(request_type="unknown"))
    assert not rec.resolved
    assert rec.level is None
    assert FLOOR.question in rec.decided_by


def test_level_one_is_the_default_when_nothing_forces_higher():
    rec = recommend_rung(_fully_answered())
    assert rec.level == 1
    assert rec.architecture == "Prompt Application"


def test_each_gate_returns_its_own_level():
    # One override per gate, against an otherwise-negative profile.
    for gate in LADDER:
        rec = recommend_rung(_fully_answered(**{gate.field: gate.stops_when}))
        assert rec.level == gate.level, f"gate {gate.level} did not decide"
        assert rec.architecture == gate.architecture


def test_failing_the_termination_test_is_level_six():
    # max_model_calls_knowable=False is the Level 5/6 boundary.
    rec = recommend_rung(_fully_answered(max_model_calls_knowable=False))
    assert rec.level == 6
    assert rec.architecture == "Autonomous Agent"


def test_a_capped_loop_is_still_level_six():
    # "max_iterations=10" is a config value, not a knowable maximum: the profile
    # still answers the Termination Test "no", and the rung is unchanged.
    rec = recommend_rung(
        _fully_answered(max_model_calls_knowable=False, next_step_depends_on_results=True)
    )
    assert rec.level == 6


def test_refuses_when_a_gate_above_the_answer_is_unknown():
    # Six of seven answered, Termination Test left blank: the chapter's worked example.
    profile = TaskProfile(
        request_type="root cause a latency spike",
        output_fully_specifiable=False,
        needs_external_knowledge=True,
        needs_fixed_multi_step=True,
        needs_live_data_or_action=True,
        next_step_depends_on_results=True,
        specialization_measured_to_win=False,
    )
    rec = recommend_rung(profile)
    assert not rec.resolved
    assert "you do not know this yet" in rec.decided_by
    assert "maximum model calls" in rec.decided_by


def test_recommendation_names_what_would_justify_the_next_rung():
    rec = recommend_rung(_fully_answered())
    assert rec.level == 1
    # Climbing off Level 1 means needing facts outside the request and the model.
    assert rec.escalate_when == next(g for g in LADDER if g.level == 2).climb_when


def test_the_top_rung_has_nowhere_left_to_climb():
    rec = recommend_rung(_fully_answered(specialization_measured_to_win=True))
    assert rec.level == 7
    assert rec.escalate_when is None
    assert "nowhere left to climb" in rec.summary()


def test_floor_and_termination_tests_expose_the_raw_three_state_answer():
    unknown = TaskProfile(request_type="unknown")
    assert floor_test(unknown) is None
    assert termination_test(unknown) is None
    answered = _fully_answered()
    assert floor_test(answered) is False
    assert termination_test(answered) is True


def test_unanswered_lists_open_questions_in_evaluation_order():
    profile = TaskProfile(
        request_type="partly specified",
        output_fully_specifiable=False,
        specialization_measured_to_win=False,
    )
    open_questions = unanswered(profile)
    assert FLOOR.question not in open_questions
    assert LADDER[0].question not in open_questions
    # The remaining gates, still in top-down ladder order.
    assert list(open_questions) == [
        g.question for g in LADDER[1:] if g.field != "specialization_measured_to_win"
    ]


def test_summary_is_readable_for_both_outcomes():
    resolved = recommend_rung(_fully_answered()).summary()
    assert "Level 1 - Prompt Application" in resolved
    assert "climb only when:" in resolved
    unresolved = recommend_rung(TaskProfile(request_type="unknown")).summary()
    assert "UNRESOLVED" in unresolved


def test_gate_levels_are_unique_and_descending():
    levels = [g.level for g in LADDER]
    assert levels == sorted(levels, reverse=True)
    assert len(set(levels)) == len(levels)
    assert FLOOR.level == 0 and BASE.level == 1
    assert isinstance(recommend_rung(_fully_answered()), Recommendation)
