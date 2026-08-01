"""What the harness must not do.

Chapter 13's three rules, asserted rather than described: a result is a rate
with an interval, a surface with nothing to score refuses instead of scoring
zero, and the sample size arithmetic is honest about how small this corpus is.

The published figures are asserted directly, so a fixture or recording change
breaks a test before it silently invalidates the chapter's prose.
"""
from __future__ import annotations

import pytest

from escalation_ladder import agent, classify, reasoning, retrieve, tools, workflow
from escalation_ladder.evaluate import (
    CLAIM_SURFACES,
    MISS,
    SURFACES,
    Observation,
    Rate,
    Surface,
    claims,
    classification_draws,
    corpus,
    detectable_lift,
    draws,
    floor_effect,
    load_recordings,
    observe,
    re_ask,
    replay,
    checkable_first,
    samples_needed,
    score,
)
from escalation_ladder.fixtures.labels import GOLDEN

INCIDENTS = corpus()
BY_NAME = {s.name: s for s in CLAIM_SURFACES + SURFACES}


# --- Rate -------------------------------------------------------------------


def test_a_perfect_run_does_not_report_certainty():
    """4 of 4 is not 100 percent, and the interval is the whole argument."""
    low, high = Rate(4, 4).interval
    assert Rate(4, 4).value == 1.0
    assert low < 0.6
    assert high == 1.0


def test_the_interval_narrows_with_n():
    assert Rate(40, 40).width < Rate(4, 4).width


def test_an_empty_rate_claims_nothing():
    assert Rate(0, 0).interval == (0.0, 1.0)


# --- refusal ----------------------------------------------------------------


def test_a_surface_with_no_sample_refuses_rather_than_scoring_zero():
    """The defect this rule exists to prevent: a zero and a gap read alike.

    A harness that scored an unmeasured surface 0.0 would report Level 4 as
    catastrophically broken on a corpus that simply never recorded it, and the
    action a team takes on those two readings is opposite.
    """
    empty = score(BY_NAME["tool choice"], [])
    assert empty.rate is None
    assert not empty.decided
    assert "no sample" in empty.scored_by


def test_an_unmeasured_run_leaves_the_denominator_alone():
    surface = Surface("always true", 0, lambda result, golden: True, "test")
    incident_id = INCIDENTS[0].incident_id
    measured = Observation(incident_id, object())
    unmeasured = Observation(incident_id, object(), missing=1)
    assert score(surface, [measured, measured]).rate == Rate(2, 2)
    assert score(surface, [measured, unmeasured]).rate == Rate(1, 1)


def test_a_correct_refusal_is_not_scored_as_a_wrong_answer():
    """Chapter 7's INC-1043: right tools, right refusal, scored a failure.

    The routing surface returns None for a refusal, so the incident lands on
    the give-up surface instead of dragging down an answer rate.
    """
    routing = BY_NAME["routing"]
    refused = observe(tools.investigate, [i for i in INCIDENTS if i.incident_id == "INC-1043"])
    assert routing.check(refused[0].result, GOLDEN["INC-1043"]) is None


def test_giving_up_is_scored_on_every_run_that_could_have():
    """Applicability comes from the rung, not from whether it quit.

    Reading the value instead scores only the runs that gave up, which is a
    rate of 1/1 computed over the single incident that quit.
    """
    give_up = BY_NAME["giving up"]
    observed = observe(agent.pursue, INCIDENTS)
    scored = score(give_up, observed)
    assert scored.rate is not None
    assert scored.rate.n > 1


# --- the recordings hold more than one sample -------------------------------


def test_four_prompts_were_recorded_twice():
    """Chapter 6 re-issues Chapter 4's classify prompt, so n is 2 not 1.

    `scripts/measure_costs.py` merges the same files last-wins, which is
    correct for a cost table and would silently halve every Level 1 eval.
    """
    repeated = {key: bodies for key, bodies in load_recordings().items() if len(bodies) > 1}
    assert len(repeated) == 4


def test_one_of_those_pairs_disagrees_on_a_closed_enum():
    """The coin flip, in the repository rather than in an anecdote."""
    incident = next(i for i in INCIDENTS if i.incident_id == "INC-1042")
    severities = {claim.severity for claim in classification_draws(incident)}
    assert len(severities) == 2


def test_the_level_1_surfaces_are_scored_over_every_draw():
    observed = claims(INCIDENTS)
    assert score(BY_NAME["service"], observed).rate == Rate(8, 8)
    assert score(BY_NAME["severity"], observed).rate == Rate(5, 8)


def test_the_floor_still_earns_its_place_as_a_rate():
    """Chapter 4 argued this from one sample per incident. Here it is as a rate."""
    raw, floored = floor_effect(INCIDENTS)
    assert (raw.passes, raw.n) == (5, 8)
    assert (floored.passes, floored.n) == (6, 7)
    assert floored.value > raw.value


# --- published figures ------------------------------------------------------


def test_the_published_surface_rates():
    observed = {
        2: observe(retrieve.advise, INCIDENTS),
        3: observe(workflow.advise, INCIDENTS),
        4: observe(tools.investigate, INCIDENTS),
        5: observe(reasoning.reason, INCIDENTS),
        6: observe(agent.pursue, INCIDENTS),
    }
    published = {
        "groundedness": (3, 3),
        "routing": (3, 3),
        "tool choice": (2, 4),
        "trajectory": (4, 4),
        "giving up": (5, 5),
    }
    for name, expected in published.items():
        surface = BY_NAME[name]
        rate = score(surface, observed[surface.level]).rate
        assert rate is not None, name
        assert (rate.passes, rate.n) == expected, name


def test_the_pipeline_breaks_when_the_upstream_sample_changes():
    """Chapter 6's claim, demonstrated without editing a prompt or a line of code.

    Both draws are real answers to the identical prompt. Replaying the earlier
    one invalidates every downstream recording three stages later.
    """
    def complete(pick: int) -> int:
        return sum(
            1
            for o in observe(workflow.advise, INCIDENTS, pick)
            if len(o.result.trace) == 5 and all(s.ok for s in o.result.trace)
        )

    assert complete(0) == 0
    assert complete(1) == 3


def test_the_cascade_can_stop_on_a_wrong_answer():
    """Chapter 11's re-ask surface: one diagnose request stops a rung too low."""
    rate, reached = re_ask(INCIDENTS, "diagnose")
    assert (rate.passes, rate.n) == (3, 4)
    assert set(reached) == {4, 5}


# --- sizing -----------------------------------------------------------------


def test_this_corpus_cannot_decide_chapter_tens_experiment():
    """Six cases per arm is powered for a 40-point swing. Nothing turns on one."""
    assert detectable_lift(6, 0.60) > 0.35
    assert samples_needed(0.60, 0.10) > 300


def test_sizing_grows_as_the_effect_shrinks():
    sizes = [samples_needed(0.60, lift) for lift in (0.30, 0.20, 0.10, 0.05)]
    assert sizes == sorted(sizes)
    assert sizes[0] == 32


def test_an_impossible_effect_refuses():
    with pytest.raises(ValueError):
        samples_needed(0.60, 0.45)


# --- the miss detector ------------------------------------------------------


def test_the_recording_miss_prefix_is_still_what_replay_returns():
    """Pinned rather than trusted: this string is how a gap is told from a refusal."""
    completer = replay()
    result = completer.parse(
        system="a prompt nobody ever recorded", user="nor this", schema=classify.Classification
    )
    assert result.failed is not None and result.failed.startswith(MISS)
    assert completer.misses == 1


def test_a_recorded_prompt_is_not_counted_as_a_miss():
    incident = next(i for i in INCIDENTS if i.incident_id == "INC-1044")
    assert draws(classify.SYSTEM, classify.build_user(incident))
    completer = replay()
    completer.parse(
        system=classify.SYSTEM,
        user=classify.build_user(incident),
        schema=classify.Classification,
    )
    assert completer.misses == 0


def test_the_floor_test_for_evals_actually_runs():
    """checkable_first shipped with its arguments swapped, and nothing called it.

    `evidence_supported(claim, incident)` reads `claim.evidence` and
    `incident.report`; called the other way round it reads `Incident.evidence`
    and `Classification.report`, neither of which exists, so the one function
    Chapter 13 offers as the Floor Test for evals raised AttributeError on every
    input. It had no caller and no test, which is why a swap survived.
    """
    incident = next(i for i in INCIDENTS if i.incident_id == "INC-1042")
    quote = incident.report.split(".")[0]

    supported = classify.Classification(
        service="payment-gateway",
        severity="SEV2",
        evidence=quote,
        summary="x",
        needs_human=False,
    )
    assert checkable_first(incident, supported) is True

    invented = classify.Classification(
        service="payment-gateway",
        severity="SEV2",
        evidence="a sentence that is not in the report",
        summary="x",
        needs_human=False,
    )
    assert checkable_first(incident, invented) is False
