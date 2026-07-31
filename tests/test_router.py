"""The behaviours Chapter 11 argues for, structurally.

- The composite is not a rung: it registers nothing and never appears in the
  cross-rung cost table.
- A page the floor can answer never constructs a client, so it cannot be broken
  by a missing credential or a degraded provider.
- Every rung attempted is recorded whether or not it answered, because a
  composite that reported only the winner would price the climb as free.
- The Effective Level is read off a measured curve, is NOT an average of level
  numbers, and is not clamped at the top of the ladder.
- The Termination Test still has an answer for a system containing a loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import RecordedCompleter, Usage
from escalation_ladder.router import (
    ASKS,
    BUDGET,
    LADDERS,
    RUNGS,
    effective_level,
    handle,
    max_model_calls,
    mix_cost,
    try_rung,
)
from escalation_ladder.rungs import RUNG_MODULES, load_all

RECORDINGS = Path(__file__).parent / "recordings"


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


# --- the composite is not a rung ------------------------------------------


def test_the_composite_registers_no_rung() -> None:
    """A router is a decision to stop building one system, not an eighth one.

    If this ever fails, the cross-rung table has grown a row that is an average
    over the other rows, and every ratio Chapter 12 quotes becomes circular.
    """
    assert "escalation_ladder.router" not in RUNG_MODULES
    registry = load_all()
    assert not any("router" in name.lower() for name in registry)
    assert not any("composite" in name.lower() for name in registry)


# --- the floor path takes no dependency on a vendor ------------------------


class _NoVendor(RuntimeError):
    """Raised if anything on a path tries to construct a real client."""


def test_a_page_the_floor_answers_never_constructs_a_client(
    incidents, monkeypatch
) -> None:
    """The strongest form of Chapter 3's availability-coupling argument.

    Asserted by making construction fatal rather than by passing no key, because
    a test that fell through to a real client would make a live call on the
    machine of any reader who has one. A missing credential, a rate limit, or a
    degraded provider cannot affect this path, and that is enforced by the call
    graph rather than by a try/except a later refactor can widen.
    """
    monkeypatch.setattr(
        "escalation_ladder.router.AnthropicCompleter",
        lambda *a, **k: (_ for _ in ()).throw(_NoVendor()),
    )
    outcome = handle(incidents["INC-1041"], "page", completer=None)

    assert outcome.level_reached == 0
    assert outcome.tokens == 0
    assert outcome.ledger.model_calls == 0
    assert outcome.climbed == 0


def test_a_page_the_floor_refuses_does_construct_one(incidents, monkeypatch) -> None:
    """The other half, so the test above is not passing for a trivial reason."""
    monkeypatch.setattr(
        "escalation_ladder.router.AnthropicCompleter",
        lambda *a, **k: (_ for _ in ()).throw(_NoVendor()),
    )
    with pytest.raises(_NoVendor):
        handle(incidents["INC-1042"], "page", completer=None)


# --- refusals are recorded, not discarded ---------------------------------


def test_every_rung_tried_is_recorded_even_when_it_refused(
    incidents, replay
) -> None:
    """INC-1043's diagnose request pays for three rungs and gets no answer.

    This is the chapter's central measured claim. A composite that recorded only
    the rung that answered would report this request as free.
    """
    outcome = handle(incidents["INC-1043"], "diagnose", replay)

    assert outcome.level_reached is None
    assert [a.level for a in outcome.attempts] == [4, 5, 6]
    assert outcome.climbed == 3
    assert outcome.tokens > 40_000
    assert all(not a.answered for a in outcome.attempts)


def test_the_cheap_ask_stops_on_the_floor_for_a_machine_alert(
    incidents, replay
) -> None:
    outcome = handle(incidents["INC-1046"], "page", replay)

    assert outcome.level_reached == 0
    assert outcome.tokens == 0
    assert outcome.decision.rota is not None


def test_a_budget_of_one_turns_the_cascade_into_a_decision(
    incidents, replay
) -> None:
    """Capping the ladder is the composite's round budget, and it is a knob.

    Chapter 7 found that a stated budget decides which rung you are on. The same
    holds here: with one attempt allowed, the diagnose ladder is Level 4 and
    nothing else.
    """
    capped = handle(incidents["INC-1043"], "diagnose", replay, budget={"diagnose": 1})

    assert [a.level for a in capped.attempts] == [4]
    assert capped.level_reached is None
    assert capped.tokens < 10_000


# --- the Effective Level --------------------------------------------------

CURVE: dict[int, float] = {0: 0.0, 1: 1288.8, 2: 1462.2, 3: 2828.2,
                           4: 6635.0, 5: 15467.0, 6: 17547.4}


def test_the_effective_level_lands_between_the_rungs_that_bracket_the_cost() -> None:
    assert effective_level(0.0, CURVE) == 0.0
    assert effective_level(1288.8, CURVE) == pytest.approx(1.0)
    assert 2.0 < effective_level(2228.2, CURVE) < 3.0


def test_the_effective_level_is_not_an_average_of_level_numbers() -> None:
    """The definitional argument, pinned on the shape a composite actually has.

    Ninety percent of traffic on the free floor and ten percent in the loop
    averages to level 0.6 if you average the labels, which reads as a system
    barely off the floor. Priced against the measured curve it is 2.2, because
    the curve is not linear in the level number - the rare expensive rung is
    most of the bill. Averaging labels understates this system by more than
    three times, and that is why the concept needs a definition rather than an
    intuition.
    """
    cost = 0.9 * CURVE[0] + 0.1 * CURVE[6]
    label_average = 0.9 * 0 + 0.1 * 6

    assert label_average == pytest.approx(0.6)
    assert effective_level(cost, CURVE) == pytest.approx(2.21, abs=0.02)
    assert effective_level(cost, CURVE) > 3 * label_average


def test_a_composite_can_cost_more_per_request_than_its_own_top_rung() -> None:
    """Not clamped, because the finding is real.

    A cascade that pays for rungs which refuse can exceed the price of every
    single rung it contains. Clamping to 6.0 would delete Chapter 11's sharpest
    number.
    """
    assert effective_level(21_041.0, CURVE) > 6.0


def test_an_effective_level_needs_a_curve() -> None:
    with pytest.raises(ValueError):
        effective_level(100.0, {})


def test_the_mix_decides_the_effective_level() -> None:
    """Two teams, identical code, three rungs apart."""
    per_ask = {"page": 859.2, "advise": 2228.2, "diagnose": 21041.0}
    cheap = mix_cost({"page": 0.80, "advise": 0.18, "diagnose": 0.02}, per_ask)
    heavy = mix_cost({"page": 0.40, "advise": 0.35, "diagnose": 0.25}, per_ask)

    assert effective_level(cheap, CURVE) == pytest.approx(2.03, abs=0.05)
    assert effective_level(heavy, CURVE) == pytest.approx(3.93, abs=0.05)


def test_an_empty_mix_is_refused_rather_than_divided_by() -> None:
    with pytest.raises(ValueError):
        mix_cost({"page": 0.0, "advise": 0.0, "diagnose": 0.0}, {"page": 1.0})


# --- the Termination Test still applies -----------------------------------


def test_the_composite_has_a_statable_ceiling_per_ask() -> None:
    """A system containing an unbounded loop still answers the Termination Test.

    The ceiling is the sum of the ladder, and the loop dominates it -- which is
    the honest reading, because the ceiling is what you must be able to pay, not
    what you usually pay.
    """
    ceilings = max_model_calls(agent_iterations=8)

    assert ceilings["page"] == 1
    assert ceilings["advise"] == 3
    assert ceilings["diagnose"] == 24
    assert ceilings["diagnose"] > ceilings["advise"] + ceilings["page"]


def test_the_ceiling_moves_with_the_agents_budget() -> None:
    assert max_model_calls(4)["diagnose"] < max_model_calls(8)["diagnose"]


# --- the ladders are the architecture -------------------------------------


def test_every_ask_has_a_ladder_and_every_ladder_rung_exists() -> None:
    assert set(LADDERS) == set(ASKS)
    for ladder in LADDERS.values():
        assert ladder == tuple(sorted(ladder)), "a ladder must climb"
        assert all(level in RUNGS for level in ladder)


def test_only_the_page_ladder_starts_on_the_floor() -> None:
    """The Floor Test run per ask rather than per system.

    No rule writes a recommendation and no rule reads live telemetry, so Level 0
    is not on those ladders at all. Putting it there would be the LLM-washing
    argument in reverse: pretending code can produce a deliverable it cannot.
    """
    assert LADDERS["page"][0] == 0
    assert 0 not in LADDERS["advise"]
    assert 0 not in LADDERS["diagnose"]


def test_the_default_budget_walks_every_ladder_to_its_end() -> None:
    assert BUDGET == {ask: len(ladder) for ask, ladder in LADDERS.items()}


def test_widening_the_diagnose_ladder_changes_what_it_returns(
    incidents, replay
) -> None:
    """The chapter's most instructive mistake, pinned so it cannot be "fixed".

    Starting the diagnose ladder at retrieval is an order of magnitude cheaper
    and answers more of the corpus. It is wrong anyway: every extra answer is a
    grounded runbook citation, which Chapter 5 established is a hypothesis rather
    than a confirmation. If somebody widens `LADDERS["diagnose"]` because the
    cost table improved, this test is where the argument lives.
    """
    import escalation_ladder.router as router_module

    shipped = router_module.LADDERS["diagnose"]
    assert shipped == (4, 5, 6)

    widened = (2, 3) + shipped
    try:
        router_module.LADDERS["diagnose"] = widened
        cheap = handle(
            incidents["INC-1044"], "diagnose", replay,
            budget={"diagnose": len(widened)},
        )
    finally:
        router_module.LADDERS["diagnose"] = shipped

    dear = handle(incidents["INC-1044"], "diagnose", replay)

    assert cheap.level_reached == 2
    assert dear.level_reached == 5
    assert cheap.tokens * 8 < dear.tokens
    assert dear.decision.routed and cheap.decision.routed


def test_one_rung_run_alone_costs_what_the_cascade_paid_for_it(
    incidents, replay
) -> None:
    """`try_rung` and `handle` must agree, because the cost curve depends on it.

    A curve built by walking ladders would only ever sample an upper rung on the
    incidents the rungs below refused, and the Effective Level would then be read
    against a curve the composite never faced.
    """
    alone, _decision, _ledger = try_rung(incidents["INC-1045"], 4, replay)
    through = handle(incidents["INC-1045"], "diagnose", replay)

    assert through.attempts[0].level == 4
    assert through.attempts[0].tokens == alone.tokens
    assert through.attempts[0].answered == alone.answered
