"""Composite level routing -- the full triage system behind one entry point.

This module is the payoff for a constraint the repository has kept since Chapter
3: every cheap path still exists. `rules.py` has not changed in eight chapters,
`classify.py` changed once by a declared rename, and this file imports both of
them alongside every rung above. Nothing below it is edited here.

It is deliberately NOT a rung. There is no `@register_rung`, so the composite
never appears in the cross-rung comparison table -- a router is the decision to
stop building one system, not an eighth architecture to compare against seven.
`diagnose.py` is standalone for the same reason. `scripts/composite_cost.py` is
this module's harness.

Two signals decide everything here, and both are free. The caller's declared ask
is free because the caller already knows it. The floor's verdict is free because
a Level 0 refusal costs zero tokens. Every signal that costs money is a signal
this module refuses to buy in order to decide what to buy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping

from escalation_ladder.agent import pursue
from escalation_ladder.classify import triage
from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.instrument import CostLedger
from escalation_ladder.llm import AnthropicCompleter, Completer, MeteredCompleter
from escalation_ladder.reasoning import reason
from escalation_ladder.retrieve import advise as advise_grounded
from escalation_ladder.rules import RoutingDecision, route
from escalation_ladder.tools import investigate
from escalation_ladder.workflow import advise as advise_pipeline

# What the caller asked for. This is the routing signal every system already
# has and most systems throw away: it is the endpoint the request arrived on,
# the button somebody pressed, or the queue it was read from. Reconstructing it
# with a model call means paying for a fact you were handed.
Ask = Literal["page", "advise", "diagnose"]

ASKS: tuple[Ask, ...] = ("page", "advise", "diagnose")


@dataclass(frozen=True)
class Rung:
    """One rung, adapted to a uniform shape the cascade can walk.

    `answered` and `decision` are separate because they disagree at Level 2:
    `retrieve.advise` can return a routed decision that is not grounded, and a
    grounded answer is what an `advise` request asked for. Every other rung's
    two predicates coincide, and writing them separately anyway keeps the
    disagreement visible instead of special-cased.
    """

    level: int
    label: str
    # Loose on purpose. Level 0 takes no completer and every rung above requires
    # one, so no single signature is honest about both -- the same asymmetry
    # `scripts/measure_costs.py` handles by inspecting each rung's parameters.
    # Narrowing this would buy a type error suppressed by a comment.
    call: Callable[..., object]
    answered: Callable[[object], bool]
    decision: Callable[[object], RoutingDecision]


def _decision(result: object) -> RoutingDecision:
    return getattr(result, "decision", result)  # type: ignore[return-value]


def _routed(result: object) -> bool:
    return _decision(result).routed


RUNGS: dict[int, Rung] = {
    0: Rung(
        level=0,
        label="Level 0: Deterministic Code",
        call=lambda incident, completer: route(incident),
        answered=_routed,
        decision=_decision,
    ),
    1: Rung(
        level=1,
        label="Level 1: Prompt Application",
        call=triage,
        answered=_routed,
        decision=_decision,
    ),
    2: Rung(
        level=2,
        label="Level 2: Retrieval-Augmented Generation",
        call=advise_grounded,
        answered=lambda r: bool(getattr(r, "grounded", False)),
        decision=_decision,
    ),
    3: Rung(
        level=3,
        label="Level 3: LLM Workflow",
        call=advise_pipeline,
        answered=_routed,
        decision=_decision,
    ),
    4: Rung(
        level=4,
        label="Level 4: Tool-Using LLM",
        call=investigate,
        answered=_routed,
        decision=_decision,
    ),
    5: Rung(
        level=5,
        label="Level 5: Multi-Step Reasoning",
        call=reason,
        answered=_routed,
        decision=_decision,
    ),
    6: Rung(
        level=6,
        label="Level 6: Autonomous Agent",
        call=pursue,
        answered=_routed,
        decision=_decision,
    ),
}


# One ladder per ask, and the ladders are the architecture. Reading them tells
# you which rungs a given request type can ever reach, which is the question
# nobody can answer about a system with a single entry point.
#
# `page` starts on the floor because a page is the one deliverable ordinary code
# can produce. `advise` and `diagnose` do not, because no rule writes a
# recommendation or reads live telemetry -- their floors are Level 2 and Level 4,
# and admitting that is the Floor Test being run per ask rather than per system.
LADDERS: dict[Ask, tuple[int, ...]] = {
    "page": (0, 1),
    "advise": (2, 3),
    "diagnose": (4, 5, 6),
}

# How many rungs of its ladder an ask may attempt. This is the composite's
# round budget, and Chapter 7's finding applies unchanged: a budget stated in
# the code is a design decision, and a budget left implicit is a design decision
# somebody else makes on your behalf.
#
# The defaults walk each ladder to the end. `diagnose` is the one worth arguing
# about -- Chapter 11 measures both settings, because a failed attempt at Level 4
# or Level 5 is billed and thrown away, while a failed attempt at Level 0 is free.
BUDGET: dict[Ask, int] = {ask: len(ladder) for ask, ladder in LADDERS.items()}


@dataclass(frozen=True)
class Attempt:
    """One rung tried, and what trying it cost whether or not it worked.

    The word is reused from Chapter 9 at a different scope on purpose:
    `agent.Attempt` is one round inside a single rung, and this is one rung
    inside a single request. Recording the refusals is the point. A composite
    that reported only the rung that answered would price every request as if
    the climb had been free, which is how a cascade hides its own cost.
    """

    level: int
    answered: bool
    input_tokens: int
    output_tokens: int
    model_calls: int
    decided_by: str

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Outcome:
    """What the composite did with one request.

    Composes `RoutingDecision` for the ninth chapter running. `level_reached` is
    the field this module adds and no rung below could have, because no rung
    below had a choice of rung to report.
    """

    incident_id: str
    ask: Ask
    decision: RoutingDecision
    level_reached: int | None
    attempts: tuple[Attempt, ...] = ()
    ledger: CostLedger = field(default_factory=CostLedger)

    @property
    def answered(self) -> bool:
        return self.level_reached is not None

    @property
    def tokens(self) -> int:
        return self.ledger.total_input_tokens + self.ledger.total_output_tokens

    @property
    def climbed(self) -> int:
        """Rungs paid for that did not answer. Zero on the floor, by design."""
        return sum(1 for attempt in self.attempts if not attempt.answered)

    def summary(self) -> str:
        where = "no rung answered" if not self.answered else f"L{self.level_reached}"
        return (
            f"{self.incident_id} ask={self.ask} -> {where}\n"
            f"  attempts: {', '.join(f'L{a.level}' for a in self.attempts) or 'none'}\n"
            f"  tokens: {self.tokens} over {self.ledger.model_calls} model calls\n"
            f"  decided by: {self.decision.decided_by}"
        )

    def attempt_table(self) -> str:
        rows = [
            f"| L{a.level} | {'answered' if a.answered else 'refused'} "
            f"| {a.tokens} | {a.model_calls} | {a.decided_by[:48]} |"
            for a in self.attempts
        ]
        header = "| Rung | Result | Tokens | Calls | Decided by |\n|---|---|--:|--:|---|"
        return "\n".join([header, *rows])


def _unanswered(incident: Incident, ask: Ask, reason_text: str) -> RoutingDecision:
    """Chapter 3's refusal shape, ninth chapter running.

    A composite that ran out of ladder refuses in the same shape as a rule that
    could not parse an alert, so the refusal counter Chapter 3 made countable
    stays countable across an architecture that did not exist then.
    """
    return RoutingDecision(
        incident_id=incident.incident_id,
        rota=None,
        severity=None,
        escalate_after_minutes=None,
        already_overdue=False,
        recent_deploy=None,
        needs_human=True,
        decided_by=f"{ask}: {reason_text}",
    )


def try_rung(
    incident: Incident, level: int, completer: Completer | None = None
) -> tuple[Attempt, RoutingDecision, CostLedger]:
    """Run exactly one rung and price it, without any ladder around it.

    `handle` walks ladders with this, and `scripts/composite_cost.py` builds the
    cost curve with it. Sharing the code is not tidiness: a curve measured by
    walking ladders only ever samples an upper rung on the incidents the rungs
    below it refused, which are the hard ones, so every upper point would be
    measured on a biased sample of the corpus and the Effective Level would be
    read against a curve the composite never actually faced.
    """
    rung = RUNGS[level]
    step = CostLedger()
    metered: Completer | None = None
    if completer is not None:
        metered = MeteredCompleter(
            inner=completer, ledger=step, stage=f"router.L{level}"
        )
    result = rung.call(incident, metered)
    decision = rung.decision(result)
    attempt = Attempt(
        level=level,
        answered=rung.answered(result),
        input_tokens=step.total_input_tokens,
        output_tokens=step.total_output_tokens,
        model_calls=step.model_calls,
        decided_by=decision.decided_by,
    )
    return attempt, decision, step


def handle(
    incident: Incident,
    ask: Ask,
    completer: Completer | None = None,
    *,
    budget: Mapping[Ask, int] | None = None,
) -> Outcome:
    """Walk one ask's ladder, stopping at the first rung that answers.

    The completer is built lazily, and that is a property worth having rather
    than a micro-optimization: a `page` request the floor answers never
    constructs a client, so it cannot fail on a missing credential, a rate
    limit, or a degraded provider. Chapter 3's argument about coupling incident
    response to a vendor's availability is enforced here by the call graph.
    """
    allowed = dict(BUDGET if budget is None else budget)
    ladder = LADDERS[ask][: max(0, allowed.get(ask, 0))]
    ledger = CostLedger()
    attempts: list[Attempt] = []
    active: Completer | None = completer
    last: RoutingDecision | None = None

    for level in ladder:
        if level > 0 and active is None:
            active = AnthropicCompleter()
        attempt, decision, step = try_rung(incident, level, active)
        for measurement in step.measurements:
            ledger.record(measurement)
        attempts.append(attempt)
        last = decision
        if attempt.answered:
            return Outcome(
                incident_id=incident.incident_id,
                ask=ask,
                decision=decision,
                level_reached=level,
                attempts=tuple(attempts),
                ledger=ledger,
            )

    exhausted = (
        f"no rung answered; tried {', '.join(f'L{a.level}' for a in attempts)}"
        if attempts
        else "no rung was attempted"
    )
    return Outcome(
        incident_id=incident.incident_id,
        ask=ask,
        decision=last if last is not None else _unanswered(incident, ask, exhausted),
        level_reached=None,
        attempts=tuple(attempts),
        ledger=ledger,
    )


def effective_level(cost: float, curve: Mapping[int, float]) -> float:
    """Locate a mean per-request cost on the measured single-rung cost curve.

    This is the Effective Level, and the definition is the argument. It is NOT
    the traffic-weighted average of level numbers, which treats the step from
    Level 0 to Level 1 as the same size as the step from Level 5 to Level 6 --
    the measured curve says those differ by an order of magnitude, so averaging
    the labels reports a number with no cost meaning.

    A result above the top of the curve is extrapolated rather than clamped, and
    that is not a rounding convenience. A composite that pays for rungs which
    refuse can cost more per request than any single rung it contains, and
    clamping to the top would delete the finding.
    """
    levels = sorted(curve)
    if not levels:
        raise ValueError("an effective level needs a measured cost curve")
    if cost <= curve[levels[0]]:
        return float(levels[0])
    for lower, upper in zip(levels, levels[1:]):
        if curve[lower] <= cost <= curve[upper]:
            span = curve[upper] - curve[lower]
            if span <= 0:
                return float(upper)
            return lower + (cost - curve[lower]) / span
    top, previous = levels[-1], levels[-2]
    span = curve[top] - curve[previous]
    return top + (cost - curve[top]) / span


def mix_cost(mix: Mapping[Ask, float], per_ask: Mapping[Ask, float]) -> float:
    """Mean cost per request for a stated traffic mix.

    The mix is the input nobody publishes and everybody has. It comes from a
    month of endpoint logs, it takes an afternoon, and it decides the Effective
    Level more than any architectural choice in this book does.
    """
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("a traffic mix must have some traffic in it")
    return sum(share * per_ask[ask] for ask, share in mix.items()) / total


def max_model_calls(agent_iterations: int) -> Mapping[Ask, int]:
    """The Termination Test, applied to the composite rather than to a rung.

    Answerable, which is the point: a system containing an unbounded loop still
    has a statable ceiling as long as the loop has a budget, and the ceiling is
    the sum of the ladder rather than the cost of the rung that usually answers.
    Chapter 8's trap applies unchanged -- these are API REQUESTS, not ledger
    entries, because one measured call can cover two requests.
    """
    per_level = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 6, 6: 2 * agent_iterations}
    return {
        ask: sum(per_level[level] for level in ladder)
        for ask, ladder in LADDERS.items()
    }
