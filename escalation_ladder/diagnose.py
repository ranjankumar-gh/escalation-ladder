"""The Escalation Ladder as executable predicates.

Chapter 2 of "The 7 GenAI Architectures".

A standalone design-review tool. It is deliberately NOT part of the incident
triage system the rest of the book builds, and imports nothing from it - you run
this one on your own systems, in your own design reviews.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    """The answers that decide a rung, for exactly one request type.

    Every answer is three-state. `None` means the team has not established it,
    and that is the state worth having: it stops a design review instead of
    letting the room guess.
    """

    request_type: str
    output_fully_specifiable: bool | None = None
    needs_external_knowledge: bool | None = None
    needs_fixed_multi_step: bool | None = None
    needs_live_data_or_action: bool | None = None
    next_step_depends_on_results: bool | None = None
    max_model_calls_knowable: bool | None = None
    specialization_measured_to_win: bool | None = None
@dataclass(frozen=True)
class Gate:
    """One rung boundary: the question, and the answer that stops the descent here."""

    level: int
    architecture: str
    question: str
    field: str
    stops_when: bool
    climb_when: str


FLOOR = Gate(
    level=0,
    architecture="Deterministic Code",
    question="Can a unit test fully specify the output for any input?",
    field="output_fully_specifiable",
    stops_when=True,
    climb_when="",
)

LADDER: tuple[Gate, ...] = (
    Gate(
        level=7,
        architecture="Multi-Agent System",
        question="Has specialization been measured to win on this task?",
        field="specialization_measured_to_win",
        stops_when=True,
        climb_when="a measured result where specialists beat one system",
    ),
    Gate(
        level=6,
        architecture="Autonomous Agent",
        question="Can you state the maximum model calls before the run?",
        field="max_model_calls_knowable",
        stops_when=False,
        climb_when="you can no longer state a maximum number of model calls",
    ),
    Gate(
        level=5,
        architecture="Multi-Step Reasoning",
        question="Does the next step depend on what earlier steps returned?",
        field="next_step_depends_on_results",
        stops_when=True,
        climb_when="which step runs next depends on what the last step returned",
    ),
    Gate(
        level=4,
        architecture="Tool-Using LLM",
        question="Does it need live data, or an action outside the model?",
        field="needs_live_data_or_action",
        stops_when=True,
        climb_when="a step needs live data, or an action outside the model",
    ),
    Gate(
        level=3,
        architecture="LLM Workflow",
        question="Does it need several model calls in a sequence you can draw?",
        field="needs_fixed_multi_step",
        stops_when=True,
        climb_when="one model call cannot do it and you can draw the steps",
    ),
    Gate(
        level=2,
        architecture="Retrieval-Augmented Generation",
        question="Does it need facts not in the request and not in the model?",
        field="needs_external_knowledge",
        stops_when=True,
        climb_when="the answer depends on facts not in the request or model",
    ),
)

BASE = Gate(
    level=1,
    architecture="Prompt Application",
    question="Nothing above forced a higher rung, and the Floor Test failed.",
    field="output_fully_specifiable",
    stops_when=False,
    climb_when="an input whose correct output you cannot specify in a test",
)

_BY_LEVEL: dict[int, Gate] = {g.level: g for g in (BASE, *LADDER)}
def floor_test(profile: TaskProfile) -> bool | None:
    """The Level 0 diagnostic. True means no model belongs on this path."""
    return profile.output_fully_specifiable


def termination_test(profile: TaskProfile) -> bool | None:
    """The Level 5/6 diagnostic. True means a finite chain: Level 5, not Level 6."""
    return profile.max_model_calls_knowable


def unanswered(profile: TaskProfile) -> tuple[str, ...]:
    """Every question this profile has not yet answered, in evaluation order."""
    gates = (FLOOR, *LADDER)
    return tuple(g.question for g in gates if getattr(profile, g.field) is None)


@dataclass(frozen=True)
class Recommendation:
    """A rung, the question that decided it, and what would justify the next one."""

    request_type: str
    level: int | None
    architecture: str | None
    decided_by: str
    escalate_when: str | None

    @property
    def resolved(self) -> bool:
        return self.level is not None

    def summary(self) -> str:
        if not self.resolved:
            return f"{self.request_type}\n  UNRESOLVED - {self.decided_by}"
        climb = self.escalate_when or "nowhere left to climb"
        return (
            f"{self.request_type}\n"
            f"  Level {self.level} - {self.architecture}\n"
            f"  decided by: {self.decided_by}\n"
            f"  climb only when: {climb}"
        )
def recommend_rung(profile: TaskProfile) -> Recommendation:
    """Return the highest rung this task requires, or refuse if it cannot be known.

    The Floor Test runs first and short-circuits, because a task whose output a
    test can fully specify needs nothing above it - that is what makes Level 0
    the floor rather than the bottom rung.

    The rest is evaluated from the top down, because the ladder is ordered by
    cost, not by capability containment: a task can need Level 4's tools without
    needing Level 2's retrieval. The first question answered against you wins,
    and a task that forces none of them lands on Level 1.
    """
    floor = floor_test(profile)
    if floor is None:
        return _refuse(profile, FLOOR)
    if floor is FLOOR.stops_when:
        return _decide(profile, FLOOR)
    for gate in LADDER:
        answer = getattr(profile, gate.field)
        if answer is None:
            return _refuse(profile, gate)
        if answer is gate.stops_when:
            return _decide(profile, gate)
    return _decide(profile, BASE)
def _decide(profile: TaskProfile, gate: Gate) -> Recommendation:
    above = _BY_LEVEL.get(gate.level + 1)
    return Recommendation(
        request_type=profile.request_type,
        level=gate.level,
        architecture=gate.architecture,
        decided_by=gate.question,
        escalate_when=above.climb_when if above else None,
    )


def _refuse(profile: TaskProfile, gate: Gate) -> Recommendation:
    return Recommendation(
        request_type=profile.request_type,
        level=None,
        architecture=None,
        decided_by=f"you do not know this yet: {gate.question}",
        escalate_when=None,
    )
if __name__ == "__main__":
    profiles = [
        TaskProfile(
            request_type="Page the on-call rota for an alert",
            output_fully_specifiable=True,
        ),
        TaskProfile(
            request_type="Summarize a free-text incident report",
            output_fully_specifiable=False,
            needs_external_knowledge=False,
            needs_fixed_multi_step=False,
            needs_live_data_or_action=False,
            next_step_depends_on_results=False,
            max_model_calls_knowable=True,
            specialization_measured_to_win=False,
        ),
        TaskProfile(
            request_type="Find the root cause of a checkout latency spike",
            output_fully_specifiable=False,
            needs_external_knowledge=True,
            needs_fixed_multi_step=True,
            needs_live_data_or_action=True,
            next_step_depends_on_results=True,
            specialization_measured_to_win=False,
        ),
    ]
    for profile in profiles:
        print(recommend_rung(profile).summary())
        print()
