"""Chapter 16 - what holds a rung in place, and what authorizes removing it.

`migrate.py` measured what a climb costs below itself. This module asks the
question in the other direction, and the two are not mirror images.

A climb is authorized by ONE CASE. A Failure Receipt is an existence proof, and
one reproducible failure at level N is a complete argument for level N+1. A
descent is authorized by an ABSENCE, and an absence is a claim about every
request nobody has seen yet. No corpus proves it and no volume of traffic will.

So the code does the two things that are available. `holds` reads every edge
that would break if a rung disappeared and splits them by which STAGE of a
deletion each one blocks - removing a rung from a ladder is an edit to a tuple,
and removing its module is an excavation. `receipt` scores the four clauses that
authorize each stage, and refuses when a clause is missing rather than reading a
missing clause as a satisfied one.

Registers no rung. Appears in no cross-rung cost table. Takes the ladder
configuration as an argument rather than importing it, for the reason
`telemetry._last_rung` gives: a configuration is not a property of a request,
and a module that grades your ladder should not be pinned to mine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from escalation_ladder.evaluate import detectable_lift, samples_needed
from escalation_ladder.migrate import PACKAGE, REPO, imports_of, role
from escalation_ladder.rungs import RUNG_MODULES

# The two stages of a deletion, and the reason this module has a type for them.
# Removing a rung from a ladder is reversible in one line and measurable the same
# week. Removing its module is neither. They are not two steps of one decision;
# they are two decisions answering to different evidence.
Stage = Literal["route", "code"]

# What a holder is: a place the rung's name appears that would have to change.
# `rung` and `consumer` are both imports, split by what imports it, because the
# two have opposite consequences and the split is the finding of this module.
Grip = Literal["ladder", "rung", "consumer", "registry", "test"]

# Which stage each grip BLOCKS, and `None` for the grips that are part of the
# deletion rather than obstacles to it. A registry line and a test file are work,
# not obstruction; counting them as blockers would make stage two unreachable for
# every rung in the book, and a check that can never pass stops being read.
#
# The load-bearing entry is `rung`. A rung that is staying cannot import a module
# that is going, so an import from one rung to another is the edge that decides
# the ORDER of a descent - and because no rung in this book imports upward, the
# order is always top-down.
BLOCKS: Mapping[Grip, Stage | None] = {
    "ladder": "route",
    "rung": "code",
    "consumer": None,
    "registry": None,
    "test": None,
}

# Level to module name. `migrate.RUNG_LEVEL` is the same mapping the other way
# up, and both are derived from the registry rather than restated.
RUNG_AT: dict[int, str] = {
    level: module.split(".")[-1] for level, module in enumerate(RUNG_MODULES)
}

# The miss rate you are willing not to notice. A removal argued on a population
# that could not have caught a rate worse than this has not been argued - it has
# been asserted with a sample size attached.
TOLERANCE = 0.05


@dataclass(frozen=True)
class Holder:
    """One thing that would break, or have to change, if this rung disappeared."""

    name: str
    grip: Grip

    @property
    def blocks(self) -> Stage | None:
        return BLOCKS[self.grip]

    def render(self) -> str:
        return f"{self.grip}:{self.name}"


def importers(module: str) -> tuple[str, ...]:
    """Every module in the package that imports this one, read from source.

    The inverse of `migrate.imports_of`, and it has to be computed rather than
    looked up, because Python records what a module imports and never what
    imports it. That asymmetry is the mechanical reason an unused rung is
    invisible: nothing about the module changes when its last caller goes away.
    """
    return tuple(
        path.stem
        for path in sorted((REPO / PACKAGE).glob("*.py"))
        if path.stem != module and module in imports_of(path.stem)
    )


def holds(
    level: int, ladders: Mapping[str, tuple[int, ...]] | None = None
) -> tuple[Holder, ...]:
    """Every edge that would break if this rung disappeared, by what it blocks.

    Ladder grips come first because they are the only ones stage one touches,
    and the only ones whose removal is a one-line edit somebody can revert from
    a phone.
    """
    module = RUNG_AT.get(level)
    if module is None:
        raise KeyError(f"level {level} is not a registered rung")
    found = [
        Holder(ask, "ladder")
        for ask, ladder in sorted((ladders or {}).items())
        if level in ladder
    ]
    found.extend(
        Holder(name, "rung" if role(f"{PACKAGE}/{name}.py") == "rung" else "consumer")
        for name in importers(module)
    )
    found.append(Holder(f"{PACKAGE}.{module}", "registry"))
    if (REPO / "tests" / f"test_{module}.py").exists():
        found.append(Holder(f"test_{module}.py", "test"))
    return tuple(found)


def unroute(
    ladders: Mapping[str, tuple[int, ...]], ask: str, level: int
) -> dict[str, tuple[int, ...]]:
    """Stage one, computed rather than hand-applied.

    Returns a whole ladder configuration rather than mutating one, so the before
    and the after can run against the same corpus in the same process. A descent
    measured by editing a module-level configuration and putting it back
    measures whatever else was reading it at the time.
    """
    if ask not in ladders:
        raise KeyError(f"no ladder for ask {ask!r}")
    updated = dict(ladders)
    updated[ask] = tuple(rung for rung in ladders[ask] if rung != level)
    return updated


def unused(ladders: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
    """Registered rungs no ladder routes to.

    Stage one is already done for these, usually by nobody in particular. Worth
    printing rather than inferring, because an unrouted rung emits no counter at
    all - it does not read zero, it reads nothing, and a dashboard built on
    decisions cannot tell those apart.
    """
    routed = {level for ladder in ladders.values() for level in ladder}
    return tuple(sorted(set(RUNG_AT) - routed))


@dataclass(frozen=True)
class Receipt:
    """The four clauses that authorize a descent, and which stage they reach.

    Named after Chapter 2's Failure Receipt and deliberately not shaped like it.
    A Failure Receipt carries one case, because it supports an existential
    claim. This carries a population, a counter that read zero across it, the
    smallest effect that population could have detected, and a step that can be
    undone - because it supports a universal one.

    `authorized` is `None` rather than `False` when no stage is reached, and the
    reason is in `decided_by`. Chapter 3's rule, arriving at the last chapter: a
    path that cannot specify a correct output refuses and says why.
    """

    level: int
    ask: str
    population: int
    answered: int
    baseline: float
    floor: float
    needed: int
    holders: tuple[Holder, ...]
    authorized: Stage | None
    decided_by: str

    @property
    def ceiling(self) -> float:
        """The largest improvement this rung could arithmetically have made."""
        return max(0.0, 1.0 - self.baseline)

    @property
    def saturated(self) -> bool:
        """True when the population cannot detect ANY effect.

        `detectable_lift` returns its own ceiling when no reachable effect size
        is detectable at this sample size, so a floor sitting on the ceiling is
        not a large number that happens to be inconvenient - it is the
        arithmetic reporting that the question was never asked.
        """
        return self.floor >= self.ceiling - 1e-3

    @property
    def blocking(self) -> tuple[Holder, ...]:
        """Holders standing between this receipt and deleting the module."""
        return tuple(h for h in self.holders if h.blocks == "code")

    @property
    def checklist(self) -> tuple[Holder, ...]:
        """Holders that are part of stage two rather than obstacles to it."""
        return tuple(h for h in self.holders if h.blocks is None)

    def render(self) -> str:
        stage = self.authorized or "nothing"
        return (
            f"L{self.level} on {self.ask}: answered {self.answered}/"
            f"{self.population}, floor {self.floor:.2f}, needs {self.needed} "
            f"-> {stage} ({self.decided_by})"
        )


def _floor(population: int, baseline: float) -> float:
    """The smallest improvement this population could have detected.

    Clamped off the endpoints, where the sizing arithmetic is undefined, and the
    clamp leans toward the FLATTERING side: a baseline of exactly 1.0 is scored
    at 0.99, which reports the smallest floor the endpoint can support. A
    removal that fails this clause fails it on numbers bent in its favor.
    """
    if population <= 0:
        return 1.0
    return detectable_lift(population, min(max(baseline, 0.01), 0.99))


def receipt(
    level: int,
    ask: str,
    reached: Iterable[int | None],
    *,
    ladders: Mapping[str, tuple[int, ...]] | None = None,
    tolerance: float = TOLERANCE,
) -> Receipt:
    """Score one rung's removal from one ask's ladder against the four clauses.

    `reached` is the rung each request in the stated population settled on, and
    `None` for a request nothing answered. It is passed in rather than computed
    here, which is the one design decision in this module worth defending:
    choosing the population is the judgment the whole receipt rests on, and a
    function that chose one for you would hide the only step that can be wrong
    on purpose.
    """
    settled = list(reached)
    population = len(settled)
    answered = sum(1 for rung in settled if rung == level)
    without = sum(1 for rung in settled if rung is not None and rung != level)
    baseline = without / population if population else 0.0
    found = holds(level, ladders)
    floor = _floor(population, baseline)
    needed = samples_needed(min(max(baseline, 0.01), 1.0 - tolerance - 0.01), tolerance)

    def scored(stage: Stage | None, why: str) -> Receipt:
        return Receipt(
            level=level,
            ask=ask,
            population=population,
            answered=answered,
            baseline=baseline,
            floor=floor,
            needed=needed,
            holders=found,
            authorized=stage,
            decided_by=why,
        )

    if population == 0:
        return scored(None, "no population: nothing was measured on this ladder")
    if not any(h.grip == "ladder" and h.name == ask for h in found):
        return scored(
            None,
            f"{ask} does not route here, so a zero counts unreachability "
            "rather than disuse",
        )
    if answered:
        return scored(
            None, f"the rung answered {answered} of {population}; it is not idle"
        )
    if population < needed:
        return scored(
            "route",
            f"reversible only: {population} requests cannot detect a "
            f"{tolerance:.0%} miss rate, and {needed} would",
        )
    held = [h for h in found if h.blocks == "code"]
    if held:
        return scored(
            "route",
            f"reversible only: held by {', '.join(h.render() for h in held)}",
        )
    return scored("code", f"answered none of {population} and no rung holds it")
