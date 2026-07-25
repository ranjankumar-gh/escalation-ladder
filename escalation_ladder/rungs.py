"""Rung registry.

Every rung module registers its entry point here, so the cross-rung cost table is generated
rather than hand-maintained. Modules that do not exist yet are skipped, which lets the
generator run from Chapter 3 onward without edits -- and means each chapter lands already
wired into the table instead of being retrofitted after the fact.
"""
from __future__ import annotations

import importlib
from typing import Callable

from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.instrument import CostLedger

RungFn = Callable[[Incident], CostLedger]

RUNGS: dict[str, RungFn] = {}

# Lowest rung first. Named by capability, never by level -- see the design rules in README.
RUNG_MODULES: tuple[str, ...] = (
    "escalation_ladder.rules",        # ch03 -- Level 0
    "escalation_ladder.classify",     # ch04 -- Level 1
    "escalation_ladder.retrieve",     # ch05 -- Level 2
    "escalation_ladder.workflow",     # ch06 -- Level 3
    "escalation_ladder.tools",        # ch07 -- Level 4
    "escalation_ladder.reasoning",    # ch08 -- Level 5
    "escalation_ladder.agent",        # ch09 -- Level 6
    "escalation_ladder.crew",         # ch10 -- Level 7
)


def register_rung(name: str) -> Callable[[RungFn], RungFn]:
    """Register a rung entry point under a display name such as "Level 3: LLM Workflow"."""

    def decorator(fn: RungFn) -> RungFn:
        RUNGS[name] = fn
        return fn

    return decorator


def load_all() -> dict[str, RungFn]:
    """Import every rung module that exists, and return the populated registry."""
    for module in RUNG_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            # Only swallow "this rung is not written yet". A bare `except ModuleNotFoundError`
            # would also swallow a missing third-party dependency inside a rung module that
            # DOES exist -- so a broken `import anthropic` in classify.py would silently drop
            # Level 1 from the published cost table instead of failing loudly.
            if exc.name == module:
                continue
            raise
    return RUNGS
