"""The framework seam.

The behaviours this chapter argues for, asserted structurally:

- a chain executes each node AT MOST ONCE, so the maximum step count is a
  property of the code rather than of the input - the Termination Test, run
  against the book's own source;
- `done` can only shorten a walk, never lengthen it;
- both implementations produce identical results on identical input, which is
  what makes Chapter 8's comparison a comparison rather than an anecdote;
- `orchestration` does not import the framework at module scope, because
  `rungs.load_all` re-raises a `ModuleNotFoundError` naming anything other than
  the rung module - so a top-level import would drop Level 5 out of the cost
  table for every reader who skipped the optional extra.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from escalation_ladder import orchestration
from escalation_ladder.orchestration import (
    LangGraphChain,
    Node,
    SequentialChain,
    chain_named,
)

def _has_langgraph() -> bool:
    try:
        import langgraph.graph  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


# Both implementations run against every behavioural test. `langgraph` is an
# optional extra, so its arm skips rather than fails when it is absent - the
# suite has to stay green for a reader who installed only what the book requires.
CHAINS = [
    pytest.param(SequentialChain(), id="sequential"),
    pytest.param(
        LangGraphChain(),
        id="langgraph",
        marks=pytest.mark.skipif(
            not _has_langgraph(), reason="langgraph extra not installed"
        ),
    ),
]


@dataclass(frozen=True)
class Counter:
    """A state that records every node that touched it."""

    visited: tuple[str, ...] = ()
    stop_after: str | None = None

    @property
    def finished(self) -> bool:
        return self.stop_after is not None and self.stop_after in self.visited


def _node(name: str) -> Node[Counter]:
    return Node(name, lambda state: replace(state, visited=state.visited + (name,)))


NODES = [_node("one"), _node("two"), _node("three")]


@pytest.mark.parametrize("chain", CHAINS)
def test_every_node_runs_once_in_order(chain) -> None:
    final = chain.walk(NODES, Counter(), done=lambda s: s.finished)
    assert final.visited == ("one", "two", "three")


@pytest.mark.parametrize("chain", CHAINS)
def test_a_node_never_runs_twice(chain) -> None:
    """The Termination Test as a unit test.

    Not a restatement of the ordering test above: this one is what would fail
    first if a later chapter quietly turned the walk into a loop, and it is the
    assertion Chapter 9 is expected to delete rather than repair.
    """
    final = chain.walk(NODES, Counter(), done=lambda s: s.finished)
    assert len(final.visited) == len(set(final.visited)) == len(NODES)


@pytest.mark.parametrize("chain", CHAINS)
def test_done_stops_the_walk_early(chain) -> None:
    final = chain.walk(NODES, Counter(stop_after="two"), done=lambda s: s.finished)
    assert final.visited == ("one", "two")


@pytest.mark.parametrize("chain", CHAINS)
def test_done_is_asked_before_the_first_node(chain) -> None:
    final = chain.walk(NODES, Counter(), done=lambda s: True)
    assert final.visited == ()


@pytest.mark.parametrize("chain", CHAINS)
def test_an_empty_chain_returns_what_it_was_given(chain) -> None:
    start = Counter(visited=("already",))
    assert chain.walk([], start, done=lambda s: s.finished) == start


@pytest.mark.skipif(not _has_langgraph(), reason="langgraph extra not installed")
@pytest.mark.parametrize(
    "start",
    [Counter(), Counter(stop_after="two"), Counter(visited=("prior",))],
)
def test_both_implementations_agree(start: Counter) -> None:
    """The claim Chapter 8's deep dive rests on, asserted rather than asserted-in-prose."""
    plain = SequentialChain().walk(NODES, start, done=lambda s: s.finished)
    graph = LangGraphChain().walk(NODES, start, done=lambda s: s.finished)
    assert plain == graph


def test_the_framework_is_not_imported_at_module_scope() -> None:
    """Read the source, not the runtime.

    A runtime check would pass whenever some other test had already imported
    `langgraph`, which is exactly when the regression matters least. Parsing the
    file cannot be fooled that way.
    """
    source = Path(orchestration.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("langgraph") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("langgraph")


def test_the_rung_registry_survives_without_the_extra() -> None:
    """Level 5 must appear in the cost table whether or not the extra is installed."""
    from escalation_ladder.rungs import load_all

    assert "Level 5: Multi-Step Reasoning" in load_all()


def test_chain_named_refuses_what_it_does_not_have() -> None:
    assert chain_named("sequential").name == "sequential"
    assert chain_named("langgraph").name == "langgraph"
    with pytest.raises(ValueError, match="unknown chain"):
        chain_named("crewai")
