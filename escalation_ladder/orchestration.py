"""The single seam between this book and a workflow framework.

Chapter 8 is where the book acquires a framework dependency, and this module is
the only file that is allowed to know about one. The reason is the same reason
`llm.py` exists: the thing behind the seam churns faster than the book does, and
a break should be one file rather than three chapters.

The contract is deliberately smaller than any framework's API. A chain walks a
FIXED sequence of nodes and stops when the sequence runs out or when `done` says
the work is finished. There is no cycle in it, no retry policy, and no way to add
a node while it is running. That is not an abstraction that happens to be small;
it is the Level 5 boundary expressed as a type.

Two implementations, both real, both exercised by the same tests:

`SequentialChain` is a `for` loop and the shipped default. `LangGraphChain` builds
the same sequence as a compiled `StateGraph`. Chapter 8's deep dive compares them
and concludes that at this rung the framework has not yet been earned, so the
default stays on the standard library and `langgraph` is an optional extra.

Chapter 9 adds `Loop` BESIDE `Chain` rather than replacing it, and the choice is
worth a sentence because the obvious reading of "Chapter 9 turns the chain into a
loop" is an edit. It is not one. `reasoning.py` still walks a chain, Chapter 11's
composite still routes to it, and Chapter 16 cannot walk a system back down to a
cheap rung whose implementation was deleted to make an expensive one look like
progress. So the two protocols sit side by side, and the difference between them
is six lines you can read in one screen - which is the entire Level 5/6 boundary,
executable.

The difference that matters is not the keyword. In `Chain`, `done` is an
optimization: delete it and every walk still terminates, because the sequence is
finite. In `Loop`, `done` IS the termination condition, and a `done` that never
returns True is a program that never returns. The bound moved out of the
structure and into a predicate, which is why Chapter 9 spends a section on what
goes inside that predicate.

That conclusion has a mechanical consequence, which is why the import of
`langgraph` sits inside the method rather than at the top of this file.
`rungs.load_all` deliberately re-raises a `ModuleNotFoundError` that names
anything other than the rung module itself, so a top-level framework import here
would drop Level 5 out of the generated cost table for every reader who did not
install the extra - and it would do it by raising from a file they never chose to
use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar

State = TypeVar("State")


@dataclass(frozen=True)
class Node(Generic[State]):
    """One step in a chain: a name for the trace, and state in, state out.

    A node returns a NEW state rather than mutating one. Chapter 6's stages were
    ordinary local variables in a function, so nothing could observe a partial
    pipeline; a chain that a framework may execute has to hand its state across a
    boundary, and the cheapest way to keep that boundary honest is to make the
    state immutable and let the walk own the only mutable reference.
    """

    name: str
    fn: Callable[[State], State]

    def __call__(self, state: State) -> State:
        return self.fn(state)


# Asked before every node, including the first. Returning True skips the rest of
# the sequence. It can only ever make a walk SHORTER, which is the property the
# Termination Test cares about.
Done = Callable[[Any], bool]


class Chain(Protocol):
    """Walk a fixed sequence of nodes.

    The maximum number of nodes executed is `len(nodes)`, for every
    implementation of this protocol, by construction. An implementation that
    could run a node twice would not be implementing this protocol - it would be
    implementing Chapter 9's.
    """

    name: str

    def walk(
        self,
        nodes: Sequence[Node[State]],
        initial: State,
        *,
        done: Done,
    ) -> State: ...


@dataclass(frozen=True)
class SequentialChain:
    """The whole of Level 5 orchestration, in six lines.

    Worth reading before the LangGraph version, because it is the baseline the
    framework has to beat. Everything a linear chain needs is here: order, an
    early exit, and a single place where state moves from one step to the next.
    """

    name: str = "sequential"

    def walk(
        self,
        nodes: Sequence[Node[State]],
        initial: State,
        *,
        done: Done,
    ) -> State:
        state = initial
        for node in nodes:
            if done(state):
                return state
            state = node(state)
        return state


@dataclass(frozen=True)
class LangGraphChain:
    """The same sequence, compiled as a `StateGraph`.

    Byte for byte the honest comparison: same nodes, same order, same early exit,
    same result. What it costs is visible in the code below - a declared state
    schema, a boxed channel, an explicit START edge, and a conditional edge per
    node to express the early exit that `for` expressed with `return`.

    The box is the part worth understanding rather than skipping. LangGraph's
    state is a set of named channels with reducers, and merging concurrent writes
    to those channels is most of what it is for. A linear chain has no concurrent
    writes, so there is nothing to merge, and the generic seam here declares one
    channel holding the domain object. That is not the framework being used
    badly - it is a linear chain having no use for the feature. Chapter 10 fans
    out, and that is where channels start doing work.
    """

    name: str = "langgraph"

    def walk(
        self,
        nodes: Sequence[Node[State]],
        initial: State,
        *,
        done: Done,
    ) -> State:
        # Imported here, not at module scope. See this module's docstring: a
        # top-level import would make an optional extra a hard requirement of the
        # rung registry.
        from langgraph.graph import END, START, StateGraph

        if not nodes:
            return initial

        graph: Any = StateGraph(dict)
        for node in nodes:
            graph.add_node(
                node.name,
                lambda box, node=node: {"state": node(box["state"])},
            )

        def branch_to(target: str) -> Callable[[dict], str]:
            return lambda box: END if done(box["state"]) else target

        graph.add_conditional_edges(
            START, branch_to(nodes[0].name), [nodes[0].name, END]
        )
        for current, following in zip(nodes, nodes[1:]):
            graph.add_conditional_edges(
                current.name, branch_to(following.name), [following.name, END]
            )
        graph.add_edge(nodes[-1].name, END)

        compiled = graph.compile()
        return compiled.invoke({"state": initial})["state"]


DEFAULT_CHAIN: Chain = SequentialChain()


def chain_named(name: str) -> Chain:
    """Look up a chain implementation by name, for scripts that compare them.

    Deliberately not a plugin registry. Two implementations are two, and a
    registry here would be the same reflex this book spends sixteen chapters
    arguing against, applied to its own source.
    """
    if name == "sequential":
        return SequentialChain()
    if name == "langgraph":
        return LangGraphChain()
    raise ValueError(f"unknown chain {name!r}; known: sequential, langgraph")


class Loop(Protocol):
    """Run ONE node until `done` says to stop.

    There is no `nodes` argument and that absence is the rung. A chain took a
    sequence because the sequence was the design; a loop takes a single node
    because the design is now "do the thing again", and how many times is a
    question about the run rather than about the code.

    No implementation of this protocol can state its maximum in advance. That is
    not a gap to be filled by a better implementation - it is the definition, and
    `Budget` in `agent.py` is where a caller buys the maximum back at a price.
    """

    name: str

    def run(
        self,
        node: Node[State],
        initial: State,
        *,
        done: Done,
    ) -> State: ...


@dataclass(frozen=True)
class WhileLoop:
    """Level 6 orchestration, in four lines. Compare with `SequentialChain`.

    The diff against the chain is one line of control flow, and every property
    Chapter 8 relied on is in that line. `for node in nodes` could not run a node
    twice and could not run longer than the sequence. `while not done(state)` can
    do both, and will, and there is no argument you can pass to this method that
    changes it.
    """

    name: str = "while"

    def run(
        self,
        node: Node[State],
        initial: State,
        *,
        done: Done,
    ) -> State:
        state = initial
        while not done(state):
            state = node(state)
        return state


@dataclass(frozen=True)
class LangGraphLoop:
    """The same loop as a compiled `StateGraph`, with an edge back to itself.

    This is the version Chapter 8 said it would collect on. A chain gave the
    framework's checkpointer nothing to do, because there was no run long enough
    to be worth resuming; a loop that reads live telemetry for several minutes
    has a genuine mid-run state, and `checkpointer` is how you get it back after
    a crash.

    `thread` is required when a checkpointer is present - it is the key the saved
    state is filed under, and passing the incident id makes a resume the same
    call as the original with the same argument.

    `resume` is NOT a convenience flag, and leaving it off is the trap. Invoking
    a compiled graph with an input applies that input to the channel, so calling
    the same graph again with the same starting state overwrites everything the
    checkpointer restored and quietly runs the whole investigation a second time.
    Resuming means invoking with `None` and letting the saved state stand. Both
    versions return the same final answer, which is why the mistake survives a
    test that only checks the answer - the difference is visible in the bill and
    nowhere else.

    `recursion_limit` is LangGraph's own circuit breaker and defaults to 25. It
    is set high here on purpose: the bound this book cares about lives in
    `agent.Budget`, and leaving a second, invisible cap in the framework would
    mean a run that stopped for a reason nothing in the ledger could explain.
    """

    name: str = "langgraph-loop"
    checkpointer: Any | None = None
    thread: str = "default"
    resume: bool = False
    recursion_limit: int = 500

    def run(
        self,
        node: Node[State],
        initial: State,
        *,
        done: Done,
    ) -> State:
        # Imported here, not at module scope, for the reason in this module's
        # docstring: a top-level import would drop two rungs out of the generated
        # cost table for every reader who skipped the optional extra.
        from langgraph.graph import END, START, StateGraph

        graph: Any = StateGraph(dict)
        graph.add_node(node.name, lambda box: {"state": node(box["state"])})

        def branch(box: dict) -> str:
            return END if done(box["state"]) else node.name

        graph.add_conditional_edges(START, branch, [node.name, END])
        # The back edge. One line, and it is the whole architecture: the node's
        # only outgoing route is a conditional that can return to itself.
        graph.add_conditional_edges(node.name, branch, [node.name, END])

        compiled = graph.compile(checkpointer=self.checkpointer)
        config: dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if self.checkpointer is not None:
            config["configurable"] = {"thread_id": self.thread}
        if self.resume:
            if self.checkpointer is None:
                raise ValueError("resume=True needs a checkpointer to resume from")
            # None, not the initial state. See the class docstring.
            return compiled.invoke(None, config)["state"]
        return compiled.invoke({"state": initial}, config)["state"]


DEFAULT_LOOP: Loop = WhileLoop()


def loop_named(name: str) -> Loop:
    """Look up a loop implementation by name, for scripts that compare them."""
    if name == "while":
        return WhileLoop()
    if name == "langgraph":
        return LangGraphLoop()
    raise ValueError(f"unknown loop {name!r}; known: while, langgraph")


# Merge N branch results back into one state. The signature is the argument: a
# fan-out is not finished when its branches are, and somebody has to say what two
# disagreeing branches produce. A framework that supplies this for you has
# supplied a default answer to a domain question.
Merge = Callable[[Any, Sequence[Any]], Any]


class Fanout(Protocol):
    """Run several nodes from ONE state and merge what comes back.

    The third protocol in this file, added in Chapter 10 for the same reason
    Chapter 9 added the second: because the two that already exist cannot express
    it. `Chain` walks a sequence and threads state through it, so branch two
    would see branch one's writes. `Loop` runs one node repeatedly. Neither can
    hand the same starting state to two nodes and be handed two states back, and
    that shape - concurrent writes to one state - is precisely what Chapter 8
    said LangGraph's channels exist for and a linear chain had no use for.

    Chapter 10 builds this, exercises it, and does NOT put it on the default
    path, because the crew that Chapter 9's receipt earns is three sequential
    roles and a review cannot begin before the investigation it reviews. The
    protocol is here so that the deep dive's argument is made against running
    code rather than against a hypothetical.
    """

    name: str

    def spread(
        self,
        nodes: Sequence[Node[State]],
        initial: State,
        *,
        merge: Merge,
    ) -> State: ...


@dataclass(frozen=True)
class ThreadFanout:
    """Branches in threads, merged by a function the caller wrote.

    Threads rather than processes because every branch here is a model call: the
    work is entirely I/O and the interpreter lock is released for all of it.
    Threads rather than `asyncio` because the seam in `llm.py` is synchronous,
    and making it async to parallelize two calls would be a rewrite of every rung
    to buy what one `ThreadPoolExecutor` already provides.

    `merge` is required and has no default, for the same reason
    `ToolSpec.consequence` has none in Chapter 7. A default merge is a silent
    answer to "what happens when the branches disagree?", and disagreement is the
    interesting case rather than the edge case.
    """

    name: str = "threads"

    def spread(
        self,
        nodes: Sequence[Node[State]],
        initial: State,
        *,
        merge: Merge,
    ) -> State:
        from concurrent.futures import ThreadPoolExecutor

        if not nodes:
            return initial
        with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            branches = list(pool.map(lambda node: node(initial), nodes))
        return merge(initial, branches)


@dataclass(frozen=True)
class LangGraphFanout:
    """The same fan-out as a compiled graph, with the merge as a channel reducer.

    This is the shape Chapter 8 named as the receipt that would earn the
    framework: several nodes writing one channel in a single superstep, with an
    `Annotated` reducer resolving the writes. Read it beside `ThreadFanout` and
    the deep dive's comparison is visible without prose - the reducer is the same
    function, moved from a call site into a type annotation, and what the
    framework adds is that it becomes impossible to forget to call it.

    That is a real property, and it is worth exactly as much as the number of
    places you would have forgotten.
    """

    name: str = "langgraph-fanout"

    def spread(
        self,
        nodes: Sequence[Node[State]],
        initial: State,
        *,
        merge: Merge,
    ) -> State:
        # Imported here, not at module scope, for the reason in this module's
        # docstring: a top-level import would drop three rungs out of the
        # generated cost table for every reader who skipped the optional extra.
        from typing import Annotated, TypedDict

        from langgraph.graph import END, START, StateGraph

        if not nodes:
            return initial

        def collect(left: list[Any], right: list[Any]) -> list[Any]:
            return left + right

        Box = TypedDict("Box", {"seed": Any, "branches": Annotated[list, collect]})

        graph: Any = StateGraph(Box)
        for node in nodes:
            graph.add_node(
                node.name,
                lambda box, node=node: {"branches": [node(box["seed"])]},
            )
            graph.add_edge(START, node.name)
            graph.add_edge(node.name, END)

        compiled = graph.compile()
        final = compiled.invoke({"seed": initial, "branches": []})
        return merge(initial, final["branches"])


DEFAULT_FANOUT: Fanout = ThreadFanout()


def fanout_named(name: str) -> Fanout:
    """Look up a fan-out implementation by name, for scripts that compare them."""
    if name == "threads":
        return ThreadFanout()
    if name == "langgraph":
        return LangGraphFanout()
    raise ValueError(f"unknown fanout {name!r}; known: threads, langgraph")
