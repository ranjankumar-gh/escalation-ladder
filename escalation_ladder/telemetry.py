"""One event record per decision, and every counter derived from it.

Eleven chapters have measured cost through `instrument.CostLedger`, and cost is
the half of observability the vendor gives you for nothing. A `Measurement`
answers "what did this consume". Every question asked at 3am is "which decision
was wrong", and no rung in this book emitted an answer to that until here.

Ten chapters left this module a list of counters -- refusals by reason, by
origin, by role, by step; rounds and steps and turns as distributions; cost
grouped by outcome, by role, by ask, by surface. They are not thirty
instruments. They are twenty-two group-bys over one record, which is the design
this module argues for: emit once, derive many.

TWO ADAPTERS, NOT ONE, and the split is the finding rather than a structural
convenience. `witness` reconstructs the decision tree from what a rung already
returns, because every result object in this book has carried `decided_by` since
Chapter 3. `TracingCompleter` wraps the seam and supplies what no return value
knows -- which prompt version ran and which draw answered it. A decision is not
a call: Level 0 decides and never calls, so a completer wrapper cannot see the
cheapest rung at all, and a prompt version is a property of the call that no
decision carries. Neither adapter alone is sufficient, and together they need no
edit to any rung module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping

from escalation_ladder.instrument import CostLedger
from escalation_ladder.llm import (
    Completer,
    Completion,
    Execute,
    ToolResult,
    ToolRun,
    ToolSpec,
    prompt_key,
)

# What happened at one decision point. Bounded on purpose: this is a counter
# dimension, and a dimension whose value set grows with traffic is not a
# dimension, it is a log line with a metrics bill attached.
Outcome = Literal["answered", "refused", "failed", "unknown"]

# The value a field takes when the seam that produced the record could not know
# it. `from_ledger` fills most of a record with this, which is what makes the
# comparison in `instrumentation_level` mechanical rather than rhetorical.
UNKNOWN = "unknown"

# Safe to group by. Every one has a value set fixed by the code rather than by
# the request: eight levels, a site vocabulary the modules define, four
# outcomes, the kinds below, and a small integer.
DIMENSIONS: tuple[str, ...] = ("level", "site", "outcome", "kind", "index")


@dataclass(frozen=True)
class Event:
    """One decision, its bounded dimensions, and its unbounded payload.

    The split between `kind` and `reason` is the whole cardinality argument in
    one dataclass. `kind` is a closed set you may put in a counter. `reason` is
    the sentence a human reads once, in one trace, having already found it by
    grouping on `kind` -- and it must never reach a metrics backend as a label.
    Six incidents in this repository produce 148 distinct `reason` strings
    against 22 kinds, and only one of those two numbers grows with traffic.
    """

    incident_id: str
    level: int
    site: str
    outcome: Outcome
    kind: str
    reason: str = ""
    index: int = 0
    parent: str | None = None
    # High-cardinality payload, and the reason INC-1046 opens this chapter: the
    # difference between a correct Level 4 investigation and that one is a
    # single tool argument, and no token counter records a word.
    fields: tuple[tuple[str, str], ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    # Vendor requests, which are NOT `model_calls`. Chapter 8 named this trap
    # and Chapter 12 asked for the counter: one metered call through `invoke`
    # covers two requests, so a rate limit is consumed at twice the rate the
    # ledger reports. Only the seam knows, which is why it is filled here and
    # not by `witness`.
    requests: int = 0
    prompt_version: str = ""
    draw: str = ""

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def key(self, *dimensions: str) -> tuple[Any, ...]:
        """The group-by key for the named dimensions, refusing unsafe ones."""
        unsafe = [d for d in dimensions if d not in DIMENSIONS]
        if unsafe:
            raise ValueError(
                f"{', '.join(unsafe)} is not a bounded dimension; "
                f"group by one of {', '.join(DIMENSIONS)}"
            )
        return tuple(getattr(self, dimension) for dimension in dimensions)

    def render(self) -> str:
        pairs = " ".join(f"{k}={v}" for k, v in self.fields)
        cost = f"{self.tokens:>6}t" if self.tokens else "      -"
        return (
            f"L{self.level} {self.site:<26} {self.outcome:<8} {self.kind:<14} "
            f"{cost}  {pairs or self.reason[:56]}"
        )


@dataclass(frozen=True)
class Trace:
    """Every event one request produced, and the queries you ask of them.

    Deliberately not a tree structure. `parent` is a site name on each event, so
    a trace is a flat sequence that can be printed as a tree, shipped as lines,
    and grouped without a traversal. Every counter this book owes is a `by` call
    on this object, which is the argument: the counters were never the
    instrumentation, they were always the query.
    """

    incident_id: str
    events: tuple[Event, ...] = ()
    ask: str | None = None

    def __add__(self, other: "Trace") -> "Trace":
        return Trace(
            self.incident_id, self.events + other.events, self.ask or other.ask
        )

    def by(self, *dimensions: str) -> dict[tuple[Any, ...], tuple[Event, ...]]:
        grouped: dict[tuple[Any, ...], list[Event]] = {}
        for event in self.events:
            grouped.setdefault(event.key(*dimensions), []).append(event)
        return {key: tuple(value) for key, value in grouped.items()}

    def where(self, **equals: Any) -> "Trace":
        kept = tuple(
            event
            for event in self.events
            if all(getattr(event, name) == value for name, value in equals.items())
        )
        return Trace(self.incident_id, kept, self.ask)

    def sites(self, prefix: str) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.site.startswith(prefix))

    @property
    def tokens(self) -> int:
        return sum(event.tokens for event in self.events)

    @property
    def model_calls(self) -> int:
        return sum(event.model_calls for event in self.events)

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(sorted({event.level for event in self.events}))

    @property
    def terminal(self) -> Event | None:
        """The event that decided the request, which is the last root."""
        roots = [e for e in self.events if e.parent is None]
        return roots[-1] if roots else None

    def spans(self) -> str:
        """The trace as an indented tree, for a human who already knows why."""
        children: dict[str | None, list[Event]] = {}
        for event in self.events:
            children.setdefault(event.parent, []).append(event)
        lines: list[str] = []
        for root in children.get(None, ()):
            lines.append(root.render())
            for child in children.get(root.site, ()):
                lines.append("  " + child.render())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reason kinds
# ---------------------------------------------------------------------------

# Every rung in this book explains itself in prose, which is right for a human
# and useless for a counter. These patterns are the translation, and writing
# them by hand is not a defect of the design -- it is the price of `decided_by`
# being readable, paid once here rather than by every consumer.
#
# Ordered: the first match wins, so specific patterns precede general ones.
_KINDS: tuple[tuple[str, str], ...] = (
    (r"^no recording for (prompt|answer)", "unrecorded"),
    (r"^(api error|no credentials|schema violated)", "seam_failure"),
    (r"^the model (declined|asked for no tools)", "declined"),
    (r"^one round of tools was not enough", "wants_more"),
    (r"^wants next:", "wants_more"),
    (r"^asked for another round", "wants_more"),
    (r"^the report is prose", "unparsed"),
    (r"^no rule|^unknown service|^no threshold", "no_rule"),
    (r"> [\d.]+ on ", "rule_fired"),
    (r"^grounded in |^cites ", "grounded"),
    (r"^no passage|^nothing retrieved|cleared the floor", "no_evidence"),
    (r"^model classification", "model_claim"),
    (r"showed |^step \d+ of \d+", "tool_evidence"),
    (r"^the telemetry does not explain", "no_explanation"),
    (r"outside these tools|^Pull |tracing", "unreachable"),
    (r"oncall SEV\d|^\S+ SEV\d", "routed"),
)

KINDS: tuple[str, ...] = tuple(dict.fromkeys(kind for _, kind in _KINDS)) + (
    "tool_error",
    "degraded",
    "bounded",
    "gave_up",
    "settled",
    "broke",
    "handoff",
    "blocked",
    "billed",
    "early_rung",
    "last_rung",
    "narrative",
    UNKNOWN,
)

# Kinds that mean the request did not complete for a reason outside the rung's
# own judgment. Grouped separately because Chapter 13's rule applies unchanged:
# an unmeasured case is not a failing one.
BROKEN: frozenset[str] = frozenset({"unrecorded", "seam_failure"})


def kind_of(reason: str) -> str:
    """Map one free-text `decided_by` onto the closed set a counter may hold.

    `narrative` is the honest bucket rather than the failure case. A model
    writing a sentence about an incident produces text no pattern will ever
    catch, and a classifier that pretended otherwise would put the request
    itself into a metric label. What matters is that the bucket is bounded and
    that its share is visible: a `narrative` rate climbing across releases is
    how you learn your reason vocabulary has drifted away from your patterns.
    """
    text = reason.strip()
    if not text:
        return UNKNOWN
    for pattern, kind in _KINDS:
        if re.search(pattern, text):
            return kind
    return "narrative"


def _outcome(ok: bool, failed: bool = False) -> Outcome:
    if failed:
        return "failed"
    return "answered" if ok else "refused"


def _event(
    incident_id: str, level: int, site: str, reason: str, ok: bool, **rest: Any
) -> Event:
    kind = kind_of(reason)
    return Event(
        incident_id=incident_id,
        level=level,
        site=site,
        outcome=_outcome(ok, failed=kind in BROKEN),
        kind=kind,
        reason=reason,
        **rest,
    )


# ---------------------------------------------------------------------------
# Adapter one: the decisions, reconstructed from what a rung already returns
# ---------------------------------------------------------------------------

# This module imports no rung. Every adapter below reads public attributes the
# rung modules have returned since the chapter that built them, so nothing under
# `escalation_ladder/` was edited to make a request observable -- and a rung this
# module has never heard of becomes observable by returning the same shapes.


def _decision_of(result: Any) -> Any:
    return getattr(result, "decision", result)


def _incident_of(result: Any) -> str:
    decision = _decision_of(result)
    return str(getattr(decision, "incident_id", getattr(result, "incident_id", "?")))


def _last_rung(outcome: Any, ladders: Mapping[str, tuple[int, ...]] | None) -> bool | None:
    """Answered at the final rung of its own ladder -- Chapter 11's counter.

    THE ONE COUNTER THAT NEEDS A FACT FROM OUTSIDE THE REQUEST, and it took
    getting it wrong to see why. The first version compared `level_reached`
    against the last rung ATTEMPTED, which is trivially true for every answered
    request, because a cascade stops at the first rung that answers. It
    reported every request as a last-rung request and looked entirely
    plausible.

    The last rung of an ask's ladder is a property of the CONFIGURATION, and no
    result object carries its own configuration. So this is the single place
    the adapter takes an argument rather than reading a return value, and
    without it the counter is `UNKNOWN` rather than quietly wrong. Chapter 11
    fixed the decision rule and Chapter 16 owes the action; this emits it and
    does nothing else, because a counter that acts on itself is a policy hiding
    in a metric.
    """
    ask = getattr(outcome, "ask", None)
    reached = getattr(outcome, "level_reached", None)
    if ladders is None or ask not in ladders or reached is None:
        return None
    return reached == ladders[ask][-1]


def _tool_events(incident_id: str, level: int, result: Any, parent: str) -> list[Event]:
    """One event per tool call, carrying the arguments the model chose.

    A counter recording "Level 4 made six tool calls" is true of a correct
    investigation and of INC-1046 alike. The arguments are what separates them,
    and they are payload rather than a dimension for the obvious reason.
    """
    calls = tuple(getattr(result, "calls", ()) or ())
    results = tuple(getattr(result, "results", ()) or ())
    by_id = {r.call_id: r for r in results if isinstance(r, ToolResult)}
    events: list[Event] = []
    for index, call in enumerate(calls):
        outcome = by_id.get(getattr(call, "call_id", ""))
        errored = bool(outcome is not None and outcome.is_error)
        empty = bool(
            outcome is not None and outcome.content.startswith("no lines matching")
        )
        kind = "tool_error" if errored else ("no_evidence" if empty else "tool_evidence")
        events.append(
            Event(
                incident_id=incident_id,
                level=level,
                site=f"tool.{call.name}",
                outcome="failed" if errored else _outcome(not empty),
                kind=kind,
                reason=(outcome.content[:200] if outcome is not None else ""),
                index=index,
                parent=parent,
                fields=tuple((k, str(v)) for k, v in sorted(call.arguments.items())),
            )
        )
    return events


def _ledger_events(
    incident_id: str, level: int, ledger: CostLedger | None, parent: str
) -> list[Event]:
    if ledger is None:
        return []
    return [
        Event(
            incident_id=incident_id,
            level=level,
            site=f"call.{measurement.label}",
            outcome="answered" if measurement.model_calls else "unknown",
            kind="billed" if measurement.model_calls else UNKNOWN,
            index=index,
            parent=parent,
            input_tokens=measurement.input_tokens,
            output_tokens=measurement.output_tokens,
            model_calls=measurement.model_calls,
        )
        for index, measurement in enumerate(ledger.measurements)
    ]


def _composite(
    result: Any, ask: str | None, ladders: Mapping[str, tuple[int, ...]] | None
) -> Trace:
    """Chapter 11's cascade: one root per rung attempted, plus the outcome."""
    incident_id = _incident_of(result)
    events: list[Event] = []
    for attempt in getattr(result, "attempts", ()):
        events.append(
            _event(
                incident_id,
                attempt.level,
                "rung",
                attempt.decided_by,
                attempt.answered,
                input_tokens=attempt.input_tokens,
                output_tokens=attempt.output_tokens,
                model_calls=attempt.model_calls,
            )
        )
    events.append(
        Event(
            incident_id=incident_id,
            level=int(getattr(result, "level_reached", None) or 0),
            site="composite",
            outcome=_outcome(bool(getattr(result, "answered", False))),
            kind={True: "last_rung", False: "early_rung", None: UNKNOWN}[
                _last_rung(result, ladders)
            ],
            reason=str(_decision_of(result).decided_by),
            index=int(getattr(result, "climbed", 0)),
            fields=(("ask", str(getattr(result, "ask", "") or "")),),
        )
    )
    return Trace(incident_id, tuple(events), getattr(result, "ask", ask))


def witness(
    result: Any,
    *,
    level: int = 0,
    ask: str | None = None,
    ladders: Mapping[str, tuple[int, ...]] | None = None,
) -> Trace:
    """Reconstruct one request's decision tree from one rung's return value.

    The dispatch is on the fields present rather than on the type, which is not
    laziness: it is the statement that this adapter depends on the shape a rung
    publishes and on nothing else. A composite recurses to one root per rung,
    because a cascade is a request that visited several rungs and a trace that
    flattened them would delete the only field Chapter 11 added.
    """
    attempts = tuple(getattr(result, "attempts", ()) or ())
    if attempts and all(hasattr(a, "level") for a in attempts):
        return _composite(result, ask, ladders)

    incident_id = _incident_of(result)
    decision = _decision_of(result)
    routed = bool(getattr(result, "routed", getattr(decision, "routed", False)))
    events: list[Event] = [
        _event(incident_id, level, "rung", str(getattr(decision, "decided_by", "")), routed)
    ]

    # Level 2: the Substitution Test, made observable. A missing citation is a
    # retrieval failure and a present one under an ungrounded answer is a
    # generation failure, and the two prompt opposite actions.
    if hasattr(result, "citation") and not hasattr(result, "calls"):
        cited = getattr(result, "citation", None)
        events.append(
            Event(
                incident_id=incident_id,
                level=level,
                site="retrieve",
                outcome=_outcome(cited is not None),
                kind="grounded" if cited is not None else "no_evidence",
                reason=str(cited or "no passage cleared the floor"),
                parent="rung",
                fields=(("citation", str(cited or "")),),
            )
        )

    # Level 3: five stages, each carrying the stage its input came from.
    for index, stage in enumerate(getattr(result, "trace", ()) or ()):
        events.append(
            _event(
                incident_id,
                level,
                f"stage.{stage.name}",
                stage.detail,
                stage.ok,
                index=index,
                parent="rung",
                fields=(("input_from", str(stage.input_from or "")),),
            )
        )
    if getattr(result, "degraded", False):
        origin = str(getattr(result, "origin", "") or "")
        events.append(
            Event(
                incident_id=incident_id,
                level=level,
                site="degrade",
                outcome="refused",
                kind="degraded",
                reason=origin,
                parent="rung",
                fields=(("origin", origin),),
            )
        )

    # Level 5: steps, and whether each step's finding survived the check.
    for index, step in enumerate(getattr(result, "steps", ()) or ()):
        events.append(
            _event(
                incident_id,
                level,
                f"step.{step.name}",
                step.detail,
                step.ok and step.finding is not None,
                index=index,
                parent="rung",
                fields=(("kept", str(step.finding is not None)),),
            )
        )

    # Level 6: rounds, and the bound that ended them.
    rounds = tuple(getattr(result, "attempts", ()) or ())
    for round_ in rounds:
        events.append(
            _event(
                incident_id,
                level,
                "round",
                str(round_.detail),
                bool(round_.ok),
                index=int(round_.index),
                parent="rung",
            )
        )
    stopped_by = getattr(result, "stopped_by", None)
    gave_up = getattr(result, "gave_up", None)
    if rounds or stopped_by is not None or gave_up is not None:
        # UNCONDITIONAL, and the first draft of this adapter was not. It emitted
        # a stop event only when `stopped_by` or `gave_up` was set, which is
        # every path the loop takes on purpose - and none of the paths it takes
        # because something underneath it broke. That measured as an
        # Instrumentation Level of 5 on a system whose top rung is 7, and the
        # missing rung was the loop. An absent field is not an observation: a
        # `stopped_by` of None means "settled by itself" and reads in the record
        # exactly like a field nobody filled in.
        broke = bool(rounds) and not rounds[-1].ok
        kind = (
            "bounded"
            if stopped_by is not None
            else "gave_up"
            if gave_up is not None
            else "broke"
            if broke
            else "settled"
        )
        events.append(
            Event(
                incident_id=incident_id,
                level=level,
                site="stop",
                outcome="answered" if kind == "settled" else "refused",
                kind=kind,
                reason=str(stopped_by or gave_up or (rounds[-1].detail if rounds else "")),
                index=len(rounds),
                parent="rung",
                fields=(("bound", str(stopped_by or kind)),),
            )
        )

    # Level 7: handoffs as first-class spans, and the role that blocked.
    for index, handoff in enumerate(getattr(result, "handoffs", ()) or ()):
        events.append(
            Event(
                incident_id=incident_id,
                level=level,
                site=f"handoff.{handoff.sender}",
                outcome="answered",
                kind="handoff",
                reason=handoff.carried[:200],
                index=index,
                parent="rung",
                fields=(
                    ("receiver", str(handoff.receiver)),
                    ("characters", str(handoff.characters)),
                ),
            )
        )
    blocked_at = getattr(result, "blocked_at", None)
    if blocked_at is not None:
        events.append(
            Event(
                incident_id=incident_id,
                level=level,
                site=f"role.{blocked_at}",
                outcome="refused",
                kind="blocked",
                reason=str(getattr(decision, "decided_by", "")),
                parent="rung",
                fields=(("role", str(blocked_at)),),
            )
        )

    events.extend(_tool_events(incident_id, level, result, "rung"))
    events.extend(
        _ledger_events(incident_id, level, getattr(result, "ledger", None), "rung")
    )
    return Trace(incident_id, tuple(events), ask)


def from_ledger(ledger: CostLedger, *, incident_id: str, level: int) -> Trace:
    """The seam this book has had for eleven chapters, expressed as a trace.

    Not a straw man. This is exactly what `CostLedger` holds, promoted into the
    same shape as everything above so the two can be asked identical questions.
    Every dimension it cannot fill is `UNKNOWN`, which is not a modelling
    convenience: it is the reason a cost dashboard answers no debugging
    question, made mechanical enough for a test to assert.
    """
    return Trace(incident_id, tuple(_ledger_events(incident_id, level, ledger, None)))


# ---------------------------------------------------------------------------
# Adapter two: the calls, from the seam that already sees every prompt
# ---------------------------------------------------------------------------


@dataclass
class TracingCompleter:
    """A `Completer` that records which prompt version ran and which draw answered.

    Chapter 13's three counters are here rather than in `witness` for a reason
    that took building both to see: they are properties of the CALL, and no
    result object in this book carries them. A rung returns what it decided, not
    what it asked. `MeteredCompleter` established the pattern in Chapter 7 -
    wrap the seam, edit no caller - and this composes with it rather than
    replacing it, because cost and provenance are two different questions about
    the same request.

    `draw` is a hash of the response rather than an index into a recording file,
    which is what makes it mean the same thing in production as it does in a
    replay: the identity of the sample that was scored. Chapter 13 measured what
    happens when you cannot name it - three of six pipelines became zero of six,
    three stages downstream, with no prompt and no code changed.
    """

    inner: Completer
    events: list[Event]
    incident_id: str
    level: int
    site: str = "prompt"

    def _record(
        self, system: str, user: str, body: str, failed: str | None, requests: int
    ) -> None:
        self.events.append(
            Event(
                incident_id=self.incident_id,
                level=self.level,
                site=self.site,
                outcome=("refused" if kind_of(failed) == "wants_more" else "failed")
                if failed
                else "answered",
                kind=kind_of(failed or "") if failed else "billed",
                reason=failed or "",
                index=len(self.events),
                parent="rung",
                requests=requests,
                prompt_version=prompt_key(system=system, user=user),
                draw=prompt_key(system="", user=body),
            )
        )

    def parse(
        self, *, system: str, user: str, schema: Any, effort: str = "low"
    ) -> Completion[Any]:
        result = self.inner.parse(
            system=system, user=user, schema=schema, effort=effort
        )
        body = result.parsed.model_dump_json() if result.parsed is not None else ""
        self._record(system, user, body, result.failed, requests=1)
        return result

    def invoke(
        self,
        *,
        system: str,
        user: str,
        tools: tuple[ToolSpec, ...],
        execute: Execute,
        schema: Any,
        effort: str = "low",
    ) -> ToolRun[Any]:
        result = self.inner.invoke(
            system=system,
            user=user,
            tools=tools,
            execute=execute,
            schema=schema,
            effort=effort,
        )
        body = result.parsed.model_dump_json() if result.parsed is not None else ""
        # Two, unless the first request came back with no tools to run and the
        # round ended there. Counting one would under-report the rung that
        # consumes rate limit fastest.
        asked = 1 if result.failed and not result.calls else 2
        self._record(system, user, body, result.failed, requests=asked)
        return result


# ---------------------------------------------------------------------------
# The counters ten chapters asked for, as queries
# ---------------------------------------------------------------------------


def _amount(event: Event, measure: str, trace: Trace | None = None) -> int:
    if measure == "tokens":
        return trace.tokens if trace is not None else event.tokens
    if measure == "requests":
        return event.requests
    return 1


@dataclass(frozen=True)
class Owed:
    """One counter an earlier chapter specified, expressed as a group-by.

    `chapter` is not decoration. It is the audit trail that this list came from
    obligations written down before the record existed, rather than from
    whatever this chapter's design happened to make easy.
    """

    chapter: int
    name: str
    by: tuple[str, ...]
    where: Mapping[str, Any] = field(default_factory=dict)
    scope: Literal["event", "request"] = "event"
    measure: Literal["count", "tokens", "requests"] = "count"
    # Site families, because a site is a namespace: `step.` is Chapter 8's five
    # steps and `call.` is the billing seam underneath all of them. A counter
    # that grouped both together would report a distribution over two different
    # things and read as though the rung had seventeen steps.
    prefix: str = ""
    exclude: tuple[str, ...] = ()

    def of(self, traces: Iterable[Trace]) -> dict[tuple[Any, ...], int]:
        """Compute this counter over a corpus of traces."""
        tallied: dict[tuple[Any, ...], int] = {}
        for trace in traces:
            selected = trace.where(**dict(self.where))
            if self.prefix or self.exclude:
                selected = Trace(
                    selected.incident_id,
                    tuple(
                        e
                        for e in selected.events
                        if e.site.startswith(self.prefix)
                        and not any(e.site.startswith(p) for p in self.exclude)
                    ),
                    selected.ask,
                )
            if self.scope == "request":
                terminal = selected.terminal
                if terminal is None:
                    continue
                amount = _amount(terminal, self.measure, trace)
                tallied[terminal.key(*self.by)] = (
                    tallied.get(terminal.key(*self.by), 0) + amount
                )
                continue
            for event in selected.events:
                amount = _amount(event, self.measure)
                tallied[event.key(*self.by)] = tallied.get(event.key(*self.by), 0) + amount
        return tallied


# Every counter Chapters 4 through 13 left this one, in the order the chapters
# were written. Nothing here is an instrument. They are twenty-one group-bys
# over one record, and the ones that read as duplicates - refusals by reason at
# four different rungs - collapse into a single query with a different filter,
# which is the whole argument.
OWED: tuple[Owed, ...] = (
    Owed(4, "refusals by reason", ("kind",), {"outcome": "refused", "site": "rung"}),
    Owed(4, "decisions by kind", ("kind",), {"level": 1, "site": "rung"}),
    Owed(5, "retrieval against generation", ("kind",), {"site": "retrieve"}),
    Owed(6, "degraded pages by origin", ("kind",), {"site": "degrade"}),
    Owed(6, "stage outcomes by site", ("site", "outcome"), prefix="stage."),
    Owed(7, "tool errors by tool", ("site",), {"kind": "tool_error"}),
    Owed(7, "second rounds asked for", ("level",), {"kind": "wants_more"}),
    Owed(7, "tool calls by tool", ("site", "outcome"), prefix="tool."),
    Owed(8, "step outcomes by step", ("site", "outcome"), prefix="step."),
    Owed(8, "steps reached, as a distribution", ("index",), prefix="step."),
    Owed(9, "outcome by bucket", ("kind",), {"site": "stop"}),
    Owed(9, "rounds reached, as a distribution", ("index",), {"site": "round"}),
    Owed(9, "cost grouped by outcome", ("outcome",), {"level": 6}, "request", "tokens"),
    Owed(10, "refusals by role", ("site",), {"kind": "blocked"}),
    Owed(10, "handoffs as spans", ("index",), {"kind": "handoff"}),
    Owed(11, "level reached, as a distribution", ("level",), {"site": "composite"}),
    Owed(11, "rungs climbed per request", ("index",), {"site": "composite"}),
    Owed(11, "requests answered at the last rung", ("kind",), {"site": "composite"}),
    Owed(12, "cost by call index", ("index",), {"kind": "billed"}, "event", "tokens"),
    Owed(12, "requests per turn", ("index",), {"site": "prompt"}, "event", "requests"),
    Owed(13, "cost grouped by surface", ("level", "site"), {}, "event", "tokens",
         exclude=("call.", "prompt", "tool.")),
    Owed(13, "draws by prompt version", ("site",), {"site": "prompt"}),
)


def counters(traces: Iterable[Trace], owed: Iterable[Owed] = OWED) -> str:
    """The whole owed set, computed and rendered as one table."""
    kept = list(traces)
    lines = ["| Ch | Counter | Grouped by | Distinct | Total |",
             "|---:|---|---|---:|---:|"]
    for entry in owed:
        tallied = entry.of(kept)
        lines.append(
            f"| {entry.chapter} | {entry.name} | {'/'.join(entry.by)} "
            f"| {len(tallied)} | {sum(tallied.values())} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The Instrumentation Level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One rung's localization question, and the query that answers it.

    `ask` returns the answer as a string or `None` when the record cannot
    produce one. That is deliberately not a check that a field exists: a seam
    passes only if running the query against a real failing request yields an
    actual answer, which is a harder bar and the only one worth anything.

    `subject` is the site family the question interrogates, and it is Chapter
    13's Scoring Surface arriving at the runtime layer. Each rung's question
    asks about the surface THAT rung added - Level 3 about stages, Level 4
    about tool arguments, Level 6 about rounds - because those are the places
    the rung can be wrong independently of its answer, which is what makes them
    the places worth localizing.

    Every question here is an obligation an earlier chapter wrote down before
    this record was designed. That does not make the exam un-self-graded, and
    the chapter says so - but the questions were not chosen to suit the answer.
    """

    level: int
    subject: str
    text: str
    ask: Callable[[Trace], str | None]


def _named(event: Event | None, *fields: str) -> str | None:
    if event is None or event.kind == UNKNOWN:
        return None
    extra = " ".join(f"{k}={v}" for k, v in event.fields if k in fields)
    return f"{event.site} kind={event.kind}{' ' + extra if extra else ''}"


def _which_rule(trace: Trace) -> str | None:
    at = [e for e in trace.where(level=0).events if e.site == "rung"]
    return _named(at[0] if at else None)


def _which_draw(trace: Trace) -> str | None:
    versioned = [e for e in trace.events if e.prompt_version]
    if not versioned:
        return None
    first = versioned[0]
    return f"prompt={first.prompt_version} draw={first.draw}"


def _retrieval_or_generation(trace: Trace) -> str | None:
    at = trace.sites("retrieve")
    return _named(at[0] if at else None, "citation")


def _which_stage(trace: Trace) -> str | None:
    failed = [e for e in trace.sites("stage.") if e.outcome != "answered"]
    return _named(failed[0] if failed else None, "input_from")


def _which_tool_argument(trace: Trace) -> str | None:
    at = [e for e in trace.sites("tool.") if e.outcome != "answered"]
    if not at:
        at = list(trace.sites("tool."))
    if not at:
        return None
    event = at[0]
    arguments = " ".join(f"{k}={v}" for k, v in event.fields)
    return f"{event.site}({arguments}) -> {event.kind}" if arguments else None


def _which_step(trace: Trace) -> str | None:
    at = [e for e in trace.sites("step.") if e.outcome != "answered"]
    return _named(at[0] if at else None, "kept")


def _which_round(trace: Trace) -> str | None:
    stops = trace.where(site="stop").events
    rounds = trace.where(site="round").events
    if not stops or not rounds:
        return None
    stop = stops[0]
    return f"stopped after {len(rounds)} rounds, {stop.kind}={dict(stop.fields).get('bound')}"


def _which_role(trace: Trace) -> str | None:
    blocked = [e for e in trace.events if e.kind == "blocked"]
    handoffs = [e for e in trace.events if e.kind == "handoff"]
    if not blocked:
        return None
    carried = sum(int(dict(e.fields).get("characters", 0)) for e in handoffs)
    return f"{blocked[0].site} after {len(handoffs)} handoffs carrying {carried} chars"


QUESTIONS: tuple[Question, ...] = (
    Question(0, "rung", "Which rule decided this, and did any fire?", _which_rule),
    Question(1, "rung", "Which prompt version ran, and which draw answered?",
             _which_draw),
    Question(2, "retrieve", "Is this a retrieval failure or a generation failure?",
             _retrieval_or_generation),
    Question(3, "stage.", "Which stage failed, and where did its input come from?",
             _which_stage),
    Question(4, "tool.", "Which tool call, with which arguments, returned nothing?",
             _which_tool_argument),
    Question(5, "step.", "Which step's finding was discarded?", _which_step),
    Question(6, "round", "How many rounds ran, and which bound stopped them?",
             _which_round),
    Question(7, "role.", "Which role blocked, and what did the handoffs carry?",
             _which_role),
)

# Outcomes that mean something to localize. `unknown` is excluded on purpose: a
# record that cannot say whether a decision succeeded has not reported a
# failure, and counting it as one would credit the poorest seam with finding the
# most problems.
WENT_WRONG: frozenset[str] = frozenset({"refused", "failed"})


def exercised(
    reference: Mapping[int, Iterable[Trace]], question: Question
) -> frozenset[str]:
    """Which requests have something for this question to localize.

    Chapter 13's rule, arriving at the runtime layer unchanged: a request where
    nothing went wrong is not a request the record failed to explain. Grading a
    seam against clean requests would score it on the population it was never
    built for, in the direction that flatters it.

    Computed once from the RICHEST record and then applied to every seam, which
    is the only fair arrangement. Letting each seam nominate its own population
    would let a record that cannot express failure at all report that it had
    none to explain.
    """
    return frozenset(
        trace.incident_id
        for trace in reference.get(question.level, ())
        if any(
            event.level == question.level
            and event.site.startswith(question.subject)
            and event.outcome in WENT_WRONG
            for event in trace.events
        )
    )


def localizes(
    seam: Mapping[int, Iterable[Trace]], question: Question, on: frozenset[str]
) -> bool:
    """Can this seam answer one rung's question on every request that needs it?

    Every such request, not the median one. A seam that localizes most failures
    and loses the tail is the sampling defect this chapter warns about, wearing
    a passing grade.
    """
    at = [t for t in seam.get(question.level, ()) if t.incident_id in on]
    return bool(at) and all(question.ask(trace) is not None for trace in at)


def instrumentation_level(
    seam: Mapping[int, Iterable[Trace]],
    reference: Mapping[int, Iterable[Trace]],
    questions: Iterable[Question] = QUESTIONS,
) -> int | None:
    """The highest rung whose failures this record can localize, or None.

    Highest CONTIGUOUS rung, and the distinction earns its place: a seam that
    answers Level 6's question and not Level 3's has not instrumented Level 6,
    it has instrumented one field of it. Debugging a rung means debugging what
    it is built on, which is the accretion argument the code has followed since
    Chapter 3, arriving at the observability layer.

    A rung the corpus never made fail counts as NOT localized, which is
    deliberately conservative and is the uncomfortable half of this number: you
    cannot demonstrate that you can debug a rung that has not broken yet, and
    the traffic that would demonstrate it is the incident you are trying to be
    ready for.
    """
    reached: int | None = None
    for question in sorted(questions, key=lambda q: q.level):
        if not localizes(seam, question, exercised(reference, question)):
            return reached
        reached = question.level
    return reached
