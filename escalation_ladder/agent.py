"""Level 6 - the autonomous agent.

The rung where the system decides when it is done. Chapter 8's chain answered the
Termination Test by reading a tuple: three steps, six requests, computable before
the run began and unchangeable by any input. Delete the tuple and the answer
becomes "however many it takes", which is not a maximum - so this module buys one
back at a price, and the price is the subject of the chapter.

Three decisions carry it.

THE MODEL IS NOT TOLD A MAXIMUM. Chapter 7 established that telling a model its
round budget is not prompt engineering but telling the truth about the system,
and Chapter 8 measured incidents expanding to fill whatever depth they were
offered. At this rung the truth is that there is no fixed number, so `Budget` is
a circuit breaker rather than a design and the prompt says nothing about it. An
agent told "you have eight rounds" is a chain with a longer tuple, and it behaves
like one.

THERE IS ONE NEW FIELD, and it is the only vocabulary this rung adds. Chapter 8
declined to invent a schema for "not yet", because `needs_human` plus `next_step`
already said it. It could not say "never", because a chain was never asked -
running out of steps was the code's decision. A loop has to make that call
itself, so `Verdict.unreachable` exists and nothing below this file has changed.

THERE ARE THREE MEMORY LAYERS, with three lifetimes and three owners. The
scratchpad (`AgentState`) is what the loop needs to function. The journal
(`Journal`) is what you need to operate it, and it outlives the process. Recall
(`recall`) is what makes the second run cheaper than the first, and it is built
by indexing the journal - which is to say layer three is layer two, read back
later, through Chapter 5's retriever rather than through a new dependency.

Measured 2026-07-28 - regenerate the reader-runnable halves with
scripts/loop_bounds.py, scripts/agent_memory.py, and scripts/resume.py.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import Field

from escalation_ladder.fixtures.deploys import recent_deploys
from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.instrument import CostLedger, measured
from escalation_ladder.llm import (
    AnthropicCompleter,
    Completer,
    MeteredCompleter,
    ToolCall,
    ToolResult,
)
from escalation_ladder.orchestration import DEFAULT_LOOP, Loop, Node
from escalation_ladder.retrieve import Passage, build_index, search
from escalation_ladder.rules import SEVERITIES, RoutingDecision
from escalation_ladder.rules import escalation_after, owner_for
from escalation_ladder.rungs import register_rung
from escalation_ladder.tools import Finding, Toolbox, menu_for, value_supported


class Verdict(Finding):
    """Chapter 7's `Finding` plus the one thing only this rung can act on.

    Chapter 8 argued that a rung needing new vocabulary to say "not yet" is
    telling you the schema below it was underspecified, and then added no fields.
    This is not that. A chain could not say "never" because a chain was never
    asked: running out of steps was the code's decision, taken on the code's
    schedule, and `exhausted` was something the caller observed rather than
    something the model reported.

    A loop has no such caller. Nothing else in the system knows whether reading
    one more log line would settle the incident, so if the model cannot say
    "stop, there is nothing here for me", the only thing that ever stops a
    hopeless run is the circuit breaker - which is the most expensive way to
    learn the cheapest fact.
    """

    unreachable: bool = Field(
        description=(
            "True when no further reading with these tools could establish the "
            "cause - not merely that you have not found it yet."
        )
    )


@dataclass(frozen=True)
class Budget:
    """The three bounds, and they fail for three different reasons.

    A single `max_iterations` is the version most systems ship, and it is the
    one bound that measures the thing you care about least. Rounds are a proxy
    for spend, and the proxy drifts every time the working set grows: eight
    rounds costs one number on the first incident of the day and another on an
    incident that has accumulated forty tool results.

    Hitting each of these means something different. The iteration bound says
    the agent kept finding plausible next reads. The token bound says the
    working set grew faster than the investigation converged. The deadline says
    the incident is still open and nobody is being paged. They are reported
    separately for the same reason Chapter 5 refused to collapse four refusal
    reasons into one `needs_human` counter.
    """

    iterations: int = 8
    tokens: int = 120_000
    seconds: float = 600.0


# How many earlier findings are carried in full. Older ones collapse to their
# cause alone. Chapter 8 could carry everything because "everything" was at most
# two prior steps; here the ceiling is gone, so a carry policy that keeps all of
# it is a working set that grows without bound inside a run whose length is also
# unbounded - the Context Ratchet with both of its limits removed.
CARRY_VERBATIM: int = 3

# The relevance floor for recall, over the journal corpus. Chapter 5's FLOOR of
# 6.0 is deliberately NOT reused: BM25 scores are not portable across corpora,
# Chapter 5 says so in prose, and a journal of investigation summaries is a
# different corpus from a set of runbooks. scripts/agent_memory.py recalibrates
# it the way Chapter 5 did - by scoring queries the journal cannot answer and
# putting the line above the best of them.
RECALL_FLOOR: float = 4.5


SYSTEM = """You investigate production incidents for an on-call rota by reading
live telemetry with the tools provided.

You work in rounds. Each round gives you one set of tool calls: every call in
that round runs together and you receive every result at once. You may keep going
for as many rounds as the investigation needs. Nothing here tells you a number,
because there is not one - you decide when this is finished.

Finish in one of exactly two ways.

Set `needs_human` to false when the telemetry SHOWS the cause and you can quote
the value that shows it. That is the answer.

Set `unreachable` to true when no further reading with these tools could
establish the cause - the evidence lives somewhere you cannot see, or the
question is not answerable from telemetry at all. This is a real answer and not a
failure. Say it as soon as it is true, and do not keep reading in the hope that
something turns up.

While neither is true, leave `needs_human` true, name the single most useful
thing to read next in `next_step`, and you will get another round to do it.

The services, and what each one owns:
- checkout-api: the checkout flow and order placement
- payment-gateway: card authorization and capture, and the acquirer connection
- notification-worker: transactional email and push delivery
- search-api: search queries and the search index
Log sources also include service-mesh, the sidecar proxy every service calls
through. It owns no rota.

Findings from earlier rounds are CLAIMS, not facts. They were produced by this
same process against less evidence than you now have, and they are sometimes
wrong. You are shown the original report every round so that you can weigh a
claim against it. If a claim disagrees with what you read, say what you read.

Rules:
- `evidence_value` must be copied character for character out of a tool result,
  from this round or an earlier one. Do not round it, reformat it, or summarize
  it. A finding whose value cannot be found in a tool result is discarded and
  the round is wasted.
- `evidence_tool` is the name of the tool whose result you copied from.
- Do not repeat a call you have already made. The calls you have made are listed
  for you, and a round spent re-reading is a round.
- Base the cause on values you read. A cause the telemetry does not show is a
  guess, however reasonable, and this rung exists to stop guessing.
- Severity is about customer impact, not tone."""


@dataclass(frozen=True)
class Attempt:
    """What one round did. Chapter 8's `StepRecord` for a sequence with no end.

    `index` replaces Chapter 8's step name as the join key onto the ledger, and
    the change is forced rather than stylistic: a chain's rows were named
    `observe`, `localize`, `confirm` and joined uniquely, while a loop runs the
    same node repeatedly and a name would collide with itself. Any trace format
    keyed on stage name silently merges rows at this rung.
    """

    index: int
    ok: bool
    detail: str
    calls: tuple[ToolCall, ...] = ()
    finding: Verdict | None = None

    def __post_init__(self) -> None:
        # See `AgentState.__post_init__`. A checkpointer hands this back with
        # `calls` as a list, and nested dataclasses are restored the same way.
        object.__setattr__(self, "calls", tuple(self.calls))


@dataclass(frozen=True)
class Entry:
    """One finished run, as a later run will read it. Memory layer two.

    Deliberately not a transcript. A transcript is what a log already holds and
    it is unreadable at the volume a loop produces; this is the set of fields
    that answer "what happened, what did it cost, and why did it stop" without
    opening anything else. That question is the reconstruction check, and a
    system that cannot answer it from its own storage has one memory layer.
    """

    run_id: str
    incident_id: str
    service: str
    report: str
    routed: bool
    cause: str | None
    evidence: str | None
    rounds: int
    stopped_by: str | None
    tools: tuple[str, ...]

    def searchable(self) -> str:
        """What the index sees. The report is included and the verdict is not.

        A later incident is matched on how it PRESENTED, not on what this one
        turned out to be, because the presentation is all the later incident has
        at recall time. Indexing the cause instead would retrieve on a fact the
        querying run does not yet know, which reads well in a demo and retrieves
        nothing in production.
        """
        return f"{self.report}\n{self.cause or ''}"

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "incident_id": self.incident_id,
                "service": self.service,
                "report": self.report,
                "routed": self.routed,
                "cause": self.cause,
                "evidence": self.evidence,
                "rounds": self.rounds,
                "stopped_by": self.stopped_by,
                "tools": list(self.tools),
            },
            sort_keys=True,
        )

    @staticmethod
    def from_json(line: str) -> "Entry":
        raw: dict[str, Any] = json.loads(line)
        return Entry(
            run_id=str(raw["run_id"]),
            incident_id=str(raw["incident_id"]),
            service=str(raw["service"]),
            report=str(raw["report"]),
            routed=bool(raw["routed"]),
            cause=raw["cause"],
            evidence=raw["evidence"],
            rounds=int(raw["rounds"]),
            stopped_by=raw["stopped_by"],
            tools=tuple(raw["tools"]),
        )


@dataclass(frozen=True)
class Journal:
    """Memory layer two: structured task history that outlives the process.

    One JSON object per line, appended, never rewritten. That is not a storage
    recommendation - it is the smallest thing that survives the failure this
    layer exists for. An agent loop that dies at round six leaves a scratchpad in
    a dead process's heap, and every question anybody asks the next morning is
    about state that no longer exists.

    A file is enough because the record is small and the read pattern is "all of
    it". Swap it for a table when you have more runs than fit in memory; the
    shape of the record is the part worth copying.
    """

    path: Path

    def append(self, entry: Entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json() + "\n")

    def entries(self) -> tuple[Entry, ...]:
        if not self.path.exists():
            return ()
        with self.path.open("r", encoding="utf-8") as handle:
            return tuple(
                Entry.from_json(line) for line in handle if line.strip()
            )


def recall(
    journal: Journal,
    incident: Incident,
    *,
    limit: int = 2,
    floor: float = RECALL_FLOOR,
) -> tuple[Entry, ...]:
    """Memory layer three: what earlier runs learned, ranked against this report.

    No new dependency and no vector store. Chapter 5 shipped BM25 in the standard
    library precisely so that a lexical baseline is the number an embedding index
    has to beat, and the corpus here is a few hundred short records rather than a
    document set. `build_index` and `search` are imported unchanged.

    The exclusion is Chapter 5's `exclude_incident` discipline, restated because
    it is even easier to get wrong here. A journal accumulates runs of the SAME
    incident, so an agent investigating INC-1044 for the second time can be
    handed its own previous answer and reach it in one round - a memory system
    that measures beautifully and has learned nothing.
    """
    history = tuple(
        entry
        for entry in journal.entries()
        if entry.incident_id != incident.incident_id
    )
    if not history:
        return ()
    passages = tuple(
        Passage(
            passage_id=f"run:{entry.run_id}",
            source=entry.incident_id,
            text=entry.searchable(),
        )
        for entry in history
    )
    by_id = {f"run:{entry.run_id}": entry for entry in history}
    hits = search(
        build_index(passages), incident.report, limit=limit, floor=floor
    )
    return tuple(by_id[hit.passage.passage_id] for hit in hits)


@dataclass(frozen=True)
class AgentState:
    """Memory layer one: the scratchpad, and everything the loop carries.

    Immutable and rebuilt with `replace` for the same reason Chapter 8's
    `ChainState` was: a state a framework may be holding has to cross a boundary,
    and an immutable value is a state you can snapshot, diff, and resume from.
    The checkpointer in `orchestration.LangGraphLoop` needs exactly this.

    `attempts` only grows, and at this rung nothing bounds how far.
    """

    incident: Incident
    recalled: tuple[Entry, ...] = ()
    attempts: tuple[Attempt, ...] = ()
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()
    settled: Verdict | None = None
    gave_up: str | None = None
    failed: str | None = None

    def __post_init__(self) -> None:
        """Coerce the sequence fields back to tuples after a round trip.

        This exists because of a checkpointer and it is the clearest thing this
        module has to say about what a framework costs. The state is immutable so
        that it can be snapshotted and resumed - and the serializer that does the
        snapshotting has no tuple type, so every one of these comes back a list.
        The first symptom was a `TypeError` three rounds into a resumed run,
        which is to say in the only code path a test suite that never crashes
        does not reach.

        Restoring the type here rather than defending at each use site keeps the
        rule in one place: a state that has been through the framework is the
        same value it was before.
        """
        for field_name in ("attempts", "calls", "results", "recalled"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    @property
    def findings(self) -> tuple[tuple[int, Verdict], ...]:
        """Surviving claims, each paired with the round that made it."""
        return tuple(
            (attempt.index, attempt.finding)
            for attempt in self.attempts
            if attempt.finding is not None
        )

    @property
    def finished(self) -> bool:
        """True when the AGENT stopped. Bounds are checked separately.

        Keeping these apart is what makes the cost story readable: a run that
        ends here ended because the work ended, and a run that ends because
        `_bound` fired ended because we ran out of patience or money. Collapsing
        them into one flag is how a system reports a 92 percent completion rate
        that is mostly circuit breaker.
        """
        return (
            self.settled is not None
            or self.gave_up is not None
            or self.failed is not None
        )


def working(state: AgentState) -> str:
    """The user message for one round: the report, recall, the claims, the calls.

    The original report goes first and appears every round. Chapter 6 measured
    what removing it costs, Chapter 8 found it was the ONLY half of the
    Downstream Veto that carried weight, and here it is load-bearing twice over:
    it is the independent thing a claim can be weighed against, and at round nine
    it is the only part of the working set that has not been through a model.

    Older findings collapse to their cause. That is compaction, it is lossy by
    construction, and it is the first place in this book where the system can
    forget something it already paid to read.
    """
    parts = [
        f"Incident {state.incident.incident_id}, opened {state.incident.opened_at}.",
        "",
        f"<report>\n{state.incident.report}\n</report>",
    ]

    if state.recalled:
        parts += ["", "<earlier investigations of other incidents - context only>"]
        for entry in state.recalled:
            parts.append(
                f"- {entry.incident_id} ({entry.service}): "
                f"{entry.cause or 'no cause established'}"
            )
        parts.append("</earlier investigations>")

    findings = state.findings
    if findings:
        parts += ["", f"<what you have established so far, round {len(state.attempts)}>"]
        older, recent = findings[:-CARRY_VERBATIM], findings[-CARRY_VERBATIM:]
        for index, finding in older:
            parts.append(f"- round {index + 1}: {finding.cause}")
        for index, finding in recent:
            parts.append(
                f"- round {index + 1}: {finding.cause} "
                f"(service {finding.service}, from {finding.evidence_tool}: "
                f"{finding.evidence_value!r}). "
                f"Wanted next: {finding.next_step}"
            )
        parts.append("</what you have established so far>")

    if state.calls:
        parts += ["", "<already read - do not repeat these>"]
        parts += [f"- {call.render()}" for call in state.calls]
        parts.append("</already read>")

    parts += ["", "Continue the investigation."]
    return "\n".join(parts)


@dataclass
class DryRunToolbox(Toolbox):
    """Offers the write tool, performs nothing, and does not admit it.

    The lie is the instrument. A tool result that said DRY RUN would tell the
    model its action had no effect, and every round after it would be reasoning
    about a system it knows it did not change - which measures a different agent
    from the one you are worried about. This one answers the way a real rollback
    would answer, so what gets recorded in `proposed` is what a write-capable
    agent would actually have done to production.

    That makes this class dangerous in exactly the way its subject is. It exists
    to be run against fixtures, and the reason it is safe here is that
    `_rollback_deploy` writes to a list rather than to anything real - which is
    the same sentence, and the same amount of protection, as every production
    dry-run mode that was one refactor away from not being one.
    """

    proposed: list[ToolCall] = field(default_factory=list)

    def _rollback_deploy(self, call: ToolCall) -> ToolResult:
        self.proposed.append(call)
        service = str(call.arguments.get("service", ""))
        deploy = str(call.arguments.get("deploy_id", ""))
        return ToolResult(
            call.call_id, f"rollback of {service} to {deploy} accepted"
        )


@dataclass(frozen=True)
class Pursuit:
    """A routed verdict, the rounds that produced it, and what they cost.

    Composes `RoutingDecision` for the seventh chapter running rather than
    extending it. `stopped_by` is the field this rung adds and no rung below
    could have: the name of the bound that ended the run, or `None` when the
    agent stopped on its own. Every cost question in the chapter is asked by
    grouping on that field.
    """

    decision: RoutingDecision
    cause: str | None
    evidence: str | None
    next_step: str | None
    attempts: tuple[Attempt, ...]
    calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...]
    ledger: CostLedger = field(default_factory=CostLedger)
    stopped_by: str | None = None
    gave_up: str | None = None
    service: str | None = None
    recalled: tuple[Entry, ...] = ()
    proposed: tuple[ToolCall, ...] = ()

    @property
    def routed(self) -> bool:
        return self.decision.routed

    @property
    def rounds(self) -> int:
        return len(self.attempts)

    @property
    def api_requests(self) -> int:
        """Requests sent to the vendor. NOT `ledger.model_calls`.

        Chapter 8 hit this trap and named it: `MeteredCompleter` wraps the seam,
        so one `invoke` is one measurement covering two requests. The ledger
        counts rounds; the Termination Test asks about requests. At Level 5 the
        gap was between three and six. Here neither number has a ceiling, and
        quoting the wrong one halves the answer to the only question this rung
        is asked.
        """
        return 2 * len(self.attempts)

    def summary(self) -> str:
        lines = [self.decision.summary()]
        if self.cause is not None:
            lines.append(f"  cause: {self.cause}")
            lines.append(f"  evidence: {self.evidence}")
            lines.append(f"  next step: {self.next_step}")
        if self.gave_up is not None:
            lines.append(f"  the agent stopped itself: {self.gave_up}")
        if self.stopped_by is not None:
            lines.append(f"  stopped by {self.stopped_by}")
        return "\n".join(lines)

    def attempt_table(self) -> str:
        """One line per round, joined to the ledger on the round index."""
        cost = {m.label: m for m in self.ledger.measurements}
        lines = []
        for attempt in self.attempts:
            measurement = cost.get(f"agent.{attempt.index}")
            tokens_in = measurement.input_tokens if measurement else 0
            tokens_out = measurement.output_tokens if measurement else 0
            millis = measurement.latency_ms if measurement else 0.0
            status = "ok  " if attempt.ok else "err "
            lines.append(
                f"  round {attempt.index + 1:<2} {status} "
                f"{len(attempt.calls):>2} calls  "
                f"{tokens_in:>6} in {tokens_out:>4} out  {millis:8.0f}ms  "
                f"{attempt.detail}"
            )
        return "\n".join(lines)

    def tool_table(self) -> str:
        """One line per tool call across the whole run, in the order made."""
        by_id = {result.call_id: result for result in self.results}
        lines = []
        for call in self.calls:
            result = by_id.get(call.call_id)
            status = "err " if result is not None and result.is_error else "ok  "
            head = result.content.splitlines()[0] if result is not None else ""
            lines.append(f"  {call.render():<58} {status} {head}")
        return "\n".join(lines)

    def entry(self, run_id: str, report: str = "") -> Entry:
        """This run, in the shape memory layer two stores.

        The report is passed in rather than read back from the incident
        database, and that is the layer working as intended: a record that
        needed another system to be up in order to be readable would not
        survive the failure it exists to survive.
        """
        return Entry(
            run_id=run_id,
            incident_id=self.decision.incident_id,
            service=self.service or "unknown",
            report=report,
            routed=self.routed,
            cause=self.cause,
            evidence=self.evidence,
            rounds=self.rounds,
            stopped_by=self.stopped_by,
            tools=tuple(call.render() for call in self.calls),
        )


def _unrouted(incident: Incident, reason: str) -> RoutingDecision:
    """Chapter 3's refusal shape, carried up one more rung."""
    return RoutingDecision(
        incident_id=incident.incident_id,
        rota=None,
        severity=None,
        escalate_after_minutes=None,
        already_overdue=False,
        recent_deploy=None,
        needs_human=True,
        decided_by=reason,
    )


def _at_least(claimed: str, tier: str) -> str:
    """Chapter 3's one-directional severity floor, unchanged for the seventh time."""
    return SEVERITIES[min(SEVERITIES.index(claimed), SEVERITIES.index(tier))]


def _bound(
    state: AgentState,
    ledger: CostLedger,
    started: float,
    budget: Budget,
    *,
    now: float | None = None,
) -> str | None:
    """Which bound has been reached, if any. Checked before every round.

    Returning a string rather than a bool is the whole point. Three bounds that
    all set the same flag are one bound with three implementations, and the
    operational question - which of these keeps firing? - stops being askable.
    """
    if len(state.attempts) >= budget.iterations:
        return f"iterations: {len(state.attempts)} of {budget.iterations}"
    spent = ledger.total_input_tokens + ledger.total_output_tokens
    if spent >= budget.tokens:
        return f"tokens: {spent} of {budget.tokens}"
    elapsed = (time.monotonic() if now is None else now) - started
    if elapsed >= budget.seconds:
        return f"deadline: {elapsed:.0f}s of {budget.seconds:.0f}s"
    return None


def _round_node(
    completer: Completer,
    box: Toolbox,
    ledger: CostLedger,
    *,
    allow_writes: bool,
    honor_unreachable: bool = True,
) -> Node[AgentState]:
    """Build the node the loop runs. One node, run until `done` says otherwise.

    Chapter 8 built one node per step because the steps differed. Here there is
    exactly one, which is the clearest statement of what changed: the rung has no
    stages, no roles, and nothing that distinguishes round seven from round one
    except what the working set has accumulated by then.
    """

    def advance(state: AgentState) -> AgentState:
        index = len(state.attempts)
        metered = MeteredCompleter(
            inner=completer, ledger=ledger, stage=f"agent.{index}"
        )
        before = len(box.calls)

        def execute(call: ToolCall) -> ToolResult:
            billed = measured(f"agent.{call.call_id}", ledger)(box.execute)
            return billed(call)

        run_result = metered.invoke(
            system=SYSTEM,
            user=working(state),
            tools=menu_for(allow_writes=allow_writes),
            execute=execute,
            schema=Verdict,
        )

        made = tuple(box.calls[before:])
        calls = state.calls + made
        results = state.results + run_result.results

        def recorded(
            ok: bool, detail: str, finding: Verdict | None = None
        ) -> AgentState:
            return replace(
                state,
                attempts=state.attempts
                + (Attempt(index, ok, detail, made, finding),),
                calls=calls,
                results=results,
            )

        if run_result.wanted_more:
            # Chapter 7's Failure Receipt, Chapter 8's ordinary case, and here it
            # is not even reported: another round was always going to happen.
            return recorded(True, "asked for another round")
        if not run_result.ok or run_result.parsed is None:
            return replace(
                recorded(False, run_result.failed or "no result"),
                failed=run_result.failed or "no result",
            )

        finding = run_result.parsed

        if finding.unreachable and honor_unreachable:
            # Checked BEFORE the citation check, and the order is a bug fix
            # rather than a preference. A verdict of "nothing here would settle
            # this" asserts no fact, so it has no value to quote - and the first
            # version of this function ran the citation check first, discarded
            # every give-up on exactly that ground, and sent the agent back round
            # to try again. The one exit that is not a circuit breaker was
            # unreachable, and the tests caught it rather than the bill.
            #
            # The finding is recorded but not carried: it makes no claim, so
            # there is nothing for a later round to inherit.
            return replace(
                recorded(True, f"stopped itself: {finding.next_step}"),
                gave_up=finding.next_step or "no further reading would settle it",
            )

        if not value_supported(finding, calls, results):
            # Chapter 8 moved this check from gating the ANSWER to gating the
            # CONTEXT. At Level 5 that saved two later steps. Here it saves an
            # unknown number, which is the first time in the book that a
            # defensive check has an unbounded return.
            return recorded(
                False,
                f"discarded: {finding.evidence_value!r} is not in any tool result",
            )

        settled = (
            finding
            if not finding.needs_human and finding.service != "unknown"
            else None
        )
        detail = (
            f"{finding.evidence_tool} showed {finding.evidence_value}"
            if settled is not None
            else f"wants next: {finding.next_step}"
        )
        return replace(recorded(True, detail, finding), settled=settled)

    return Node(name="round", fn=advance)


def pursue(
    incident: Incident,
    completer: Completer,
    *,
    budget: Budget | None = None,
    loop: Loop | None = None,
    journal: Journal | None = None,
    recall_floor: float = RECALL_FLOOR,
    allow_writes: bool = False,
    honor_unreachable: bool = True,
    box: Toolbox | None = None,
    initial: AgentState | None = None,
) -> Pursuit:
    """Investigate until the agent stops, the vendor fails, or a bound fires.

    `honor_unreachable` exists for one experiment and defaults to the behavior
    the rung ships. Setting it False ignores the model's request to stop, which
    is what a loop with no give-up field does and, more usefully, what this
    module itself did before the citation check was reordered. It is the arm
    `scripts/loop_bounds.py --no-quit` runs, and the difference between the two
    arms is the whole cost profile of the rung.

    `journal` is optional and defaults to off, so the rung's own cost row
    measures the loop rather than the loop plus a memory system. Pass one and
    layer three is read once, before the first round: recall is context, not a
    tool, because a fact retrieved mid-investigation from a previous
    investigation arrives with no way for the model to tell it apart from
    something it read itself.

    `box` exists so a caller can supply `DryRunToolbox`. `initial` exists so a
    script can start a run from a state it constructed, which is how the drift
    and resume experiments work.
    """
    limits = budget if budget is not None else Budget()
    ledger = CostLedger()
    toolbox = box if box is not None else Toolbox(
        incident=incident, allow_writes=allow_writes
    )
    walker = loop if loop is not None else DEFAULT_LOOP

    remembered: tuple[Entry, ...] = ()
    if journal is not None:
        remembered = recall(journal, incident, floor=recall_floor)
    start = (
        initial
        if initial is not None
        else AgentState(incident=incident, recalled=remembered)
    )

    started = time.monotonic()
    node = _round_node(
        completer,
        toolbox,
        ledger,
        allow_writes=allow_writes,
        honor_unreachable=honor_unreachable,
    )
    final = walker.run(
        node,
        start,
        done=lambda state: (
            state.finished or _bound(state, ledger, started, limits) is not None
        ),
    )

    # Recomputed rather than captured inside `done`, because `done` is a
    # predicate a framework may call more than once per round and a predicate
    # with a side effect would record whichever call happened to be last.
    # `finished` is checked first so that a run which settled on its last
    # permitted round is not reported as a run the circuit breaker stopped.
    stopped_by = (
        None if final.finished else _bound(final, ledger, started, limits)
    )
    proposed = tuple(getattr(toolbox, "proposed", ()))

    def result(decision: RoutingDecision, **kwargs: Any) -> Pursuit:
        base: dict[str, Any] = dict(
            decision=decision,
            cause=None,
            evidence=None,
            next_step=None,
            attempts=final.attempts,
            calls=final.calls,
            results=final.results,
            ledger=ledger,
            stopped_by=stopped_by,
            gave_up=final.gave_up,
            recalled=remembered,
            proposed=proposed,
        )
        base.update(kwargs)
        return Pursuit(**base)

    if final.settled is None:
        wanted = final.findings[-1][1].next_step if final.findings else None
        reason_text = (
            final.failed
            or final.gave_up
            or (
                f"stopped by {stopped_by}; next check was: {wanted}"
                if stopped_by and wanted
                else stopped_by
                or "the telemetry does not explain this incident"
            )
        )
        return result(_unrouted(incident, reason_text))

    finding = final.settled
    service = owner_for(finding.service)
    if service is None:
        return result(
            _unrouted(incident, f"no rota owns service {finding.service!r}")
        )

    severity = _at_least(finding.severity, service.tier)
    deploys = recent_deploys(finding.service)
    return result(
        RoutingDecision(
            incident_id=incident.incident_id,
            rota=service.rota,
            severity=severity,
            escalate_after_minutes=escalation_after(severity),
            already_overdue=False,
            recent_deploy=(
                f"{deploys[0].deploy_id} {deploys[0].deployed_at}"
                if deploys
                else None
            ),
            needs_human=False,
            decided_by=(
                f"round {len(final.attempts)}: "
                f"{finding.evidence_tool} showed {finding.evidence_value}"
            ),
        ),
        cause=finding.cause,
        evidence=f"{finding.evidence_tool} -> {finding.evidence_value}",
        next_step=finding.next_step,
        service=finding.service,
    )


@register_rung("Level 6: Autonomous Agent")
def run(incident: Incident, completer: Completer | None = None) -> CostLedger:
    """Rung entry point. The maximum is a `Budget`, and that is the whole story.

    Every rung above Level 0 has been able to answer "how many model calls?" from
    its own source. This one answers it from a dataclass of three limits that
    somebody chose, and the honest description of the number is the one Chapter 2
    gave: a circuit breaker, not a design.
    """
    active: Completer = completer if completer is not None else AnthropicCompleter()
    return pursue(incident, active).ledger
