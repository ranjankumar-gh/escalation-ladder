"""Level 5 - multi-step reasoning.

The rung where a call's arguments can come from an earlier call's result. Chapter
7 gave the model one round: every tool ran together, so a fact learned in one
result could not be used by another. INC-1044 routed 2 times in 15 because of
exactly that. The mesh logs are keyed `peer=inventory`; the model searches them
for `checkout`, because checkout is what it was asked about, and the string
`inventory` lives in checkout-api's own log line, which arrives in a sibling
result rather than an earlier one.

This module is a CHAIN, not a loop, and the distinction is the whole rung.
`STEPS` is a tuple. The walk executes each step at most once and can stop early,
so the maximum number of model calls is `2 * len(STEPS)` before the run begins
and no input can raise it. Reaching the last step is the design rather than an
intervention, which is what separates this from `max_iterations`, and it is why
Chapter 9's first change is to delete the tuple.

Three decisions carry the chapter.

There is NO NEW SCHEMA. Every step emits Chapter 7's `Finding`, unchanged. The
field that makes a chain possible was already there: Level 4 had to be able to
say "the telemetry does not explain this, and here is what I would read next",
so `needs_human` plus `next_step` is a request for another step written by a rung
that could not take one. What changes at Level 5 is not the vocabulary, it is
that something now acts on it.

Every step is shown the ORIGINAL report and is told that earlier findings are
claims. That is Chapter 6's Downstream Veto carried into a chain, and it is the
only defence against the second half of the Context Ratchet: intermediate state
only accumulates, so a wrong early conclusion is re-read as established fact by
every step that follows it, each of which spends real tool calls confirming a
fiction.

The citation check moved. Chapter 4 checked a quote to gate an ANSWER; Chapter 7
checked a value for the same reason. Here `value_supported` gates the CONTEXT: a
finding whose cited value is not in any tool result does not enter the carried
state at all. The same check, one layer earlier, stops being diagnostic and
starts being preventive - and it still cannot catch a real value that was
misread, which is what the veto is for.

Measured 2026-07-27 - regenerate the reader-runnable halves with
scripts/chain_depth.py and scripts/drift.py, or replay the recordings in tests/
for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

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
from escalation_ladder.orchestration import DEFAULT_CHAIN, Chain, Node
from escalation_ladder.rules import SEVERITIES, RoutingDecision
from escalation_ladder.rules import escalation_after, owner_for
from escalation_ladder.rungs import register_rung
from escalation_ladder.tools import (
    Finding,
    Toolbox,
    ToolSpec,
    menu_for,
    value_supported,
)


@dataclass(frozen=True)
class Step:
    """One step's name and the job it is given, in the words the model reads.

    `purpose` is prompt text rather than documentation. It is the only thing that
    differs between steps, which is the point: the steps are not different code
    paths, they are the same capability pointed at a narrowing question.
    """

    name: str
    purpose: str


# THE ARCHITECTURE, and the answer to the Termination Test. Three entries, so at
# most six model calls and three rounds of tools, computable by reading this
# tuple and knowing that `llm.invoke` makes two requests.
#
# The names are roles rather than a script. Nothing here says which tool to call
# or what to search for - that is chosen at run time from what the previous step
# read, and it is why a path can exist through this chain that nobody wrote.
STEPS: tuple[Step, ...] = (
    Step(
        "observe",
        "Establish what is actually failing. Read the telemetry for the service "
        "the report names. You are looking for the symptom in the system's own "
        "words, including any identifier - a cluster, a peer, an upstream, a "
        "queue - that the report does not contain.",
    ),
    Step(
        "localize",
        "Find the component responsible. Use the identifiers the previous step "
        "read. If a name appeared in a log line and you have not looked at what "
        "it refers to, look at it now - that is usually the whole gap between a "
        "symptom and a cause.",
    ),
    Step(
        "confirm",
        "Confirm or reject the cause. Read the one thing that would settle it, "
        "and quote the value that does. If the telemetry still does not show a "
        "cause, say so rather than reporting the most plausible story.",
    ),
)

# Each step is one `invoke`, and `invoke` is two requests. Printed by the chapter
# and asserted by a test, because a number derived in prose is a number that goes
# stale in prose.
MAX_MODEL_CALLS: int = 2 * len(STEPS)


SYSTEM = """You investigate production incidents for an on-call rota by reading
live telemetry with the tools provided.

You work in a fixed number of steps. Each step gives you one round of tools:
every call in that round runs together and you receive every result at once.
There is no way to earn an extra step, and the last step is the last word. You
are told which step you are on and how many there are.

Within a step, choose the set of calls that answers that step's question rather
than a first call whose result you plan to build on - the building happens
between steps, not inside one.

The services, and what each one owns:
- checkout-api: the checkout flow and order placement
- payment-gateway: card authorization and capture, and the acquirer connection
- notification-worker: transactional email and push delivery
- search-api: search queries and the search index
Log sources also include service-mesh, the sidecar proxy every service calls
through. It owns no rota.

Findings from earlier steps are CLAIMS, not facts. They were produced by this
same process against less evidence than you now have, and they are sometimes
wrong. You are shown the original report every step so that you can weigh a
claim against it. If a claim disagrees with what you read, say what you read.

Rules:
- `evidence_value` must be copied character for character out of a tool result,
  from this step or an earlier one. Do not round it, reformat it, or summarize
  it. A finding whose value cannot be found in a tool result is discarded and
  the step is wasted.
- `evidence_tool` is the name of the tool whose result you copied from.
- Set `needs_human` to false ONLY when the telemetry shows the cause. On any
  earlier step that usually means true, and that is the expected answer rather
  than a failure - it is how you ask for the next step.
- When `needs_human` is true, `next_step` is the single most useful thing to
  read next, named specifically enough to execute: which source, which service,
  which string. "Investigate further" wastes the step it buys.
- Base the cause on values you read. A cause the telemetry does not show is a
  guess, however reasonable, and this rung exists to stop guessing.
- Severity is about customer impact, not tone."""


@dataclass(frozen=True)
class StepRecord:
    """What one step did. Chapter 6's `StageRecord`, for a sequence chosen at
    run time rather than declared at import time.

    `detail` is this module's `decided_by`. Cost is deliberately absent for the
    same reason it was absent there - the trace says what happened, the ledger
    says what it cost, and they join on the step name.

    `finding` is `None` when the step produced nothing that survived the citation
    check. Keeping it here rather than in a parallel tuple is not tidiness: a
    separate list of findings and a list of records drift apart the first time a
    step is discarded, and the code that pairs them for the next step's prompt
    would then attribute one step's claim to another step's name.
    """

    name: str
    ok: bool
    detail: str
    calls: tuple[ToolCall, ...] = ()
    finding: Finding | None = None


@dataclass(frozen=True)
class ChainState:
    """Everything one step hands the next.

    Immutable, and rebuilt with `replace` at every step, so the accumulated state
    is a value that can be inspected, diffed, and injected into rather than a
    mutable object a framework happens to be holding. `scripts/drift.py` needs
    exactly that: it starts a walk from a state carrying a claim the model never
    made.

    `records` is the Context Ratchet in one field. It only ever grows.
    """

    incident: Incident
    records: tuple[StepRecord, ...] = ()
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()
    settled: Finding | None = None
    failed: str | None = None

    @property
    def findings(self) -> tuple[tuple[str, Finding], ...]:
        """Surviving claims, each paired with the step that made it."""
        return tuple(
            (record.name, record.finding)
            for record in self.records
            if record.finding is not None
        )

    @property
    def finished(self) -> bool:
        return self.settled is not None or self.failed is not None


def carried(state: ChainState, step: Step, index: int, total: int) -> str:
    """The user message for one step: the report, the claims, and the question.

    The order is deliberate. The original report goes FIRST and appears at every
    step, because a step that sees only the accumulated summary has no
    independent thing to weigh a claim against - Chapter 6 measured what that
    costs and this is the same mechanism one rung up.

    Tool calls already made are listed by name and arguments only, without their
    results. Two reasons. The model needs them to avoid re-reading what it has
    already read, and it needs the RESULTS not to be re-sent, because re-sending
    every result at every step is what turns the ratchet from linear into
    quadratic.
    """
    parts = [
        f"Incident {state.incident.incident_id}, opened {state.incident.opened_at}.",
        "",
        f"<report>\n{state.incident.report}\n</report>",
        "",
        f"You are on step {index + 1} of {total}: {step.name}.",
        step.purpose,
    ]

    if state.findings:
        parts += ["", "<claims from earlier steps - these may be wrong>"]
        for name, finding in state.findings:
            parts.append(
                f"- step {name}: {finding.cause} "
                f"(service {finding.service}, from {finding.evidence_tool}: "
                f"{finding.evidence_value!r}). "
                f"Wanted next: {finding.next_step}"
            )
        parts.append("</claims>")

    if state.calls:
        parts += ["", "<already read - do not repeat these>"]
        parts += [f"- {call.render()}" for call in state.calls]
        parts.append("</already read>")

    return "\n".join(parts)


def _step_node(
    step: Step,
    index: int,
    total: int,
    completer: Completer,
    box: Toolbox,
    ledger: CostLedger,
    tools: tuple[ToolSpec, ...] | None = None,
) -> Node[ChainState]:
    """Build the node that runs one step. The closure holds the run's context so
    the walked state stays a plain value the framework can carry."""

    def advance(state: ChainState) -> ChainState:
        metered = MeteredCompleter(
            inner=completer, ledger=ledger, stage=f"reasoning.{step.name}"
        )

        before = len(box.calls)

        def execute(call: ToolCall) -> ToolResult:
            billed = measured(f"reasoning.{call.call_id}", ledger)(box.execute)
            return billed(call)

        run_result = metered.invoke(
            system=SYSTEM,
            user=carried(state, step, index, total),
            tools=menu_for() if tools is None else tools,
            execute=execute,
            schema=Finding,
        )

        made = tuple(box.calls[before:])
        results = state.results + run_result.results
        calls = state.calls + made

        def recorded(
            ok: bool, detail: str, finding: Finding | None = None
        ) -> ChainState:
            return replace(
                state,
                records=state.records
                + (StepRecord(step.name, ok, detail, made, finding),),
                calls=calls,
                results=results,
            )

        if run_result.wanted_more:
            # Chapter 7 issued its Failure Receipt on this exact signal. Here it
            # is the ordinary case: the model wants another round and the chain
            # is about to give it one. Code that copied Level 4's refusal branch
            # into a chain would refuse the incidents the chain exists to solve.
            return recorded(True, "asked for another round; the next step is it")
        if not run_result.ok or run_result.parsed is None:
            # A vendor failure stops the walk. Two more steps against a provider
            # that is already failing spends real money to arrive at the same
            # refusal, and incidents are exactly when a provider is degraded.
            return replace(
                recorded(False, run_result.failed or "no result"),
                failed=run_result.failed or "no result",
            )

        finding = run_result.parsed
        if not value_supported(finding, calls, results):
            # The citation check, moved one layer earlier than Chapter 7 put it.
            # There it rejected an answer; here it refuses a claim ENTRY into the
            # carried state, so an invented value cannot be re-read as fact by
            # the steps that follow. The step is spent either way - what is
            # saved is every step after it.
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

    return Node(name=step.name, fn=advance)


@dataclass(frozen=True)
class Reasoning:
    """A routed finding, the chain that produced it, and what the chain cost.

    Composes `RoutingDecision` rather than extending it, for the sixth chapter
    running. `exhausted` is this rung's `wanted_more`: the chain ran out of steps
    while the model still had a specific next check to name, which is the shape
    of the Failure Receipt this chapter issues.
    """

    decision: RoutingDecision
    cause: str | None
    evidence: str | None
    next_step: str | None
    steps: tuple[StepRecord, ...]
    calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...]
    ledger: CostLedger = field(default_factory=CostLedger)
    exhausted: bool = False
    # The service the chain concluded, kept alongside the rota it produced.
    # Two services in this book share one rota, so a wrong service does not
    # always reach a wrong pager - and a system that only recorded the rota
    # could not tell a corrected mistake from a mistake that happened to be
    # survivable. `scripts/drift.py` measures against this field for exactly
    # that reason.
    service: str | None = None

    @property
    def routed(self) -> bool:
        return self.decision.routed

    @property
    def api_requests(self) -> int:
        """Requests actually sent to the vendor.

        NOT `ledger.model_calls`, and the difference is worth stating once. The
        ledger meters the seam, and one `invoke` is one measurement covering two
        requests - the round that chooses tools and the round that answers. So
        the ledger counts STEPS and this counts what the Termination Test asks
        about. A chapter that quoted the ledger here would have published a
        maximum of three against a real ceiling of six.
        """
        return 2 * len(self.steps)

    def summary(self) -> str:
        lines = [self.decision.summary()]
        if self.cause is not None:
            lines.append(f"  cause: {self.cause}")
            lines.append(f"  evidence: {self.evidence}")
            lines.append(f"  next step: {self.next_step}")
        elif self.next_step is not None:
            lines.append(f"  ran out of steps still wanting: {self.next_step}")
        return "\n".join(lines)

    def step_table(self) -> str:
        """One line per step, joined to the ledger on the step name.

        The input-token column is the Context Ratchet, printed. It rises at every
        step against a request whose report has not changed, and reading it is
        how a team discovers what their chain is re-buying.
        """
        cost = {m.label: m for m in self.ledger.measurements}
        lines = []
        for record in self.steps:
            measurement = cost.get(f"reasoning.{record.name}")
            tokens_in = measurement.input_tokens if measurement else 0
            tokens_out = measurement.output_tokens if measurement else 0
            millis = measurement.latency_ms if measurement else 0.0
            status = "ok  " if record.ok else "err "
            lines.append(
                f"  {record.name:<9} {status} {len(record.calls):>2} calls  "
                f"{tokens_in:>6} in {tokens_out:>4} out  {millis:8.0f}ms  "
                f"{record.detail}"
            )
        return "\n".join(lines)

    def tool_table(self) -> str:
        """One line per tool call across the whole chain, in the order made."""
        by_id = {result.call_id: result for result in self.results}
        lines = []
        for call in self.calls:
            result = by_id.get(call.call_id)
            status = "err " if result is not None and result.is_error else "ok  "
            head = result.content.splitlines()[0] if result is not None else ""
            lines.append(f"  {call.render():<58} {status} {head}")
        return "\n".join(lines)


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
    """Chapter 3's one-directional severity floor, unchanged for the sixth time."""
    return SEVERITIES[min(SEVERITIES.index(claimed), SEVERITIES.index(tier))]


def reason(
    incident: Incident,
    completer: Completer,
    *,
    steps: tuple[Step, ...] = STEPS,
    chain: Chain | None = None,
    initial: ChainState | None = None,
    tools: tuple[ToolSpec, ...] | None = None,
) -> Reasoning:
    """Walk the chain, then route on what it established.

    `steps` is a parameter so that `scripts/chain_depth.py` can run the same code
    over a shorter chain and measure how much of the corpus each depth answers.
    It is not a runtime knob and it is not an iteration cap: passing a different
    tuple changes which steps exist, and the maximum is still `len(steps)` before
    the walk begins.

    `initial` exists for `scripts/drift.py`, which starts a walk from a state
    carrying a claim the model never made.
    """
    ledger = CostLedger()
    box = Toolbox(incident=incident)
    walker = chain if chain is not None else DEFAULT_CHAIN
    start = initial if initial is not None else ChainState(incident=incident)

    nodes = [
        _step_node(step, index, len(steps), completer, box, ledger, tools)
        for index, step in enumerate(steps)
    ]
    final = walker.walk(nodes, start, done=lambda state: state.finished)

    surviving = final.findings
    wanted = surviving[-1][1].next_step if surviving else None

    if final.settled is None:
        reason_text = final.failed or (
            f"the chain ran out of steps; next check was: {wanted}"
            if wanted
            else "the telemetry does not explain this incident"
        )
        return Reasoning(
            decision=_unrouted(incident, reason_text),
            cause=None,
            evidence=None,
            next_step=None,
            steps=final.records,
            calls=final.calls,
            results=final.results,
            ledger=ledger,
            exhausted=final.failed is None,
        )

    finding = final.settled
    service = owner_for(finding.service)
    if service is None:
        return Reasoning(
            decision=_unrouted(incident, f"no rota owns service {finding.service!r}"),
            cause=None,
            evidence=None,
            next_step=None,
            steps=final.records,
            calls=final.calls,
            results=final.results,
            ledger=ledger,
        )

    severity = _at_least(finding.severity, service.tier)
    deploys = recent_deploys(finding.service)
    return Reasoning(
        decision=RoutingDecision(
            incident_id=incident.incident_id,
            rota=service.rota,
            severity=severity,
            escalate_after_minutes=escalation_after(severity),
            already_overdue=False,
            recent_deploy=(
                f"{deploys[0].deploy_id} {deploys[0].deployed_at}" if deploys else None
            ),
            needs_human=False,
            decided_by=(
                f"step {len(final.records)} of {len(steps)}: "
                f"{finding.evidence_tool} showed {finding.evidence_value}"
            ),
        ),
        cause=finding.cause,
        evidence=f"{finding.evidence_tool} -> {finding.evidence_value}",
        next_step=finding.next_step,
        steps=final.records,
        calls=final.calls,
        results=final.results,
        ledger=ledger,
        service=finding.service,
    )


@register_rung("Level 5: Multi-Step Reasoning")
def run(incident: Incident, completer: Completer | None = None) -> CostLedger:
    """Rung entry point. At most six model calls and three rounds of tools.

    One line, following Chapter 6's declared pattern. The ledger lives inside
    `reason` because the cost of this rung is distributed across steps, and the
    per-step split is the one number this rung can report that Level 4 cannot.
    """
    active: Completer = completer if completer is not None else AnthropicCompleter()
    return reason(incident, active).ledger
