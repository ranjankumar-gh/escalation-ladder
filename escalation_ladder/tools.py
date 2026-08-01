"""Level 4 - the tool-using LLM.

The rung where the model stops asking for values and starts getting them. One
round of tool calls: the model picks from a fixed menu, every result comes back
at once, and the next response is the answer. There is no loop, and no iteration
cap standing in for one.

Three decisions carry the chapter.

`ToolSpec.consequence` has no default, so a tool cannot exist in this book
without someone stating what it can cause. `menu_for` filters writes out unless
asked for them explicitly, which means the set of state changes this rung can
produce is a property of the type system rather than of the prompt. That is the
Blast Radius Schema: prompts change how likely an outcome is, and only the
schema changes whether it is reachable.

The model chooses WHAT to read and never WHEN. No tool takes a timestamp; the
window is bound here, to the incident under investigation. Every field you keep
out of a tool schema is a field a model cannot get wrong, and a free `until`
parameter buys nothing except the chance to read yesterday during an outage.

Tool errors name what was wrong and what was available. A model that asked for a
series that does not exist can correct itself inside the round it is already in,
which is the difference between one wasted call and a refusal.

Measured 2026-07-27 - regenerate with tools/measure_ch07.py in the book repo,
or replay the recordings in tests/ for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field

from escalation_ladder.classify import normalized
from escalation_ladder.fixtures.deploys import recent_deploys
from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.fixtures.logs import LOG_SOURCES, search_logs
from escalation_ladder.fixtures.metrics import (
    NOW,
    _parse,
    known_metrics,
    query_metric,
)
from escalation_ladder.instrument import CostLedger, measured
# Level 4 importing Level 2, which is the direction that is allowed and the one
# `agent.py` has used since Chapter 9. A tool is a capability, and a capability
# the system already has is the cheapest kind to offer. The edge that would be
# fatal is the reverse one - `retrieve` importing `tools` would make Chapter
# 16's descent to Level 2 impossible.
from escalation_ladder.retrieve import Index, default_index, search
from escalation_ladder.llm import (
    AnthropicCompleter,
    Completer,
    MeteredCompleter,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from escalation_ladder.rules import SERVICES, SEVERITIES, RoutingDecision
from escalation_ladder.rules import escalation_after, owner_for
from escalation_ladder.rungs import register_rung

# Every window a tool will accept. An enum rather than an integer, for the same
# reason Chapter 4 put the service list in an enum: it is the only constraint
# enforced while the tokens are being generated. A free integer field invites a
# model to ask for a year of one-minute samples, and the schema that was
# supposed to bound the request would report the violation after the bill.
WINDOWS: tuple[int, ...] = (15, 60, 240, 1440)

_SERVICE_ENUM = sorted(SERVICES)
_METRIC_ENUM = sorted(
    {metric for service in SERVICES for metric in known_metrics(service)}
)


def _object(properties: dict[str, dict], required: list[str]) -> dict:
    """A strict JSON Schema object. `strict: True` needs both of these set."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


QUERY_METRIC = ToolSpec(
    name="query_metric",
    description=(
        "Read a time series for one service over the window ending at the "
        "incident. Returns one sample per minute with timestamps."
    ),
    parameters=_object(
        {
            "service": {"type": "string", "enum": _SERVICE_ENUM},
            "metric": {"type": "string", "enum": _METRIC_ENUM},
            "minutes": {"type": "integer", "enum": list(WINDOWS)},
        },
        ["service", "metric", "minutes"],
    ),
    consequence="read",
)

RECENT_DEPLOYS = ToolSpec(
    name="recent_deploys",
    description="List deploys for one service, most recent first.",
    parameters=_object(
        {"service": {"type": "string", "enum": _SERVICE_ENUM}},
        ["service"],
    ),
    consequence="read",
)

SEARCH_LOGS = ToolSpec(
    name="search_logs",
    description=(
        "Case-insensitive substring search over one source's log lines in the "
        "window ending at the incident. Sources include infrastructure "
        "components as well as services."
    ),
    parameters=_object(
        {
            "source": {"type": "string", "enum": list(LOG_SOURCES)},
            "pattern": {"type": "string"},
            "minutes": {"type": "integer", "enum": list(WINDOWS)},
        },
        ["source", "pattern", "minutes"],
    ),
    consequence="read",
)

# Defined, costed, and then not handed to anything. Writing it out is the point:
# it is three lines longer than `recent_deploys` and identical in shape, which is
# the whole difficulty. A model cannot tell these apart, and neither can a JSON
# schema. `menu_for` can, because `consequence` is a field rather than a
# convention.
ROLLBACK_DEPLOY = ToolSpec(
    name="rollback_deploy",
    description="Roll a service back to a previous deploy.",
    parameters=_object(
        {
            "service": {"type": "string", "enum": _SERVICE_ENUM},
            "deploy_id": {"type": "string"},
        },
        ["service", "deploy_id"],
    ),
    consequence="write",
)

# Added in Chapter 15, and the addition is the demonstration. One entry in this
# tuple reaches Levels 4, 5, and 6 at once, because all three read their menu
# from `menu_for` rather than each holding its own list. It is `consequence:
# read`, so `blast_radius` does not move.
#
# Default OFF, and that is the chapter's second point rather than caution.
# Offering a tool changes the prompt for three rungs, so the code cost of this
# addition is one line and the evaluation cost is three rungs re-scored. A flag
# is what lets those two numbers be paid at different times.
SEARCH_RUNBOOKS = ToolSpec(
    name="search_runbooks",
    description=(
        "Search the runbook and past-incident corpus for a written procedure. "
        "Returns the best-matching passages with citable identifiers."
    ),
    parameters=_object(
        {"query": {"type": "string"}},
        ["query"],
    ),
    consequence="read",
)

TOOLS: tuple[ToolSpec, ...] = (
    QUERY_METRIC,
    RECENT_DEPLOYS,
    SEARCH_LOGS,
    SEARCH_RUNBOOKS,
    ROLLBACK_DEPLOY,
)

TOOL_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOLS}

def menu_for(
    *, allow_writes: bool = False, allow_runbooks: bool = False
) -> tuple[ToolSpec, ...]:
    """The tools a model is offered. Read-only unless a caller says otherwise.

    The default is the security property. A team that adds a write tool gets it
    excluded until somebody writes `allow_writes=True` at a call site, in a diff,
    with a reviewer - rather than getting it exposed the moment it is defined.

    `allow_runbooks` is the same mechanism used for a capability rather than for
    a permission, and it deliberately does NOT generalize into a registry of
    flag names. The write filter reads `consequence`, which is a field every
    tool must fill in; a name-keyed registry would exclude the tools somebody
    remembered to list, which is the convention this seam exists to avoid.
    """
    return tuple(
        spec
        for spec in TOOLS
        if (allow_writes or spec.consequence == "read")
        and (allow_runbooks or spec.name != SEARCH_RUNBOOKS.name)
    )


def blast_radius(tools: tuple[ToolSpec, ...]) -> tuple[str, ...]:
    """Every state change this menu can cause, worst case, in one line each.

    Computable at design time from the menu alone, which is the claim. Run it
    in a test and assert the result, and a tool that widens what the system can
    do fails review instead of shipping.
    """
    return tuple(
        f"{spec.name}: {spec.description.split('.')[0].strip().lower()}"
        for spec in tools
        if spec.consequence == "write"
    )


# The round budget is stated because it is true. A model that plans a second
# round it will never get is being lied to by omission, and the seam discards
# the work it did on that plan.
#
# Both versions of this paragraph were measured, and the pair is the chapter's
# sharpest result. Told there is one round, the model calls 9.4 tools per
# request and never asks again. Told nothing, it calls 4.1 and asks again in 14
# of 15 runs - reading checkout's logs, learning the name of a cluster, and
# wanting to go back and search for it. That second behaviour is dynamic
# decomposition, which is Level 5. At this rung the prompt is not tuning
# anything; it decides which rung you are on.
SYSTEM = """You investigate production incidents for an on-call rota by reading
live telemetry with the tools provided.

You get one round of tools. Every call you make runs together and you receive
every result at once. There is no second round, so choose the set of calls that
will answer the question rather than a first call whose result you plan to build
on. If one round cannot establish the cause, say so rather than guessing.

The services, and what each one owns:
- checkout-api: the checkout flow and order placement
- payment-gateway: card authorization and capture, and the acquirer connection
- notification-worker: transactional email and push delivery
- search-api: search queries and the search index
Log sources also include service-mesh, the sidecar proxy every service calls
through. It owns no rota.

Then report the cause, citing the tool result that shows it.

Rules:
- `evidence_value` must be copied character for character out of a tool result.
  Do not round it, reformat it, or summarize it. If you cannot copy a value that
  demonstrates the cause, set `needs_human` to true.
- `evidence_tool` is the name of the tool whose result you copied from.
- Base the cause on values you read. A cause that the telemetry does not show is
  a guess, however reasonable, and this rung exists to stop guessing.
- If the results do not explain the incident, set `needs_human` to true and say
  what you would read next. Do not report a cause you cannot cite.
- Severity is about customer impact, not tone."""


class Finding(BaseModel):
    """What one round of tools established.

    No `decided_by`, for the reason Chapter 4 gave `Classification` none: this is
    a claim, and `decided_by` belongs on the `RoutingDecision` that acting on the
    claim produces.
    """

    service: Literal[
        "checkout-api",
        "payment-gateway",
        "notification-worker",
        "search-api",
        "unknown",
    ]
    severity: Literal["SEV1", "SEV2", "SEV3"]
    cause: str = Field(description="One sentence. What the telemetry shows.")
    evidence_tool: str = Field(description="The tool whose result you copied from.")
    evidence_value: str = Field(description="Copied verbatim from that result.")
    next_step: str = Field(description="The single next action for the on-call.")
    needs_human: bool


def build_user(incident: Incident) -> str:
    return (
        f"Incident {incident.incident_id}, opened {incident.opened_at}.\n\n"
        "Investigate the report between the markers using the tools, then "
        "report the cause.\n\n"
        f"<report>\n{incident.report}\n</report>"
    )


def value_supported(
    finding: Finding,
    calls: tuple[ToolCall, ...],
    results: tuple[ToolResult, ...],
) -> bool:
    """Did the cited value actually appear in the cited tool's output?

    The third time this book runs the same check and the first time the thing
    being checked is a number the model was never shown twice. Chapter 4 checked
    a quote against a report, Chapter 5 checked one against a retrieved passage,
    and both were catching invention. This one catches arithmetic: a model that
    reads 0.1873 and reports "roughly 19 percent" has said something true and
    uncitable, and a paging path cannot tell that apart from a number it made up.
    """
    if not finding.evidence_value.strip():
        return False
    from_tool = {
        call.call_id for call in calls if call.name == finding.evidence_tool
    }
    haystack = normalized(
        " ".join(
            result.content for result in results if result.call_id in from_tool
        )
    )
    return normalized(finding.evidence_value) in haystack


@dataclass
class Toolbox:
    """Executes tool calls against the incident under investigation.

    The window is bound here rather than in the schema. `until` is
    `opened_at + 30m`, clamped to now, so a query about a two-month-old incident
    reads the two-month-old window and a query about a live one reads the live
    window - without the model ever choosing a timestamp, getting a timezone
    wrong, or being able to ask what checkout looked like last March.
    """

    incident: Incident
    allow_writes: bool = False
    calls: list[ToolCall] = field(default_factory=list)
    _index: Index | None = field(default=None, repr=False, compare=False)

    def _corpus(self) -> Index:
        """Built once per investigation, and never containing this incident."""
        if self._index is None:
            self._index = default_index(self.incident)
        return self._index

    @property
    def until(self) -> str:
        opened = _parse(self.incident.opened_at) + timedelta(minutes=30)
        end = min(opened, NOW)
        return end.isoformat().replace("+00:00", "Z")

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        spec = TOOL_BY_NAME.get(call.name)
        if spec is None:
            return self._error(call, f"unknown tool {call.name!r}")
        if spec.consequence == "write" and not self.allow_writes:
            # Reachable only if a caller passed a write tool in the menu and then
            # refused it here. Belt and braces: the filter is the boundary, this
            # is the assertion that the boundary held.
            return self._error(call, f"{call.name} is not permitted in this session")
        handler = getattr(self, f"_{call.name}", None)
        if handler is None:
            return self._error(call, f"tool {call.name!r} has no implementation")
        return handler(call)

    def _error(self, call: ToolCall, message: str) -> ToolResult:
        return ToolResult(call.call_id, message, is_error=True)

    def _query_metric(self, call: ToolCall) -> ToolResult:
        service = str(call.arguments.get("service", ""))
        metric = str(call.arguments.get("metric", ""))
        minutes = int(call.arguments.get("minutes", 60))
        available = known_metrics(service)
        if metric not in available:
            # The legible error. Naming what exists costs one f-string and turns
            # a dead call into a correctable one inside the same round.
            return self._error(
                call,
                f"unknown metric {metric!r} for {service}; "
                f"available: {', '.join(available)}",
            )
        series = query_metric(service, metric, minutes, ending_at=self.until)
        if not series:
            return self._error(call, f"no samples for {metric} on {service}")
        values = [value for _, value in series]
        body = "\n".join(f"{stamp} {value}" for stamp, value in series[-8:])
        return ToolResult(
            call.call_id,
            f"{metric} on {service}, {minutes}m to {self.until}\n"
            f"min {min(values)} max {max(values)} last {values[-1]}\n"
            f"last 8 samples:\n{body}",
        )

    def _recent_deploys(self, call: ToolCall) -> ToolResult:
        service = str(call.arguments.get("service", ""))
        deploys = recent_deploys(service)
        if not deploys:
            return self._error(
                call,
                f"no deploys on record for {service}; "
                f"known services: {', '.join(sorted(SERVICES))}",
            )
        body = "\n".join(
            f"{d.deploy_id} {d.deployed_at} {d.version} by {d.author}: {d.summary}"
            for d in deploys[:5]
        )
        return ToolResult(call.call_id, f"deploys for {service}, newest first\n{body}")

    def _search_logs(self, call: ToolCall) -> ToolResult:
        source = str(call.arguments.get("source", ""))
        pattern = str(call.arguments.get("pattern", ""))
        minutes = int(call.arguments.get("minutes", 60))
        if source not in LOG_SOURCES:
            return self._error(
                call,
                f"unknown log source {source!r}; "
                f"available: {', '.join(LOG_SOURCES)}",
            )
        lines = search_logs(source, pattern, minutes, ending_at=self.until)
        if not lines:
            return ToolResult(
                call.call_id,
                f"no lines matching {pattern!r} in {source} "
                f"({minutes}m to {self.until})",
            )
        body = "\n".join(line.render() for line in lines)
        return ToolResult(call.call_id, f"{len(lines)} matching lines\n{body}")

    def _search_runbooks(self, call: ToolCall) -> ToolResult:
        query = str(call.arguments.get("query", ""))
        hits = search(self._corpus(), query)
        if not hits:
            # Chapter 5's floor, reaching a tool result unchanged. An empty
            # tuple is how "this corpus does not answer that" leaves `search`,
            # and turning it into a legible error rather than an empty string
            # is what lets the model correct itself inside the round.
            return self._error(
                call, f"no passage scores above the floor for {query!r}"
            )
        body = "\n\n".join(
            f"[{hit.passage.passage_id}] {hit.passage.source} "
            f"(score {hit.score:.2f}, matched {', '.join(hit.matched)})\n"
            f"{hit.passage.text}"
            for hit in hits
        )
        return ToolResult(call.call_id, f"{len(hits)} passages for {query!r}\n{body}")

    def _rollback_deploy(self, call: ToolCall) -> ToolResult:
        raise AssertionError(
            "rollback_deploy reached its implementation. The menu filter is the "
            "boundary and it did not hold."
        )


@dataclass(frozen=True)
class Investigation:
    """A finding backed by tool output, routed, with the round it took.

    Composes `RoutingDecision` the way `GroundedAdvice` did, rather than
    extending it. Four rungs on, Chapter 3's routing shape is still the shape
    every rung returns.
    """

    decision: RoutingDecision
    cause: str | None
    evidence: str | None
    next_step: str | None
    calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...]
    ledger: CostLedger = field(default_factory=CostLedger)
    wanted_more: bool = False

    @property
    def routed(self) -> bool:
        return self.decision.routed

    def summary(self) -> str:
        lines = [self.decision.summary()]
        if self.cause is not None:
            lines.append(f"  cause: {self.cause}")
            lines.append(f"  evidence: {self.evidence}")
            lines.append(f"  next step: {self.next_step}")
        if self.wanted_more:
            lines.append("  the model asked for a second round of tools")
        return "\n".join(lines)

    def tool_table(self) -> str:
        """One line per tool call, joined to the ledger on the call id."""
        cost = {m.label: m for m in self.ledger.measurements}
        by_id = {result.call_id: result for result in self.results}
        lines = []
        for call in self.calls:
            result = by_id.get(call.call_id)
            measurement = cost.get(f"tools.{call.call_id}")
            millis = measurement.latency_ms if measurement is not None else 0.0
            status = "err " if result is not None and result.is_error else "ok  "
            head = result.content.splitlines()[0] if result is not None else ""
            lines.append(f"  {call.render():<52} {status} {millis:6.2f}ms  {head}")
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
    """Chapter 3's one-directional severity floor, unchanged for the fifth time."""
    return SEVERITIES[min(SEVERITIES.index(claimed), SEVERITIES.index(tier))]


def _stop(
    incident: Incident,
    reason: str,
    box: Toolbox,
    results: tuple[ToolResult, ...],
    ledger: CostLedger,
    *,
    wanted_more: bool = False,
) -> Investigation:
    return Investigation(
        decision=_unrouted(incident, reason),
        cause=None,
        evidence=None,
        next_step=None,
        calls=tuple(box.calls),
        results=results,
        ledger=ledger,
        wanted_more=wanted_more,
    )


def investigate(
    incident: Incident,
    completer: Completer,
    *,
    allow_writes: bool = False,
    tools: tuple[ToolSpec, ...] | None = None,
) -> Investigation:
    """Read live telemetry in one round, then route on what it showed.

    `tools` was added in Chapter 15 and is the generalization of `allow_writes`,
    not a second flag beside it. A rung that takes a boolean per kind of tool
    has to be edited for every new kind; a rung that takes a menu never has to
    be edited again. The default reproduces Chapter 7 exactly.
    """
    ledger = CostLedger()
    box = Toolbox(incident=incident, allow_writes=allow_writes)
    metered = MeteredCompleter(inner=completer, ledger=ledger, stage="tools.model")

    def execute(call: ToolCall) -> ToolResult:
        billed = measured(f"tools.{call.call_id}", ledger)(box.execute)
        return billed(call)

    run_result = metered.invoke(
        system=SYSTEM,
        user=build_user(incident),
        tools=menu_for(allow_writes=allow_writes) if tools is None else tools,
        execute=execute,
        schema=Finding,
    )
    results = run_result.results

    if run_result.wanted_more:
        return _stop(
            incident,
            "one round of tools was not enough",
            box,
            results,
            ledger,
            wanted_more=True,
        )
    if not run_result.ok or run_result.parsed is None:
        return _stop(
            incident, run_result.failed or "no result", box, results, ledger
        )

    finding = run_result.parsed
    if finding.needs_human or finding.service == "unknown":
        return _stop(incident, "the telemetry does not explain this incident",
                     box, results, ledger)
    if not value_supported(finding, tuple(box.calls), results):
        return _stop(
            incident,
            f"evidence not found in {finding.evidence_tool} output: "
            f"{finding.evidence_value!r}",
            box,
            results,
            ledger,
        )

    service = owner_for(finding.service)
    if service is None:
        return _stop(
            incident, f"no rota owns service {finding.service!r}", box, results, ledger
        )

    severity = _at_least(finding.severity, service.tier)
    deploys = recent_deploys(finding.service)
    return Investigation(
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
            decided_by=f"{finding.evidence_tool} showed {finding.evidence_value}",
        ),
        cause=finding.cause,
        evidence=f"{finding.evidence_tool} -> {finding.evidence_value}",
        next_step=finding.next_step,
        calls=tuple(box.calls),
        results=results,
        ledger=ledger,
    )


@register_rung("Level 4: Tool-Using LLM")
def run(incident: Incident, completer: Completer | None = None) -> CostLedger:
    """Rung entry point. Two measured model calls and one round of tools.

    One line, following Chapter 6's declared pattern rather than Chapters 3 to
    5': the ledger lives inside `investigate`, because the cost of this rung is
    distributed across a set of calls the model chose and a request-level total
    could not say which tool was expensive.
    """
    active: Completer = completer if completer is not None else AnthropicCompleter()
    return investigate(incident, active).ledger
