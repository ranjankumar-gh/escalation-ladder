"""Level 3 - the LLM workflow.

Five stages in an order this module fixes: classify, extract, retrieve, draft,
route. Three of them are ordinary code. Two of them call a model. That ratio is
the chapter's argument rather than an implementation detail - the Floor Test does
not retire at Level 0, it runs on every stage, and most stages fail it.

The capability this rung adds over Chapter 5 is exactly one thing: a later stage
whose input is an earlier stage's output. Chapter 5 could not build its query
from anything that had read the report, so INC-1045 retrieved nothing while its
runbook sat in the corpus. Stage 2 here is that missing hop, and it costs no
model call at all.

Three decisions carry the chapter.

Every stage records what it produced and which stage produced its input. A
refusal therefore has two names: the stage that reported it and the stage that
caused it. Without the second one, every counter in a pipeline points at the last
stage able to detect a fault, which is almost never the stage that created it.
`origin_of` walks back through the deterministic stages, because a stage that
passed the Floor Test is a pure function of its input and can be exonerated by
its unit tests. The stages you made deterministic are the stages you can rule
out, which is a second reason to have as many of them as you can.

When retrieval finds nothing, the pipeline does not refuse. It falls back to the
Level 1 answer it is already holding - the classification from stage 1, routed
through Chapter 3's tables - which costs zero extra tokens and beats a refusal.
That fallback is only available because the cheap paths were never deleted.

Cost is measured inside the pipeline rather than around it, because at this rung
the cost is distributed across stages and a single number for the request hides
which stage spent it. `MeteredCompleter` wraps Chapter 4's seam instead of
wrapping each stage, so `classify.classify` becomes measurable here without
being edited there.

Measured 2026-07-26 - regenerate with tools/measure_ch06.py in the book repo,
or replay the recordings in tests/ for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pydantic import BaseModel

from escalation_ladder.classify import Classification, classify, decide
from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.instrument import CostLedger, measured
# MeteredCompleter moved to llm.py in Chapter 7, where the protocol it
# implements lives and where Level 4 can reach it without importing Level 3.
# Re-exported here so callers written against Chapter 6 keep working.
from escalation_ladder.llm import (
    AnthropicCompleter,
    Completer,
    Completion,
    MeteredCompleter,
)
from escalation_ladder.retrieve import (
    Grounded,
    GroundedAdvice,
    Hit,
    Index,
    citation_supported,
    context_block,
    default_index,
    search,
)
from escalation_ladder.rules import (
    SEVERITIES,
    RoutingDecision,
    escalation_after,
    owner_for,
)
from escalation_ladder.rungs import register_rung

T = TypeVar("T", bound=BaseModel)

StageKind = Literal["model", "code"]

# The sequence, and what each stage costs. This tuple IS the architecture. A rung
# whose stages you can enumerate at import time is a rung whose steps you can
# draw, which is Chapter 2's Level 3 gate. Level 5 is where this stops being
# knowable before the run starts.
STAGES: tuple[tuple[str, StageKind], ...] = (
    ("classify", "model"),
    ("extract", "code"),
    ("retrieve", "code"),
    ("draft", "model"),
    ("route", "code"),
)

STAGE_KIND: dict[str, StageKind] = dict(STAGES)

# Which stage hands each stage its input. `None` means the stage reads the
# request itself and has nobody upstream to blame.
STAGE_INPUT: dict[str, str | None] = {
    "classify": None,
    "extract": "classify",
    "retrieve": "extract",
    "draft": "retrieve",
    "route": "draft",
}


@dataclass(frozen=True)
class StageRecord:
    """What one stage did, and where its input came from.

    `detail` is this module's `decided_by`, one per stage instead of one per
    request. A pipeline that records only its final answer can tell you that it
    refused and never which of five things went wrong.

    Cost is deliberately not here. The trace says what happened, the ledger says
    what it cost, and the two join on the stage name - so a stage can be timed
    without the record of its decision growing a latency field nobody reads.
    """

    name: str
    kind: StageKind
    ok: bool
    detail: str

    @property
    def input_from(self) -> str | None:
        return STAGE_INPUT[self.name]


def origin_of(
    trace: tuple[StageRecord, ...], reporting: str, *, blames_input: bool
) -> str:
    """The nearest upstream stage that could actually have been wrong.

    A stage fails in one of two ways and they must not share a counter. Its own
    machinery can break - the vendor is down, it invented a citation - in which
    case it is the origin and `blames_input` is false. Or it can refuse what it
    was handed, in which case the fault was created somewhere upstream and this
    walk finds where.

    The walk skips deterministic stages and stops at the first one that calls a
    model. A stage that passed the Floor Test is a pure function of its input, so
    given a green unit test it did not invent the problem - it inherited it. The
    stages you made deterministic are the stages you can exonerate, which is a
    second reason to have as many of them as you can.

    One hop of judgement, no model, no heuristics. It gets you from "the drafter
    refuses a third of the time" to "the drafter is handed nothing usable a third
    of the time", which is a different quarter's work. Going past the nearest
    probabilistic stage needs Chapter 5's Substitution Test, and that one stays
    manual on purpose - and neither this walk nor that test can name a fault that
    belongs to no stage at all, such as a corpus that simply does not contain the
    answer.
    """
    if not blames_input:
        return reporting
    by_name = {record.name: record for record in trace}
    current = by_name.get(reporting)
    while current is not None:
        upstream_name = current.input_from
        if upstream_name is None:
            return current.name
        upstream = by_name.get(upstream_name)
        if upstream is None:
            return upstream_name
        if upstream.kind == "model":
            return upstream.name
        current = upstream
    return reporting


@dataclass(frozen=True)
class TracedAdvice:
    """Chapter 5's advice, plus the trace that says which stage is responsible.

    Composes `GroundedAdvice` rather than replacing it, for the reason
    `GroundedAdvice` composed `RoutingDecision`: the rungs below still run, and
    Chapter 11 needs every shape in this chain to survive.

    `reported_by` and `origin` are the same string when a stage broke on its own
    and differ exactly when it broke on something handed to it. That divergence
    is the number worth putting on a dashboard.
    """

    advice: GroundedAdvice
    trace: tuple[StageRecord, ...]
    ledger: CostLedger = field(default_factory=CostLedger)
    reported_by: str | None = None
    origin: str | None = None
    degraded: bool = False

    @property
    def decision(self) -> RoutingDecision:
        return self.advice.decision

    @property
    def routed(self) -> bool:
        return self.advice.decision.routed

    @property
    def grounded(self) -> bool:
        return self.advice.grounded

    def summary(self) -> str:
        lines = [self.advice.summary()]
        if self.degraded:
            lines.append("  degraded: routed from the classification, no runbook step")
        if self.reported_by is not None:
            lines.append(f"  reported by: {self.reported_by}")
            lines.append(f"  origin: {self.origin}")
        return "\n".join(lines)

    def stage_table(self) -> str:
        """The trace joined to the ledger, one line per stage that ran."""
        cost = {m.label: m for m in self.ledger.measurements}
        lines = []
        for record in self.trace:
            measurement = cost.get(f"workflow.{record.name}")
            millis = measurement.latency_ms if measurement is not None else 0.0
            tokens = (
                measurement.input_tokens + measurement.output_tokens
                if measurement is not None
                else 0
            )
            lines.append(
                f"  {record.name:<9} {record.kind:<5} "
                f"{'ok  ' if record.ok else 'stop'} "
                f"{millis:8.1f}ms {tokens:>5} tok  {record.detail}"
            )
        return "\n".join(lines)


DRAFT_SYSTEM = """You triage production incident reports for an on-call rota,
using runbook and past-incident passages retrieved for the report.

An earlier stage has already classified this report. Its classification is given
to you, and IT MAY BE WRONG. The passages were retrieved using that
classification, so if the classification was wrong the passages are about the
wrong thing.

Return the owning service, the severity, the id of the ONE passage that decides
the next step, a verbatim quote from that passage, and the single next action the
on-call engineer should take.

The services, and what each one owns:
- checkout-api: the checkout flow and order placement
- payment-gateway: card authorization and capture, and the acquirer connection
- notification-worker: transactional email and push delivery
- search-api: search queries and the search index

Rules:
- Judge the passages against the REPORT, not against the classification. If the
  passages and the report contradict the earlier classification, return the
  service they support rather than the one you were given.
- Set `needs_human` to true when no retrieved passage addresses this incident,
  including when a passage is about the right service but the wrong failure. A
  passage about the right service and the wrong failure is not an answer.
- `passage_id` must be copied from the passage markers. `quote` must be copied
  character for character from that same passage.
- `next_step` must be an action named in the cited passage. Do not supply a step
  from your own knowledge, however reasonable that step is.
- Severity is about customer impact, not tone."""


def extract(claim: Classification, incident: Incident) -> str:
    """Stage 2. Build the retrieval query by ADDING the classification to the report.

    One field and an f-string, and the shape of it is measured rather than
    chosen. Replacing the report with the classification - service plus summary,
    which is what Chapter 5's Failure Receipt demonstrated - scores exactly the
    same context recall as Chapter 5's raw report: it fixes INC-1045 and breaks
    INC-1044, because a classification is a lossy summary and the report is the
    only lossless copy of the reporter's own words. Prepending the service name
    instead keeps both. See `scripts/compare_queries.py`.

    The reflex here is a model call - a "query rewriter" - because the stage sits
    between two model calls and looks like it should be one. Run the Floor Test
    on it instead. The correct output is fully specified by the classification
    and the report, a unit test says so, and the comparison script measures what
    the model-written alternative buys.
    """
    return f"{claim.service} {incident.report}"


def build_user(
    incident: Incident, claim: Classification, hits: tuple[Hit, ...]
) -> str:
    """Stage 4's user turn: passages, then the earlier stage's claim, then the report.

    The claim is labeled as an earlier stage's output rather than presented as
    fact. A stage handed an upstream answer with no provenance has no grounds to
    disagree with it, and a pipeline whose stages cannot disagree can only
    compound.
    """
    return (
        "Retrieved passages:\n\n"
        f"{context_block(hits)}\n\n"
        "An earlier stage classified this report as:\n"
        f"  service: {claim.service}\n"
        f"  severity: {claim.severity}\n"
        f"  summary: {claim.summary}\n\n"
        "Classify the incident report between the markers and give the next "
        "step, citing one passage above.\n\n"
        f"<report>\n{incident.report}\n</report>"
    )


def _at_least(claimed: str, tier: str) -> str:
    """Chapter 3's one-directional severity floor, unchanged for the fourth time."""
    return SEVERITIES[min(SEVERITIES.index(claimed), SEVERITIES.index(tier))]


def _unrouted(incident: Incident, reason: str) -> GroundedAdvice:
    """Chapter 3's refusal shape, carried up one more rung."""
    return GroundedAdvice(
        decision=RoutingDecision(
            incident_id=incident.incident_id,
            rota=None,
            severity=None,
            escalate_after_minutes=None,
            already_overdue=False,
            recent_deploy=None,
            needs_human=True,
            decided_by=reason,
        ),
        next_step=None,
        citation=None,
        quote=None,
    )


def _stop(
    incident: Incident,
    trace: list[StageRecord],
    ledger: CostLedger,
    stage: str,
    reason: str,
    *,
    blames_input: bool,
) -> TracedAdvice:
    """Refuse, naming both the stage that noticed and the stage responsible."""
    frozen = tuple(trace)
    return TracedAdvice(
        advice=_unrouted(incident, reason),
        trace=frozen,
        ledger=ledger,
        reported_by=stage,
        origin=origin_of(frozen, stage, blames_input=blames_input),
    )


def advise(
    incident: Incident,
    completer: Completer,
    index: Index | None = None,
) -> TracedAdvice:
    """Run the five stages in order, or stop and say which stage stopped it.

    Same name and same contract as `retrieve.advise`, deliberately. A rung that
    can be swapped for the rung below without changing its caller is a rung
    Chapter 16 can walk back down.
    """
    trace: list[StageRecord] = []
    ledger = CostLedger()
    metered = MeteredCompleter(
        inner=completer, ledger=ledger, stage="workflow.classify"
    )

    claim = classify(incident, metered)
    classified = not (claim.needs_human or claim.service == "unknown")
    trace.append(
        StageRecord(
            "classify",
            "model",
            classified,
            f"{claim.service} {claim.severity}" if classified else claim.summary,
        )
    )
    if not classified:
        return _stop(
            incident,
            trace,
            ledger,
            "classify",
            claim.summary or "no classification",
            blames_input=False,
        )

    @measured("workflow.extract", ledger)
    def _extract() -> str:
        return extract(claim, incident)

    query = _extract()
    trace.append(StageRecord("extract", "code", True, f"query {query!r}"))

    active = index if index is not None else default_index(incident)

    @measured("workflow.retrieve", ledger)
    def _search() -> tuple[Hit, ...]:
        return search(active, query)

    hits = _search()
    trace.append(
        StageRecord(
            "retrieve",
            "code",
            bool(hits),
            ", ".join(f"{hit.passage.passage_id} {hit.score:.2f}" for hit in hits)
            if hits
            else "no passage cleared the relevance floor",
        )
    )
    if not hits:
        # Degrade rather than refuse. The classification was paid for at stage 1
        # and routes on its own, so refusing here would throw away a working
        # answer from the rung below in order to report the failure of the rung
        # above - the wrong trade on a paging path.
        fallback = decide(incident, claim)
        if not fallback.routed:
            return _stop(
                incident,
                trace,
                ledger,
                "retrieve",
                "no passage cleared the relevance floor",
                blames_input=True,
            )
        frozen = tuple(trace)
        return TracedAdvice(
            advice=GroundedAdvice(
                decision=fallback, next_step=None, citation=None, quote=None
            ),
            trace=frozen,
            ledger=ledger,
            # Attributed even though it routed. A degraded answer is a
            # reduced-quality outcome with a cause, and "retrieval found nothing"
            # is what was noticed rather than what happened. Recording only the
            # site here would put the one path with no second opinion on it into
            # the one bucket nobody reviews.
            reported_by="retrieve",
            origin=origin_of(frozen, "retrieve", blames_input=True),
            degraded=True,
        )

    metered.stage = "workflow.draft"
    completion = metered.parse(
        system=DRAFT_SYSTEM,
        user=build_user(incident, claim, hits),
        schema=Grounded,
    )
    if not completion.ok or completion.parsed is None:
        reason = completion.failed or "no result"
        trace.append(StageRecord("draft", "model", False, reason))
        return _stop(incident, trace, ledger, "draft", reason, blames_input=False)

    drafted = completion.parsed
    if drafted.needs_human or drafted.service == "unknown":
        reason = "the retrieved passages do not address this incident"
        trace.append(StageRecord("draft", "model", False, reason))
        return _stop(incident, trace, ledger, "draft", reason, blames_input=True)
    if not citation_supported(drafted, hits):
        reason = f"citation not supported: {drafted.passage_id!r}"
        trace.append(StageRecord("draft", "model", False, reason))
        return _stop(incident, trace, ledger, "draft", reason, blames_input=False)

    corrected = drafted.service != claim.service
    trace.append(
        StageRecord(
            "draft",
            "model",
            True,
            f"cites {drafted.passage_id}"
            + (f", corrected service to {drafted.service}" if corrected else ""),
        )
    )

    @measured("workflow.route", ledger)
    def _route() -> RoutingDecision | None:
        service = owner_for(drafted.service)
        if service is None:
            return None
        severity = _at_least(drafted.severity, service.tier)
        return RoutingDecision(
            incident_id=incident.incident_id,
            rota=service.rota,
            severity=severity,
            escalate_after_minutes=escalation_after(severity),
            already_overdue=False,
            recent_deploy=None,
            needs_human=False,
            decided_by=f"grounded in {drafted.passage_id}, 5 stages",
        )

    routed = _route()
    if routed is None:
        reason = f"no rota owns service {drafted.service!r}"
        trace.append(StageRecord("route", "code", False, reason))
        return _stop(incident, trace, ledger, "route", reason, blames_input=True)

    trace.append(
        StageRecord("route", "code", True, f"{routed.rota} {routed.severity}")
    )
    return TracedAdvice(
        advice=GroundedAdvice(
            decision=routed,
            next_step=drafted.next_step,
            citation=drafted.passage_id,
            quote=drafted.quote,
        ),
        trace=tuple(trace),
        ledger=ledger,
    )


@register_rung("Level 3: LLM Workflow")
def run(incident: Incident, completer: Completer | None = None) -> CostLedger:
    """Rung entry point. Up to two measured model calls, one per model stage.

    A departure from Chapters 3 to 5, where `run` re-walked the rung's flow with
    the measurement wrapped around it. At Level 3 the cost is distributed across
    stages, so the ledger has to live inside `advise` - and a `run` that
    duplicated a five-stage sequence to measure it would eventually measure a
    sequence the rung no longer runs.
    """
    active: Completer = completer if completer is not None else AnthropicCompleter()
    return advise(incident, active).ledger
