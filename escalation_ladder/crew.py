"""Level 7 - three roles, two handoffs, and one authority boundary.

Chapter 9 closed by issuing a receipt with a deliberately narrow scope, and this
module is built to that scope and no wider:

    A single loop has exactly one tool menu, so it has exactly one blast radius,
    and it holds that authority in every round including the ones it later
    retracts. Investigating and acting cannot be separated inside one agent,
    because there is no round the architecture distinguishes.

That earns a separate authority boundary. It does not earn roles improving
quality, it does not earn three agents being smarter than one, and it does not
earn a reviewer catching things the investigator missed. Every one of those is a
claim about outcomes, and this book's rule is that a claim about outcomes is
settled by measurement or it is not kept. So this module makes exactly one thing
true by construction and leaves the rest to `scripts/specialization.py`.

WHAT IS TRUE BY CONSTRUCTION. `MENUS` gives each role its own tool list, and
`blast_radius` over those lists is a three-line assertion in `test_crew.py`. The
investigator can read everything and change nothing. The reviewer holds no tools
at all. The remediator can change one thing and read nothing - which is the half
of the boundary nobody builds, and the more important half. An actor that cannot
gather its own evidence cannot talk itself into acting; it can only act on what
somebody handed it, and refusing is always available.

WHAT IS NOT. Whether splitting the work this way produces better answers than
Chapter 9's single loop is not knowable from this file. It needs an A/B against
a labeled outcome set, and Chapter 10 computes what that experiment has to beat
before it argues about whether to run it.

THE SHAPE, and it is worth naming plainly because the name on the tin obscures
it: three sequential roles are Chapter 6's workflow with a Level 6 agent at each
stage. There is no fan-out here, no concurrency, and nothing for a state channel
to merge - a review cannot start before the investigation it reviews. That is
why `orchestration.Fanout` exists, is exercised, and is not used by the default
path. Chapter 10's deep dive is where that decision is argued.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from escalation_ladder.agent import Budget, DryRunToolbox, Pursuit, Verdict, pursue
from escalation_ladder.classify import normalized
from escalation_ladder.fixtures.deploys import recent_deploys
from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.instrument import CostLedger, measured
from escalation_ladder.llm import (
    AnthropicCompleter,
    Completer,
    MeteredCompleter,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from escalation_ladder.rules import RoutingDecision, escalation_after, owner_for
from escalation_ladder.rungs import register_rung
from escalation_ladder.tools import TOOLS, Toolbox, blast_radius

Role = Literal["investigator", "reviewer", "remediator"]

ROLES: tuple[Role, ...] = ("investigator", "reviewer", "remediator")


# The whole authority boundary, as data. Built by FILTERING `tools.TOOLS` on
# `consequence` rather than by listing tool names, so a tool added in a later
# chapter lands in a role's menu because of what it does rather than because
# somebody remembered to add it in two places.
#
# Read the reviewer's entry twice. An empty menu is not an oversight and it is
# not thrift: a reviewer that can read is a second investigator, and the thing it
# would then be reviewing is its own reading. The reviewer's whole value is that
# it holds a different position on the same evidence, and a tool call is how it
# would leave that position.
MENUS: dict[Role, tuple[ToolSpec, ...]] = {
    "investigator": tuple(s for s in TOOLS if s.consequence == "read"),
    "reviewer": (),
    "remediator": tuple(s for s in TOOLS if s.consequence == "write"),
}


def menu_for_role(role: Role) -> tuple[ToolSpec, ...]:
    """The tools one role is offered. There is no argument that widens this.

    Chapter 7's `menu_for` takes `allow_writes` because one agent held one menu
    and a caller had to be able to say which. Here the question is answered by
    which role is being built, so the parameter would be a way to get it wrong.
    """
    return MENUS[role]


# What the reviewer is given about the investigation, and the only knob in this
# module that changes the cost of the rung rather than its behavior.
#
# "results" hands over every tool result the investigator collected. "claim"
# hands over the verdict alone. Both are real designs shipped by real teams, the
# second is roughly free, and Chapter 10 measures the gap between them because
# the gap IS the coordination cost. A reviewer given the claim alone is checking
# the investigator's conclusion against the investigator's own selection of
# evidence, which is a consistency check wearing a veto's clothes.
Carry = Literal["results", "claim"]
CARRY_DEFAULT: Carry = "results"


REVIEW_SYSTEM = """You review a proposed on-call page before it is sent.

An investigator read live telemetry and reached a conclusion. You have the
original incident report, the investigator's conclusion, and the raw tool output
it worked from. You have no tools: you cannot read anything further, and you are
not being asked to investigate.

You are being asked one question: does the evidence support this conclusion?

The investigator is fallible and its conclusion is a claim, not a fact. Judge it
against the ORIGINAL report and the raw tool output rather than against how
confident it sounds. Reject it when the cited value does not appear in the
output, when the output shows a different service failing, or when the report
describes symptoms the conclusion does not account for.

Endorsing a wrong page and rejecting a right one are both errors. Say which you
are less sure of in `reason`.
"""


REMEDIATE_SYSTEM = """You decide whether to roll back a deploy.

A reviewed conclusion is below, along with the deploys currently on that service.
You have exactly one tool and it changes production. You cannot read metrics,
logs, or anything else; if you want evidence you did not receive, the answer is
`none` and a sentence saying what you would have read.

Roll back only when the reviewed conclusion names a deploy as the cause AND that
deploy is in the list. A rollback that is merely plausible is a second incident.
"""


class Review(BaseModel):
    """The reviewer's judgment. A claim about a claim, so no `decided_by`.

    `quoted` is Chapter 4's checkable field, one handoff further along: the
    reviewer must copy the line it judged from, so an endorsement that cites
    nothing present in what it was given fails a check rather than a taste test.
    """

    endorsed: bool
    reason: str = Field(description="One sentence. Why it holds, or why it does not.")
    quoted: str = Field(
        description=(
            "Copied verbatim from the report or the tool output you were given. "
            "The line your judgment rests on."
        )
    )


class Remediation(BaseModel):
    """What the remediator decided, beside whatever tool call it made.

    Two records of one decision, on purpose. `action` is what the model says it
    did; `Resolution.proposed` is what the seam saw it do. A run where those
    disagree is the finding, and a schema that made the disagreement
    unrepresentable would have hidden it.
    """

    action: Literal["rollback", "none"]
    deploy_id: str = Field(description="The deploy rolled back, or an empty string.")
    reason: str = Field(description="One sentence.")


@dataclass(frozen=True)
class Handoff:
    """One role handing work to the next, and the text that crossed.

    This exists so coordination cost is a measured object rather than an
    inference. `carried` is the exact user message the receiving role was sent,
    so `scripts/crew_cost.py` prices it with `count_tokens` and the Break-Even
    Test is arithmetic over real strings rather than an estimate.
    """

    sender: Role
    receiver: Role
    carried: str

    @property
    def characters(self) -> int:
        return len(self.carried)


@dataclass(frozen=True)
class Resolution:
    """What the crew produced, what each role contributed, and what it cost.

    Composes `RoutingDecision` for the eighth chapter running. `blocked_at` is
    the field this rung adds and no rung below could have: the name of the role
    that stopped the work. A single agent that refuses has one refusal site, so
    naming it would say nothing; three roles that can each refuse produce three
    different problems, and Chapter 5 already refused to collapse four refusal
    reasons into one counter.
    """

    decision: RoutingDecision
    investigation: Pursuit
    review: Review | None = None
    remediation: Remediation | None = None
    proposed: tuple[ToolCall, ...] = ()
    handoffs: tuple[Handoff, ...] = ()
    ledger: CostLedger = field(default_factory=CostLedger)
    blocked_at: Role | None = None
    carry: Carry = CARRY_DEFAULT

    @property
    def routed(self) -> bool:
        return self.decision.routed

    @property
    def api_requests(self) -> int:
        """Requests sent to the vendor, summed across every role.

        Chapter 8 named this trap and Chapter 9 inherited it: `MeteredCompleter`
        wraps the seam, so one `invoke` is one measurement covering two requests.
        The investigator's count is already correct in `Pursuit.api_requests`;
        the reviewer is a single `parse` and therefore one request, and the
        remediator is an `invoke` and therefore two when it runs.
        """
        total = self.investigation.api_requests
        if self.review is not None:
            total += 1
        if self.remediation is not None:
            total += 2
        return total

    @property
    def roles_run(self) -> tuple[Role, ...]:
        active: list[Role] = ["investigator"]
        if self.review is not None:
            active.append("reviewer")
        if self.remediation is not None:
            active.append("remediator")
        return tuple(active)

    def summary(self) -> str:
        if not self.routed:
            return f"{self.decision.incident_id}: {self.decision.decided_by}"
        return (
            f"{self.decision.incident_id} -> {self.decision.rota} "
            f"({self.decision.severity}); {self.decision.decided_by}"
        )

    def handoff_table(self) -> str:
        """One row per handoff, in characters. Tokens need a key; this does not.

        Deliberately reported in characters rather than tokens. A token count is
        the number the cost argument needs and it costs a network round trip to
        obtain; a character count is free, is available in a unit test, and moves
        in the same direction. `scripts/crew_cost.py` converts.
        """
        rows = ["| From | To | Characters |", "|:--|:--|--:|"]
        for hop in self.handoffs:
            rows.append(f"| {hop.sender} | {hop.receiver} | {hop.characters:,} |")
        return "\n".join(rows)


def _unrouted(incident: Incident, reason: str, blocked_at: Role) -> RoutingDecision:
    """Refuse, and name the role that refused. Chapter 3's shape, ninth chapter.

    The reason is prefixed with the role because at this rung "needs_human" is
    the least informative thing the system knows. Three roles refuse for three
    different reasons and the fixes are three different pieces of work.
    """
    return RoutingDecision(
        incident_id=incident.incident_id,
        rota=None,
        severity=None,
        escalate_after_minutes=None,
        already_overdue=False,
        recent_deploy=None,
        needs_human=True,
        decided_by=f"{blocked_at}: {reason}",
    )


def review_user(
    incident: Incident, verdict: Verdict, pursuit: Pursuit, *, carry: Carry
) -> str:
    """The reviewer's prompt, and the one place this rung's cost is decided.

    The original report goes in first and goes in verbatim. Chapter 6 measured
    what removing it costs, Chapter 8 separated the Downstream Veto's two
    prerequisites and found this structural one was the only half carrying
    weight, and here it is the difference between a reviewer and a rubber stamp:
    it is the only thing in this prompt the investigator did not choose.
    """
    parts = [
        f"Incident {incident.incident_id}, opened {incident.opened_at}.",
        f"<report>\n{incident.report}\n</report>",
        (
            "<conclusion>\n"
            f"service: {verdict.service}\n"
            f"severity: {verdict.severity}\n"
            f"cause: {verdict.cause}\n"
            f"cited from {verdict.evidence_tool}: {verdict.evidence_value}\n"
            f"proposed next step: {verdict.next_step}\n"
            "</conclusion>"
        ),
    ]
    if carry == "results":
        body = "\n".join(
            f"[{result.call_id}] {result.content}" for result in pursuit.results
        )
        parts.append(f"<tool-output>\n{body}\n</tool-output>")
    return "\n\n".join(parts)


def remediate_user(verdict: Verdict, review: Review) -> str:
    """The remediator's prompt: a reviewed conclusion, and a list from code.

    The deploy list is read by `recent_deploys`, which is Chapter 3's
    deterministic floor and passes the Floor Test outright. Chapter 6 established
    that the Floor Test runs per stage rather than once per system; this is the
    same finding at the point where it matters most, because the fact the actor
    needs in order to act correctly is a lookup, and a lookup does not have to be
    a tool a model is trusted to call.
    """
    deploys = recent_deploys(verdict.service)
    listed = (
        "\n".join(f"- {d.deploy_id} deployed {d.deployed_at}" for d in deploys)
        or "- none in the window"
    )
    return (
        "<reviewed-conclusion>\n"
        f"service: {verdict.service}\n"
        f"cause: {verdict.cause}\n"
        f"reviewer: {review.reason}\n"
        f"reviewer quoted: {review.quoted}\n"
        "</reviewed-conclusion>\n\n"
        f"<deploys service=\"{verdict.service}\">\n{listed}\n</deploys>"
    )


def quote_supported(review: Review, carried: str) -> bool:
    """Did the reviewer's quoted line appear in what the reviewer was given?

    The fourth outing for this check and the first time it guards a JUDGMENT
    rather than a claim. Chapter 4 checked a quote against a report, Chapter 5
    against a retrieved passage, Chapter 8 against a tool result. Here it catches
    a reviewer that endorsed a page by quoting a line nobody sent it - which is
    the specific way a role with no tools invents evidence, because inventing is
    the only way it can produce any.
    """
    if not review.quoted.strip():
        return False
    return normalized(review.quoted) in normalized(carried)


def resolve(
    incident: Incident,
    completer: Completer,
    *,
    budget: Budget | None = None,
    carry: Carry = CARRY_DEFAULT,
    review_enabled: bool = True,
) -> Resolution:
    """Investigate, review, and - only if both cleared - propose a remediation.

    `review_enabled` exists for one experiment: it removes the middle role and
    leaves the boundary, so the cost of reviewing is separable from the cost of
    splitting authority. Teams buy them as one purchase and they are two.

    There is no `allow_writes`. Chapter 7 needed one because a single agent held a
    single menu and a caller had to decide what went in it; here the write menu
    belongs to a role, and a flag that took it away would be deleting the
    remediator rather than configuring it. The rollback is never executed for a
    different reason - the toolbox is a `DryRunToolbox` - and that is a property
    of this book's fixtures rather than of the architecture, which is exactly
    what makes the blast radius worth asserting in a test.
    """
    ledger = CostLedger()
    handoffs: list[Handoff] = []

    investigation = pursue(
        incident,
        MeteredCompleter(inner=completer, ledger=ledger, stage="crew.investigator"),
        budget=budget,
    )
    settled = _settled_verdict(investigation)
    if settled is None:
        return Resolution(
            decision=investigation.decision,
            investigation=investigation,
            handoffs=(),
            ledger=ledger,
            blocked_at="investigator",
            carry=carry,
        )

    if not review_enabled:
        return _page(incident, investigation, settled, None, ledger, (), carry)

    carried = review_user(incident, settled, investigation, carry=carry)
    handoffs.append(Handoff("investigator", "reviewer", carried))

    metered = MeteredCompleter(inner=completer, ledger=ledger, stage="crew.reviewer")
    judged = metered.parse(system=REVIEW_SYSTEM, user=carried, schema=Review)
    if not judged.ok or judged.parsed is None:
        return Resolution(
            decision=_unrouted(
                incident, judged.failed or "no result", "reviewer"
            ),
            investigation=investigation,
            handoffs=tuple(handoffs),
            ledger=ledger,
            blocked_at="reviewer",
            carry=carry,
        )

    review = judged.parsed
    if not quote_supported(review, carried):
        return Resolution(
            decision=_unrouted(
                incident,
                f"quoted {review.quoted!r}, which is not in what it was sent",
                "reviewer",
            ),
            investigation=investigation,
            review=review,
            handoffs=tuple(handoffs),
            ledger=ledger,
            blocked_at="reviewer",
            carry=carry,
        )
    if not review.endorsed:
        return Resolution(
            decision=_unrouted(incident, review.reason, "reviewer"),
            investigation=investigation,
            review=review,
            handoffs=tuple(handoffs),
            ledger=ledger,
            blocked_at="reviewer",
            carry=carry,
        )

    remediation, proposed = _remediate(
        incident, settled, review, completer, ledger, handoffs
    )
    return _page(
        incident,
        investigation,
        settled,
        review,
        ledger,
        tuple(handoffs),
        carry,
        remediation=remediation,
        proposed=proposed,
    )


def _settled_verdict(investigation: Pursuit) -> Verdict | None:
    """The investigator's conclusion, or None when it did not reach one.

    Read off the last surviving finding rather than off `Pursuit.cause`, because
    the reviewer needs the whole typed claim - the cited tool and value included -
    and a `Pursuit` that routed has already flattened those into strings.
    """
    if not investigation.routed:
        return None
    for attempt in reversed(investigation.attempts):
        if attempt.finding is not None:
            return attempt.finding
    return None


def _remediate(
    incident: Incident,
    verdict: Verdict,
    review: Review,
    completer: Completer,
    ledger: CostLedger,
    handoffs: list[Handoff],
) -> tuple[Remediation | None, tuple[ToolCall, ...]]:
    """Offer the write menu, execute nothing real, and record what was proposed.

    The toolbox is a `DryRunToolbox` unless a caller says otherwise, and it lies
    convincingly for the reason Chapter 9 gave: a tool result announcing DRY RUN
    would measure a different actor from the one anybody is worried about.
    """
    carried = remediate_user(verdict, review)
    handoffs.append(Handoff("reviewer", "remediator", carried))

    box: Toolbox = DryRunToolbox(incident=incident, allow_writes=True)
    metered = MeteredCompleter(inner=completer, ledger=ledger, stage="crew.remediator")

    def execute(call: ToolCall) -> ToolResult:
        billed = measured(f"crew.remediator.{call.call_id}", ledger)(box.execute)
        return billed(call)

    outcome = metered.invoke(
        system=REMEDIATE_SYSTEM,
        user=carried,
        tools=menu_for_role("remediator"),
        execute=execute,
        schema=Remediation,
    )
    return (outcome.parsed if outcome.ok else None), tuple(
        getattr(box, "proposed", ())
    )


def _page(
    incident: Incident,
    investigation: Pursuit,
    verdict: Verdict,
    review: Review | None,
    ledger: CostLedger,
    handoffs: tuple[Handoff, ...],
    carry: Carry,
    *,
    remediation: Remediation | None = None,
    proposed: tuple[ToolCall, ...] = (),
) -> Resolution:
    """Turn a reviewed conclusion into a page, through Chapter 3's rules.

    Routing goes through `owner_for` and `escalation_after` for the eighth
    chapter running. Seven rungs have now been built on top of a lookup table
    that has not changed since Chapter 3, and Chapter 16 can only walk any of
    this back down because none of them rewrote it.
    """
    service = owner_for(verdict.service)
    if service is None:
        return Resolution(
            decision=_unrouted(
                incident, f"no rota owns service {verdict.service!r}", "investigator"
            ),
            investigation=investigation,
            review=review,
            remediation=remediation,
            proposed=proposed,
            handoffs=handoffs,
            ledger=ledger,
            blocked_at="investigator",
            carry=carry,
        )

    severity = investigation.decision.severity or verdict.severity
    endorsement = (
        f"reviewer endorsed: {review.reason}" if review is not None else "no reviewer"
    )
    return Resolution(
        decision=RoutingDecision(
            incident_id=incident.incident_id,
            rota=service.rota,
            severity=severity,
            escalate_after_minutes=escalation_after(severity),
            already_overdue=False,
            recent_deploy=investigation.decision.recent_deploy,
            needs_human=False,
            decided_by=f"{investigation.decision.decided_by}; {endorsement}",
        ),
        investigation=investigation,
        review=review,
        remediation=remediation,
        proposed=proposed,
        handoffs=handoffs,
        ledger=ledger,
        carry=carry,
    )


def coordination_characters(resolution: Resolution) -> int:
    """Everything that crossed a role boundary, in characters.

    The free half of the Break-Even Test. This is the quantity a single agent
    does not pay at all: Chapter 9's loop carries its working set forward inside
    one conversation, and nothing is re-stated to a fresh reader. Every character
    counted here is a character bought twice.
    """
    return sum(hop.characters for hop in resolution.handoffs)


@register_rung("Level 7: Multi-Agent System")
def run(incident: Incident, completer: Completer | None = None) -> CostLedger:
    """Rung entry point. Three roles, and the top of the ladder.

    Note what the registry gets: one ledger, exactly as it got from Level 0's
    lookup table. That is the point of the cost table - eight architectures, one
    unit, one comparison - and it is the last row.

    DECLARED DEPARTURE, and it is a measurement decision rather than an
    architectural one. Every rung below returns a refusal shape when the seam
    fails, because those are production paths and Chapter 4 established that a
    vendor outage should degrade a system rather than break it. This function is
    not a production path - it is what `scripts/measure_costs.py` calls to build
    the published table - and a seam failure here means a role never ran. Handing
    back the surviving roles' ledger would publish a Level 7 row assembled from
    part of a Level 7 system.

    That is not hypothetical. It happened, it is the fourth outing for this
    defect in this repo, and the first version of this table printed a Level 7
    row byte-identical to Level 6's, because the crew replays its investigator
    from Chapter 9's recordings and has recordings for neither of the other two
    roles. So the rule the repo adopted at Chapter 8's gate applies here in its
    strongest form: report "not measured", never a number that is missing its
    expensive half.
    """
    active: Completer = completer if completer is not None else AnthropicCompleter()
    outcome = resolve(incident, active)
    if outcome.blocked_at is not None and outcome.review is None:
        # Distinguishes a seam failure from a judgment. A reviewer that REJECTED
        # a page produced a `Review` and is a real, cheap, measurable outcome; a
        # reviewer that never answered produced nothing, and averaging it in
        # would price a role that did not run.
        if outcome.blocked_at != "investigator" or not outcome.investigation.attempts:
            raise RuntimeError(
                f"{outcome.blocked_at} never answered for {incident.incident_id}: "
                f"{outcome.decision.decided_by}"
            )
    return outcome.ledger


__all__ = [
    "CARRY_DEFAULT",
    "MENUS",
    "REMEDIATE_SYSTEM",
    "REVIEW_SYSTEM",
    "ROLES",
    "Carry",
    "Handoff",
    "Remediation",
    "Resolution",
    "Review",
    "Role",
    "blast_radius",
    "coordination_characters",
    "menu_for_role",
    "quote_supported",
    "remediate_user",
    "resolve",
    "review_user",
    "run",
]
