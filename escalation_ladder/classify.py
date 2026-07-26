"""Level 1 - the prompt application.

One model call turns a free-text incident report into a typed `Classification`.
This rung adds exactly one capability over Chapter 3: language understanding
over input the system already has. It reads no documents, queries nothing, and
makes no second call, so the schema below is the whole interface.

Every failure path returns Chapter 3's refusal shape, which is why the cheap
path had to survive. When the vendor is unreachable this module degrades to the
behavior the system had before it existed.

Measured 2026-07-26 over 100 calls to claude-opus-5 at effort low, against the
four reports Level 0 refuses: 3.8s p50 / 7.6s p99, 1098 input and 128 output
tokens per request, no failures. Routes 4/4 with 4/4 services correct and 2/4
severities correct from the model - 3/4 once the Chapter 3 severity floor
applies. Regenerate with tools/measure_ch04.py in the book repo. If SYSTEM
changes, the recordings in tests/ are invalidated by design: prompt_key hashes
the prompt.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.instrument import CostLedger, measured
from escalation_ladder.llm import AnthropicCompleter, Completer, Completion
from escalation_ladder.rules import (
    SEVERITIES,
    RoutingDecision,
    escalation_after,
    owner_for,
)
from escalation_ladder.rungs import register_rung


class Classification(BaseModel):
    """What one model call is allowed to return.

    Note what is not here: no root cause, no fix, no runbook reference, no
    confidence score. Level 1 has read nothing beyond the report, so a field it
    could only fill by guessing has no business being in the schema.

    `service` carries an "unknown" sentinel because an enum with no way to say
    "not one of these" is a default branch written as a type - it forces a
    confident answer out of an input that does not contain one.
    """

    service: Literal[
        "checkout-api",
        "payment-gateway",
        "notification-worker",
        "search-api",
        "unknown",
    ]
    severity: Literal["SEV1", "SEV2", "SEV3"]
    summary: str
    evidence: str
    needs_human: bool


SYSTEM = """You triage production incident reports for an on-call rota.

Return the owning service, the severity, a one-line summary, and a verbatim
quote from the report that decides the service.

The services, and what each one owns:
- checkout-api: the checkout flow and order placement
- payment-gateway: card authorization and capture, and the acquirer connection
- notification-worker: transactional email and push delivery
- search-api: search queries and the search index

Rules:
- Identify the service from the symptom the report describes, not from whether
  the report names a service. Reports are written by people, and people
  describe what broke. "Their card was charged" identifies payment-gateway.
- Use "unknown" only when the described symptom belongs to none of those
  services, or belongs to several of them equally.
- `evidence` must be copied character for character from the report. Do not
  paraphrase, and do not quote text that is not there.
- Severity is about customer impact, not tone. A calm report of a payment
  failure outranks an alarmed report of a cosmetic bug.
- Set `needs_human` to true when you cannot identify the service, and not
  merely because the report is vague about scope. A report that is imprecise
  about how many customers are affected but clear about which service is
  broken is routable; the engineer who gets paged will narrow the scope."""


def build_user(incident: Incident) -> str:
    """The user turn: the report, fenced and labeled as data.

    Fencing untrusted text helps and is not a security boundary. The closed
    enum in `Classification` is the boundary, because it constrains the tokens
    rather than the model's intentions.
    """
    return (
        "Classify the incident report between the markers.\n\n"
        f"<report>\n{incident.report}\n</report>"
    )


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def evidence_supported(claim: Classification, incident: Incident) -> bool:
    """Is the model's quote actually in the report?

    Whitespace and case are normalized, because a model that re-wrapped a line
    has not invented anything. Nothing else is forgiven: a paraphrase fails
    this check, and so does a quote from a report nobody was shown.
    """
    quote = _normalized(claim.evidence)
    return bool(quote) and quote in _normalized(incident.report)


def _unknown(reason: str) -> Classification:
    """A classification that refuses, and says why in `summary`.

    `severity` is required by the schema and meaningless here; it is never read
    because `needs_human` short-circuits `decide` before severity is used.
    """
    return Classification(
        service="unknown",
        severity="SEV3",
        summary=reason,
        evidence="",
        needs_human=True,
    )


def _needs_human(incident: Incident, reason: str) -> RoutingDecision:
    """Chapter 3's refusal shape, reused verbatim."""
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
    """The more severe of the model's answer and the service tier.

    Chapter 3's one-directional severity rule, surviving contact with a model:
    a model may raise severity above the service tier and may never lower it.
    """
    return SEVERITIES[min(SEVERITIES.index(claimed), SEVERITIES.index(tier))]


def classify(incident: Incident, completer: Completer) -> Classification:
    """Classify one report with a single model call, or refuse.

    Every exit is either a classification whose quote checks out, or an
    `unknown` that names why - which keeps Chapter 3's property intact: the
    refusal rate is countable, and its reasons are countable separately.
    """
    completion = completer.parse(
        system=SYSTEM, user=build_user(incident), schema=Classification
    )
    if not completion.ok or completion.parsed is None:
        return _unknown(completion.failed or "no result")
    claim = completion.parsed
    # A declared refusal is honored before the quote is checked. Checking
    # evidence first would file every honest "I cannot tell" under "invented
    # evidence" and merge two counters that have to stay apart.
    if claim.needs_human or claim.service == "unknown":
        return _unknown(claim.summary or "the model declined to classify")
    if not evidence_supported(claim, incident):
        return _unknown(f"quoted evidence not in the report: {claim.evidence!r}")
    return claim


def decide(incident: Incident, claim: Classification) -> RoutingDecision:
    """Turn a checked claim into a page, using Chapter 3's tables.

    Level 1 supplies the service and the severity. Which rota owns that service
    and how long before the page escalates are still lookups, because they never
    needed a model and they do not need one now.
    """
    if claim.needs_human or claim.service == "unknown":
        return _needs_human(incident, claim.summary or "the model declined to route")

    service = owner_for(claim.service)
    if service is None:
        # Unreachable while the schema and the table agree, which is what
        # tests/test_classify.py asserts. Left in because the alternative is a
        # KeyError on a paging path the first time someone edits one of them.
        return _needs_human(incident, f"no rota owns service {claim.service!r}")

    severity = _at_least(claim.severity, service.tier)
    return RoutingDecision(
        incident_id=incident.incident_id,
        rota=service.rota,
        severity=severity,
        escalate_after_minutes=escalation_after(severity),
        # Level 1 has no alert line, so there is no firing duration to compare
        # against the SLA. An honest False beats a plausible guess.
        already_overdue=False,
        recent_deploy=None,
        needs_human=False,
        decided_by=f"model classification, evidence {claim.evidence!r}",
    )


def triage(incident: Incident, completer: Completer) -> RoutingDecision:
    """Classify a report and turn the result into a page, or refuse."""
    return decide(incident, classify(incident, completer))


@register_rung("Level 1: Prompt Application")
def run(incident: Incident, completer: Completer | None = None) -> CostLedger:
    """Rung entry point. Exactly one measured model call per incident."""
    ledger = CostLedger()
    active: Completer = completer if completer is not None else AnthropicCompleter()

    @measured("classify.ask", ledger)
    def _ask() -> Completion[Classification]:
        return active.parse(
            system=SYSTEM, user=build_user(incident), schema=Classification
        )

    _ask()
    return ledger
