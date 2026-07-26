"""Level 2 - retrieval-augmented generation.

One model call again, like Chapter 4, but the call now carries passages fetched
from a corpus at request time. That is the only capability this rung adds: facts
that are neither in the request nor reliably in the model.

Two decisions here are the chapter's argument rather than implementation detail.

The retriever has a relevance FLOOR as well as a limit. A `top_k` with no floor
always returns k passages, so it can never say "nothing here answers this" - it
is Chapter 3's `else: return DEFAULT` written as a ranking function, and it
produces the same failure one rung up: a fluent answer built on the least
irrelevant documents in the corpus. The floor is a measurement, not a taste
setting: it sits above the best score achieved by the queries in
`fixtures.retrieval_labels.UNANSWERABLE`.

Runbooks are indexed WHOLE rather than split by section. That is the opposite of
the reflex, and it is measured: a runbook's symptoms section is written in the
language of the report, and its checks section is written in the language of the
fix, so cutting between them puts the words that match the query and the words
that answer it in different passages. See `scripts/compare_retrievers.py`.

Standard library only, plus the pydantic Chapter 4 already introduced. On a
corpus this size a lexical retriever is the honest baseline, and the baseline is
the number any decision to buy an index has to beat.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from escalation_ladder.classify import normalized
from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.fixtures.retrieval_labels import LABELS
from escalation_ladder.fixtures.runbooks_index import runbook_paths
from escalation_ladder.instrument import CostLedger, measured
from escalation_ladder.llm import AnthropicCompleter, Completer, Completion
from escalation_ladder.rules import (
    SEVERITIES,
    RoutingDecision,
    escalation_after,
    owner_for,
)
from escalation_ladder.rungs import register_rung

# BM25 parameters, at the published defaults. They are defaults rather than
# tuning: a corpus this size has nowhere near the data needed to justify moving
# them, and a team tuning k1 before measuring recall is optimizing the half of
# the system that was not broken.
K1 = 1.2
B = 0.75

# Below this score, the system refuses instead of answering from whatever ranked
# highest. Calibrated in `scripts/compare_retrievers.py` against queries this
# corpus cannot answer - see `UNANSWERABLE` - and it must be re-derived whenever
# the corpus changes, because BM25 scores are relative to a corpus.
FLOOR = 6.0

# How many passages go into the prompt when several clear the floor. Small on
# purpose: input tokens are roughly linear in this number and the third passage
# is rarely the one that helps.
LIMIT = 3

_WORD = re.compile(r"[a-z0-9_]+")

# Stripped in this order, longest first. Deliberately not a real stemmer: a
# Porter implementation is a hundred lines that nobody on the team will read,
# and this handles the case that actually costs recall here - a report saying
# "duplicates" against a runbook saying "duplicate". Nothing below closes a
# semantic gap, and the chapter is explicit that it does not try to.
_SUFFIXES = ("ies", "ing", "ed", "s")
_MIN_STEM = 4


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            if suffix == "ies":
                return f"{token[:-3]}y"
            return token[: -len(suffix)]
    return token


def terms(text: str) -> list[str]:
    """Lowercase word tokens, lightly stemmed.

    No stopword list. Inverse document frequency already drives common words to
    near-zero weight, and a stopword list is a second corpus to maintain that
    silently deletes `for` out of an alert grammar.

    Stemming is not optional decoration. Before it, `emails`, `duplicates`, and
    `times` all had a document frequency of zero against a corpus that says
    `email`, `duplicate`, and `time` - three of the query's most diagnostic
    terms scoring nothing because of one character each.
    """
    return [_stem(token) for token in _WORD.findall(text.casefold())]


@dataclass(frozen=True)
class Passage:
    """One retrievable unit of the corpus.

    `passage_id` is stable and human-readable because it travels all the way to
    the engineer who gets paged. A citation nobody can resolve by hand is not a
    citation.
    """

    passage_id: str
    source: str
    text: str


def chunk_runbook(text: str, stem: str) -> Passage:
    """One passage per runbook.

    Not a typo, and not laziness. These documents run to a couple of hundred
    words with a symptoms section written in the language of a report and a
    checks section written in the language of a fix. Splitting them separates
    the half a query can find from the half the engineer needs, which is
    measurable and is measured.
    """
    body = text.strip()
    lines = body.splitlines()
    title = lines[0].lstrip("# ").strip() if lines else stem
    return Passage(passage_id=stem, source=title, text=body)


def chunk_incident(incident: Incident) -> Passage:
    """One passage per resolved past incident, symptom and resolution together.

    Splitting these would produce a corpus whose findable half never contains
    its useful half - the same failure as splitting a runbook, arrived at from
    the other direction.
    """
    return Passage(
        passage_id=f"incident:{incident.incident_id}",
        source=f"past incident {incident.incident_id}",
        text=(
            f"{incident.title}\n\n"
            f"Reported: {incident.report}\n"
            f"Root cause: {incident.root_cause}\n"
            f"Resolution: {incident.resolution}"
        ),
    )


def build_corpus(exclude_incident: str | None = None) -> tuple[Passage, ...]:
    """Every runbook, plus every past incident except the one being triaged.

    `exclude_incident` is not a nicety. The past-incident corpus contains the
    incident under triage, complete with its root cause and its resolution, and
    an index that can return it answers every question perfectly while measuring
    nothing. Forgetting this argument is how a retrieval eval comes back at 100
    percent the first time anyone runs it.
    """
    passages: list[Passage] = [
        chunk_runbook(path.read_text(encoding="utf-8"), path.stem)
        for path in runbook_paths()
    ]
    passages.extend(
        chunk_incident(incident)
        for incident in load_incidents()
        if incident.incident_id != exclude_incident
    )
    return tuple(passages)


@dataclass(frozen=True)
class Index:
    """A corpus with its term statistics precomputed."""

    passages: tuple[Passage, ...]
    frequencies: tuple[dict[str, int], ...]
    lengths: tuple[int, ...]
    document_frequency: dict[str, int]
    average_length: float

    def inverse_document_frequency(self, term: str) -> float:
        """Rarity weight. Zero for a term that appears in no passage at all."""
        seen = self.document_frequency.get(term, 0)
        if seen == 0:
            return 0.0
        total = len(self.passages)
        return math.log(1.0 + (total - seen + 0.5) / (seen + 0.5))


def build_index(passages: tuple[Passage, ...]) -> Index:
    """Precompute term statistics once per corpus."""
    frequencies: list[dict[str, int]] = []
    lengths: list[int] = []
    document_frequency: dict[str, int] = {}
    for passage in passages:
        counts: dict[str, int] = {}
        tokens = terms(passage.text)
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        frequencies.append(counts)
        lengths.append(len(tokens))
        for token in counts:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    average = sum(lengths) / len(lengths) if lengths else 0.0
    return Index(
        passages=passages,
        frequencies=tuple(frequencies),
        lengths=tuple(lengths),
        document_frequency=document_frequency,
        average_length=average,
    )


@dataclass(frozen=True)
class Hit:
    """One retrieved passage, its score, and the words that earned it.

    `matched` is this module's `decided_by`. A retrieval that cannot say which
    terms produced the match is as hard to argue with as a route that cannot say
    which rule fired, and reading those terms is usually enough to diagnose a
    bad retrieval without any other tooling.
    """

    passage: Passage
    score: float
    matched: tuple[str, ...]


def _bm25(index: Index, position: int, query_terms: list[str]) -> float:
    counts = index.frequencies[position]
    length = index.lengths[position]
    total = 0.0
    for term in query_terms:
        frequency = counts.get(term, 0)
        if frequency == 0:
            continue
        norm = 1.0 - B + B * (length / index.average_length)
        total += index.inverse_document_frequency(term) * (
            frequency * (K1 + 1.0) / (frequency + K1 * norm)
        )
    return total


def search(
    index: Index,
    query: str,
    *,
    limit: int = LIMIT,
    floor: float = FLOOR,
) -> tuple[Hit, ...]:
    """Rank by BM25, admit by the floor, and return nothing when nothing clears.

    The empty tuple is the load-bearing return value, exactly as `None` was in
    Chapter 3's `parse_alert`. It is how "this corpus does not answer that" gets
    out of this function, and it is what makes the refusal in `advise` possible.
    """
    query_terms = terms(query)
    unique = set(query_terms)
    hits: list[Hit] = []
    for position, passage in enumerate(index.passages):
        score = _bm25(index, position, query_terms)
        if score < floor:
            continue
        matched = tuple(
            sorted(t for t in unique if t in index.frequencies[position])
        )
        hits.append(Hit(passage, score, matched))
    # Ties break on passage_id, so one corpus always produces one ordering. A
    # retrieval eval that moves between runs is measuring its own sort.
    hits.sort(key=lambda hit: (-hit.score, hit.passage.passage_id))
    return tuple(hits[:limit])


def default_index(incident: Incident) -> Index:
    """The corpus this incident is allowed to see: everything but itself."""
    return build_index(build_corpus(exclude_incident=incident.incident_id))


class Grounded(BaseModel):
    """What one grounded call is allowed to return.

    Chapter 4's schema plus a citation and a next step. `passage_id` cannot be a
    `Literal` the way `service` can, because the retrieved set is different on
    every request - so it is the first field in this book whose validity is
    checked entirely by code rather than constrained by an enum.
    """

    service: Literal[
        "checkout-api",
        "payment-gateway",
        "notification-worker",
        "search-api",
        "unknown",
    ]
    severity: Literal["SEV1", "SEV2", "SEV3"]
    passage_id: str
    quote: str
    next_step: str
    needs_human: bool


SYSTEM = """You triage production incident reports for an on-call rota, using
runbook and past-incident passages retrieved for the report.

Return the owning service, the severity, the id of the ONE passage that decides
the next step, a verbatim quote from that passage, and the single next action the
on-call engineer should take.

The services, and what each one owns:
- checkout-api: the checkout flow and order placement
- payment-gateway: card authorization and capture, and the acquirer connection
- notification-worker: transactional email and push delivery
- search-api: search queries and the search index

Rules:
- Identify the service from the symptom the report describes, not from whether
  the report names a service. Reports are written by people, and people describe
  what broke.
- The passages were selected by a keyword search that knows nothing about this
  report. They may be about a different service entirely. Judge whether a
  passage is about THIS incident before using it.
- Set `needs_human` to true when no retrieved passage addresses this incident,
  including when a passage is about the right service but the wrong failure. A
  passage about the right service and the wrong failure is not an answer.
- `passage_id` must be copied from the passage markers. `quote` must be copied
  character for character from that same passage.
- `next_step` must be an action named in the cited passage. Do not supply a step
  from your own knowledge, however reasonable that step is.
- Severity is about customer impact, not tone."""


def context_block(hits: tuple[Hit, ...]) -> str:
    """The retrieved passages, fenced and labeled with the ids to cite."""
    return "\n\n".join(
        f'<passage id="{hit.passage.passage_id}">\n{hit.passage.text}\n</passage>'
        for hit in hits
    )


def build_user(incident: Incident, hits: tuple[Hit, ...]) -> str:
    """The user turn: retrieved passages first, then the report.

    Passages lead because a passage is byte-identical across far more requests
    than a report is, and a cache prefix only pays when it is a prefix.
    """
    return (
        "Retrieved passages:\n\n"
        f"{context_block(hits)}\n\n"
        "Classify the incident report between the markers and give the next "
        "step, citing one passage above.\n\n"
        f"<report>\n{incident.report}\n</report>"
    )


def citation_supported(claim: Grounded, hits: tuple[Hit, ...]) -> bool:
    """Was the cited passage retrieved, and is the quote really inside it?

    Two failures, one check, and they are not the same failure. An id that was
    never retrieved means the model invented a source. A real id whose text does
    not contain the quote means the model attributed something to a document
    that does not say it - the citation failure that survives review, because
    the reference resolves and nobody opens it.
    """
    by_id = {hit.passage.passage_id: hit.passage for hit in hits}
    passage = by_id.get(claim.passage_id)
    if passage is None:
        return False
    quote = normalized(claim.quote)
    return bool(quote) and quote in normalized(passage.text)


@dataclass(frozen=True)
class GroundedAdvice:
    """A page, plus the document that says what to do next.

    Composes Chapter 3's `RoutingDecision` rather than replacing it. The routing
    half is unchanged by this rung, and a rung that rewrote it would break the
    fallback Chapter 4 depends on.
    """

    decision: RoutingDecision
    next_step: str | None
    citation: str | None
    quote: str | None

    @property
    def grounded(self) -> bool:
        return self.decision.routed and self.citation is not None

    def summary(self) -> str:
        if not self.decision.routed:
            return self.decision.summary()
        return (
            f"{self.decision.summary()}\n"
            f"  next step: {self.next_step}\n"
            f"  source: {self.citation}\n"
            f"  quote: {self.quote!r}"
        )


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


def _at_least(claimed: str, tier: str) -> str:
    """Chapter 3's one-directional severity floor, unchanged again."""
    return SEVERITIES[min(SEVERITIES.index(claimed), SEVERITIES.index(tier))]


def advise(
    incident: Incident,
    completer: Completer,
    index: Index | None = None,
) -> GroundedAdvice:
    """Retrieve, then answer from what was retrieved, or refuse.

    Every exit names its reason, so the refusals stay countable and separable.
    An empty retrieval, a model declining the retrieved set, and an unsupported
    citation are three different problems with three different fixes, and a
    single `needs_human` counter hides which one you have.
    """
    active = index if index is not None else default_index(incident)
    hits = search(active, incident.report)
    # No passages means no call. A question the corpus cannot answer costs
    # nothing rather than costing a full-context request that produces a guess.
    if not hits:
        return _unrouted(incident, "no passage cleared the relevance floor")

    completion = completer.parse(
        system=SYSTEM, user=build_user(incident, hits), schema=Grounded
    )
    if not completion.ok or completion.parsed is None:
        return _unrouted(incident, completion.failed or "no result")

    claim = completion.parsed
    if claim.needs_human or claim.service == "unknown":
        return _unrouted(
            incident, "the retrieved passages do not address this incident"
        )
    if not citation_supported(claim, hits):
        return _unrouted(incident, f"citation not supported: {claim.passage_id!r}")

    service = owner_for(claim.service)
    if service is None:
        return _unrouted(incident, f"no rota owns service {claim.service!r}")

    severity = _at_least(claim.severity, service.tier)
    return GroundedAdvice(
        decision=RoutingDecision(
            incident_id=incident.incident_id,
            rota=service.rota,
            severity=severity,
            escalate_after_minutes=escalation_after(severity),
            already_overdue=False,
            recent_deploy=None,
            needs_human=False,
            decided_by=f"grounded in {claim.passage_id}",
        ),
        next_step=claim.next_step,
        citation=claim.passage_id,
        quote=claim.quote,
    )


def context_recall(
    *,
    limit: int = LIMIT,
    floor: float = FLOOR,
) -> dict[str, float]:
    """Retrieval quality alone, with no model anywhere in the measurement.

    The share of each incident's labeled answer spans that actually reached the
    prompt. Labels are spans rather than passage ids on purpose, so this number
    is comparable across chunking strategies instead of being defined by one.

    This is the number that has to move before any generation change can be
    credited with anything, and it costs nothing to compute.
    """
    scores: dict[str, float] = {}
    for incident in load_incidents():
        label = LABELS.get(incident.incident_id)
        if label is None:
            continue
        hits = search(
            default_index(incident), incident.report, limit=limit, floor=floor
        )
        context = normalized(" ".join(hit.passage.text for hit in hits))
        found = sum(1 for span in label.answers if normalized(span) in context)
        scores[incident.incident_id] = found / len(label.answers)
    return scores


@register_rung("Level 2: Retrieval-Augmented Generation")
def run(incident: Incident, completer: Completer | None = None) -> CostLedger:
    """Rung entry point. One retrieval and at most one measured model call."""
    ledger = CostLedger()
    active: Completer = completer if completer is not None else AnthropicCompleter()
    index = default_index(incident)

    @measured("retrieve.search", ledger)
    def _search() -> tuple[Hit, ...]:
        return search(index, incident.report)

    hits = _search()
    if not hits:
        return ledger

    @measured("retrieve.ask", ledger)
    def _ask() -> Completion[Grounded]:
        return active.parse(
            system=SYSTEM, user=build_user(incident, hits), schema=Grounded
        )

    _ask()
    return ledger
