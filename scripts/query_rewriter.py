"""The stage Chapter 6 decided NOT to build, kept so the decision stays checkable.

Stage 2 of the workflow turns a classification into a retrieval query. The reflex
is to make that a model call - it sits between two model calls, and "query
rewriting" is a named technique with papers behind it. Chapter 6 runs the Floor
Test on the stage instead and ships an f-string.

This module is the alternative that claim is measured against. It is a real
implementation, not a strawman: a focused prompt, a typed output, and the same
retriever underneath. `scripts/compare_queries.py` replays its recorded output
alongside the three code-built queries, so the comparison costs nothing to
re-read and one small call per incident to regenerate.

Regenerating needs a key:  python tools/measure_ch06.py  (in the book repo)
"""
from __future__ import annotations

from pydantic import BaseModel

from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.llm import Completer

SYSTEM = """You write search queries for a keyword search over a corpus of
on-call runbooks and resolved past incidents.

Given an incident report, write the query most likely to retrieve the runbook
that says what to do about it.

The corpus is indexed by BM25 over plain words. There is no semantic matching,
so a term that does not appear in a document contributes nothing. Runbooks are
written in operational vocabulary: service names, metric names, error classes,
and the names of the components involved.

Return only the query text. Do not explain it."""


class RewrittenQuery(BaseModel):
    """One search query, and the model's reason for it.

    `reason` is here so a bad query can be read rather than guessed at. It is
    never used to build the query, and nothing downstream reads it.
    """

    query: str
    reason: str


def build_user(incident: Incident) -> str:
    return (
        "Write a search query for the incident report between the markers.\n\n"
        f"<report>\n{incident.report}\n</report>"
    )


def rewrite(incident: Incident, completer: Completer) -> RewrittenQuery | None:
    """Ask a model for a search query, or return None if the call failed."""
    completion = completer.parse(
        system=SYSTEM, user=build_user(incident), schema=RewrittenQuery
    )
    return completion.parsed if completion.ok else None
