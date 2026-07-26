"""The retrieval system Chapter 5 argues against, built so it can be measured.

Three defects, and every one of them is what a framework quickstart hands you:

  1. fixed-size chunks with no overlap, cutting through sentences and dropping
     the document title, so a chunk cannot say what it is about
  2. unconditional top-k, with no relevance floor, so the retriever always
     returns three passages and can never report that it found nothing
  3. a prompt that says "answer using the context below" and never gives the
     model permission to say the context does not apply

None of the three is stupid in isolation, which is why this configuration is
everywhere. Together they guarantee that a question the corpus cannot answer
still produces a fluent, cited answer.

Run:  python scripts/strawman_rag.py            (needs ANTHROPIC_API_KEY)
      python scripts/strawman_rag.py --show     (retrieval only, no key needed)
"""
from __future__ import annotations

import argparse

from pydantic import BaseModel

from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.fixtures.runbooks_index import runbook_paths
from escalation_ladder.llm import AnthropicCompleter, Completer
from escalation_ladder.retrieve import (
    Hit,
    Index,
    Passage,
    build_index,
    chunk_incident,
    search,
)

CHUNK_CHARS = 400
TOP_K = 3


class Answer(BaseModel):
    """The strawman schema. Note what is missing: any way to decline."""

    service: str
    severity: str
    next_step: str
    source: str


STRAWMAN_SYSTEM = """You are an incident triage assistant.

Use the context below to answer. Give the owning service, the severity, the next
step the on-call engineer should take, and the source you used."""


def strawman_index(exclude: str | None) -> Index:
    """Fixed windows over the raw file bytes. No titles, no boundaries."""
    passages: list[Passage] = []
    for path in runbook_paths():
        text = path.read_text(encoding="utf-8").strip()
        for n, start in enumerate(range(0, len(text), CHUNK_CHARS)):
            passages.append(
                Passage(
                    passage_id=f"{path.stem}~{n}",
                    source=path.stem,
                    text=text[start : start + CHUNK_CHARS],
                )
            )
    passages.extend(
        chunk_incident(i) for i in load_incidents() if i.incident_id != exclude
    )
    return build_index(tuple(passages))


def strawman_user(incident: Incident, hits: tuple[Hit, ...]) -> str:
    context = "\n\n---\n\n".join(hit.passage.text for hit in hits)
    return f"Context:\n\n{context}\n\nIncident report:\n\n{incident.report}"


def ask(incident: Incident, completer: Completer) -> Answer | None:
    # floor=0.0 is the whole point: something always comes back.
    hits = search(
        strawman_index(incident.incident_id),
        incident.report,
        limit=TOP_K,
        floor=0.0,
    )
    completion = completer.parse(
        system=STRAWMAN_SYSTEM,
        user=strawman_user(incident, hits),
        schema=Answer,
    )
    return completion.parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="retrieval only")
    parser.add_argument("--incident", default="INC-1045")
    args = parser.parse_args()

    incident = next(
        i for i in load_incidents() if i.incident_id == args.incident
    )
    hits = search(
        strawman_index(incident.incident_id),
        incident.report,
        limit=TOP_K,
        floor=0.0,
    )
    print(f"{incident.incident_id}: {incident.report}\n")
    print(f"retrieved {len(hits)} passages, floor=0.0:")
    for hit in hits:
        head = " ".join(hit.passage.text.split())[:78]
        print(f"  {hit.score:6.2f}  {hit.passage.passage_id:26s} {head}")
    if args.show:
        return 0

    answer = ask(incident, AnthropicCompleter())
    print("\nthe strawman's answer:")
    if answer is None:
        print("  (no parsable answer)")
        return 1
    print(f"  service   : {answer.service}")
    print(f"  severity  : {answer.severity}")
    print(f"  next step : {answer.next_step}")
    print(f"  source    : {answer.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
