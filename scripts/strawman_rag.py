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

Run:  python scripts/strawman_rag.py               (needs ANTHROPIC_API_KEY)
      python scripts/strawman_rag.py --show        (retrieval only, no key)
      python scripts/strawman_rag.py --samples 25  (a rate, not an anecdote)

`--samples` exists because a single response is something to print and not
something to trust - Chapter 4's rule, applied to Chapter 5's own opening scene.
It reports how often the strawman declines the irrelevant context versus how
often it answers from it anyway, and what lands in the two fields a closed enum
would have constrained.
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


VALID_SERVICES = {
    "checkout-api",
    "payment-gateway",
    "notification-worker",
    "search-api",
}
VALID_SEVERITIES = {"SEV1", "SEV2", "SEV3"}


def routable(answer: Answer) -> bool:
    """Could routing code actually act on this?

    The only question that matters about a strawman answer. Whether the prose is
    wise is a separate argument; whether `service` is a rota key and `severity`
    is a tier is decidable by a set membership test, and it is the test the
    schema should have been making.
    """
    return (
        answer.service in VALID_SERVICES and answer.severity in VALID_SEVERITIES
    )


def sample(incident: Incident, completer: Completer, runs: int) -> int:
    """Run the strawman `runs` times and report rates rather than an anecdote."""
    routable_count = 0
    declined = 0
    failed = 0
    services: dict[str, int] = {}

    for n in range(runs):
        answer = ask(incident, completer)
        if answer is None:
            failed += 1
            print(f"  {n:3d}  (no parsable answer)")
            continue
        ok = routable(answer)
        routable_count += ok
        # A free-text field carrying a sentence is the model declining with
        # nowhere to put the refusal. That is the failure the chapter is about.
        if not ok:
            declined += 1
        key = answer.service if ok else f"<prose: {answer.service[:40]}...>"
        services[key] = services.get(key, 0) + 1
        print(f"  {n:3d}  {'ROUTABLE' if ok else 'unroutable'}  {answer.service[:56]}")

    counted = runs - failed
    print()
    print("=" * 70)
    print(f"strawman on {incident.incident_id}, {runs} runs")
    print("=" * 70)
    print(f"  routable by rota lookup : {routable_count}/{counted}")
    print(f"  unroutable output       : {declined}/{counted}")
    print(f"  no parsable answer      : {failed}/{runs}")
    print("\n  what landed in `service`:")
    for value, count in sorted(services.items(), key=lambda kv: -kv[1]):
        print(f"    {count:3d}  {value}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="retrieval only")
    parser.add_argument("--incident", default="INC-1045")
    parser.add_argument(
        "--samples", type=int, default=0, help="run N times and report rates"
    )
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

    if args.samples:
        print(f"\nrunning {args.samples} samples ({args.samples} calls):\n")
        return sample(incident, AnthropicCompleter(), args.samples)

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
