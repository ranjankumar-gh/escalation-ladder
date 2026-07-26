"""Chapter 6's retrieval experiment. No API key, no network, no spend.

Level 2 built its query from the raw report, because at that rung nothing had
read the report yet. Level 3 builds it from stage 1's classification. That is the
entire capability the rung adds, so it is worth knowing what it buys - and what
shape of it buys anything - before paying for a second model call anywhere.

Four queries, one retriever, one golden set:

  1. `report`    - Chapter 5's query. The raw incident report.
  2. `replaced`  - the reflex, and what Chapter 5's Failure Receipt demonstrated:
                   `service + summary`, the classification INSTEAD of the report.
  3. `augmented` - what this chapter ships: `service + report`. The classification
                   ADDED to the report rather than replacing it.
  4. `rewritten` - a model asked to write a search query, replayed from a
                   recording so this script still costs nothing. The stage that
                   the Floor Test says should not be a model call.

Classifications come from Chapter 4's real recorded output, so 2 and 3 are built
from an answer a model actually gave rather than from the label.

Scored on context recall against `fixtures.retrieval_labels` - the same labeled
answer spans Chapter 5 used, so the numbers are comparable across both chapters.

Run:  python scripts/compare_queries.py
"""
from __future__ import annotations

import json
from pathlib import Path

from escalation_ladder.classify import Classification, classify, normalized
from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.fixtures.retrieval_labels import LABELS
from escalation_ladder.llm import RecordedCompleter
from escalation_ladder.retrieve import FLOOR, LIMIT, default_index, search
from escalation_ladder.workflow import extract

RECORDINGS = Path(__file__).resolve().parent.parent / "tests" / "recordings"
CH04 = RECORDINGS / "ch04_classify.json"
CH06 = RECORDINGS / "ch06_workflow.json"

COLUMNS = ("report", "replaced", "augmented", "rewritten")


def _load(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def recorded_classification(incident: Incident) -> Classification | None:
    """Chapter 4's real answer for this report, replayed for free."""
    completer = RecordedCompleter(recordings=_load(CH04))  # type: ignore[arg-type]
    claim = classify(incident, completer)
    if claim.needs_human or claim.service == "unknown":
        return None
    return claim


def rewritten_query(incident: Incident) -> str | None:
    """The model-written search query, replayed from Chapter 6's recording."""
    queries = _load(CH06).get("_rewritten_queries")
    if not isinstance(queries, dict):
        return None
    value = queries.get(incident.incident_id)
    return value if isinstance(value, str) else None


def queries_for(incident: Incident) -> dict[str, str | None]:
    """The four query shapes for one incident, or None where one is unavailable."""
    claim = recorded_classification(incident)
    return {
        "report": incident.report,
        "replaced": f"{claim.service} {claim.summary}" if claim else None,
        "augmented": extract(claim, incident) if claim else None,
        "rewritten": rewritten_query(incident),
    }


def recall(incident: Incident, query: str) -> float:
    """Share of this incident's labeled answer spans that reached the prompt."""
    label = LABELS[incident.incident_id]
    hits = search(default_index(incident), query, limit=LIMIT, floor=FLOOR)
    context = normalized(" ".join(hit.passage.text for hit in hits))
    found = sum(1 for span in label.answers if normalized(span) in context)
    return found / len(label.answers)


def main() -> None:
    incidents = [i for i in load_incidents() if i.incident_id in LABELS]
    scored: dict[str, list[float]] = {name: [] for name in COLUMNS}

    print(f"Context recall @{LIMIT}, floor {FLOOR}\n")
    header = f"{'incident':<10}" + "".join(f"{name:>11}" for name in COLUMNS)
    print(header)
    for incident in incidents:
        queries = queries_for(incident)
        row = f"{incident.incident_id:<10}"
        for name in COLUMNS:
            query = queries[name]
            if query is None:
                row += f"{'--':>11}"
                continue
            score = recall(incident, query)
            scored[name].append(score)
            row += f"{score:>11.2f}"
        print(row)

    print()
    for name in COLUMNS:
        values = scored[name]
        if not values:
            print(f"{name:<10} mean    --   (no recording)")
            continue
        mean = sum(values) / len(values)
        print(f"{name:<10} mean {mean:>5.2f}   over {len(values)} incidents")

    comparable = [i for i in incidents if queries_for(i)["augmented"] is not None]
    if comparable:
        print(f"\nOn the {len(comparable)} incidents Chapter 4 classifies:")
        for name in COLUMNS:
            values = [
                recall(i, q)
                for i in comparable
                if (q := queries_for(i)[name]) is not None
            ]
            if len(values) == len(comparable):
                print(f"  {name:<10} {sum(values) / len(values):.2f}")

    print("\nThe queries themselves:\n")
    for incident in comparable:
        queries = queries_for(incident)
        print(f"{incident.incident_id}")
        for name in COLUMNS:
            query = queries[name]
            print(f"  {name:<10}: {query if query else '-- no recording'}")


if __name__ == "__main__":
    main()
