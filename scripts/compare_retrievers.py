"""Chapter 5's retrieval experiments. No API key, no network, no spend.

Three questions, all answered by measurement rather than by preference:

  1. Where does the relevance floor go? Calibrated against queries the corpus
     cannot answer, because a floor set by taste is a number nobody can defend.
  2. Which chunking wins? Whole runbooks, split by section, or fixed-size
     windows - scored on labeled answer spans, which is the only ground truth
     that does not favor one of the three by construction.
  3. What does dropping the floor cost? This is the strawman: unconditional
     top-k, which is Chapter 3's default branch wearing a ranking function.

Run:  python scripts/compare_retrievers.py
"""
from __future__ import annotations

from typing import Callable

from escalation_ladder.classify import normalized
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.fixtures.retrieval_labels import LABELS, UNANSWERABLE
from escalation_ladder.fixtures.runbooks_index import runbook_paths
from escalation_ladder.retrieve import (
    FLOOR,
    LIMIT,
    Index,
    Passage,
    build_index,
    chunk_incident,
    chunk_runbook,
    search,
)

STRAWMAN_CHUNK = 400


def _past_incidents(exclude: str | None) -> list[Passage]:
    return [
        chunk_incident(incident)
        for incident in load_incidents()
        if incident.incident_id != exclude
    ]


def whole_index(exclude: str | None) -> Index:
    """What the module ships: one passage per runbook."""
    passages = [
        chunk_runbook(path.read_text(encoding="utf-8"), path.stem)
        for path in runbook_paths()
    ]
    return build_index(tuple(passages + _past_incidents(exclude)))


def section_index(exclude: str | None) -> Index:
    """One passage per `##` section. The reflex, and it is measured below."""
    passages: list[Passage] = []
    for path in runbook_paths():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip()
        heading, body = None, []

        def flush() -> None:
            if heading is None:
                return
            joined = "\n".join(body).strip()
            passages.append(
                Passage(
                    passage_id=f"{path.stem}#{'-'.join(heading.lower().split())}",
                    source=title,
                    text=f"{title}\n\n## {heading}\n{joined}",
                )
            )

        for line in lines[1:]:
            if line.startswith("## "):
                flush()
                heading, body = line[3:].strip(), []
            elif heading is not None:
                body.append(line)
        flush()
    return build_index(tuple(passages + _past_incidents(exclude)))


def fixed_index(exclude: str | None, size: int = STRAWMAN_CHUNK) -> Index:
    """Fixed character windows, no overlap, no title carried into the chunk."""
    passages: list[Passage] = []
    for path in runbook_paths():
        text = path.read_text(encoding="utf-8").strip()
        for n, start in enumerate(range(0, len(text), size)):
            passages.append(
                Passage(
                    passage_id=f"{path.stem}~{n}",
                    source=path.stem,
                    text=text[start : start + size],
                )
            )
    return build_index(tuple(passages + _past_incidents(exclude)))


def context_recall(
    builder: Callable[[str | None], Index],
    *,
    limit: int,
    floor: float,
) -> tuple[float, dict[str, float]]:
    """Share of labeled answer spans that reached the prompt."""
    per: dict[str, float] = {}
    for incident in load_incidents():
        label = LABELS[incident.incident_id]
        hits = search(
            builder(incident.incident_id),
            incident.report,
            limit=limit,
            floor=floor,
        )
        context = normalized(" ".join(hit.passage.text for hit in hits))
        found = sum(1 for span in label.answers if normalized(span) in context)
        per[incident.incident_id] = found / len(label.answers)
    return sum(per.values()) / len(per), per


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    rule("1. WHERE THE FLOOR GOES")
    print("Top BM25 score for a real incident report:")
    real = []
    for incident in load_incidents():
        hits = search(
            whole_index(incident.incident_id), incident.report, limit=1, floor=0.0
        )
        top = hits[0].score if hits else 0.0
        real.append(top)
        best = hits[0].passage.passage_id if hits else "-"
        print(f"   {incident.incident_id}   {top:6.2f}   {best}")

    print("\nTop BM25 score for a query this corpus cannot answer:")
    junk = []
    index = whole_index(None)
    for query in UNANSWERABLE:
        hits = search(index, query, limit=1, floor=0.0)
        top = hits[0].score if hits else 0.0
        junk.append(top)
        best = hits[0].passage.passage_id if hits else "(nothing matched)"
        print(f"              {top:6.2f}   {best:22s} {query[:34]}")

    print(f"\n   lowest real report       : {min(real):.2f}")
    print(f"   highest unanswerable     : {max(junk):.2f}")
    print(f"   separable                : {min(real) > max(junk)}")
    print(f"   FLOOR in retrieve.py     : {FLOOR}")

    rule("2. WHICH CHUNKING WINS  (context recall on labeled answer spans)")
    header = "  ".join(f"k={k}" for k in (1, 3, 5))
    print(f"{'strategy':14s} {'passages':>9s}  " + header)
    for name, builder in (
        ("whole runbook", whole_index),
        ("by section", section_index),
        (f"fixed {STRAWMAN_CHUNK}ch", fixed_index),
    ):
        count = len(builder(None).passages)
        cells = []
        for k in (1, 3, 5):
            mean, _ = context_recall(builder, limit=k, floor=0.0)
            cells.append(f"{mean:.2f}")
        print(f"{name:14s} {count:9d}  " + "  ".join(f"{c:>4s}" for c in cells))

    print("\nPer incident, whole runbook at k=3 (floor off):")
    _, per = context_recall(whole_index, limit=3, floor=0.0)
    for incident_id, score in per.items():
        print(f"   {incident_id}  {score:.2f}")

    rule("3. WHAT THE FLOOR COSTS AND BUYS")
    for floor in (0.0, FLOOR):
        mean, per = context_recall(whole_index, limit=LIMIT, floor=floor)
        empty = sum(
            1
            for incident in load_incidents()
            if not search(
                whole_index(incident.incident_id),
                incident.report,
                limit=LIMIT,
                floor=floor,
            )
        )
        answered = sum(
            1
            for query in UNANSWERABLE
            if search(whole_index(None), query, limit=LIMIT, floor=floor)
        )
        label = "no floor (strawman)" if floor == 0.0 else f"floor={floor}"
        print(
            f"   {label:22s} recall={mean:.2f}  "
            f"incidents refused={empty}/6  "
            f"unanswerable queries answered={answered}/{len(UNANSWERABLE)}"
        )

    rule("4. THE SUBSTITUTION TEST, APPLIED TO THE QUERY")
    for incident in load_incidents():
        label = LABELS[incident.incident_id]
        index = whole_index(incident.incident_id)
        for tag, query in (
            ("report only ", incident.report),
            ("+service    ", f"{label.service} {incident.report}"),
        ):
            hits = search(index, query, limit=LIMIT, floor=0.0)
            context = normalized(" ".join(hit.passage.text for hit in hits))
            found = sum(1 for span in label.answers if normalized(span) in context)
            top = hits[0].passage.passage_id if hits else "-"
            print(
                f"   {incident.incident_id} {tag} recall={found}/{len(label.answers)}"
                f"  top={top}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
