"""What every climb in this book cost below itself, read from the repo's tags.

Free. No key, no network, no spend, and every figure in Chapter 15 regenerates
on a fresh clone that has the tags:

    python -m scripts.migration_cost

Point it at your own repository and your own release tags to get the same
number for your own system:

    python -m scripts.migration_cost --tags v1.4.0 v1.5.0 v1.6.0

Like `scripts/observe_rungs.py` this registers no rung and appears in no
cross-rung cost table.
"""
from __future__ import annotations

import argparse

from escalation_ladder import migrate as mg

LEVEL_OF_TAG: dict[str, str] = {
    "ch03": "L0",
    "ch04": "L1",
    "ch05": "L2",
    "ch06": "L3",
    "ch07": "L4",
    "ch08": "L5",
    "ch09": "L6",
    "ch10": "L7",
}


def label(tag: str) -> str:
    """`ch07-tool-using-llm` reads as `Ch7 L4`; a non-rung chapter has no
    level."""
    stem = tag.split("-")[0]
    number = stem.removeprefix("ch").lstrip("0")
    return f"Ch{number} {LEVEL_OF_TAG.get(stem, '')}".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--scope", default=mg.PACKAGE)
    args = parser.parse_args()

    tags = tuple(args.tags) if args.tags else mg.chapter_tags()
    measured = mg.climbs(tags, scope=args.scope)

    print("\n== what each climb cost below itself ==\n")
    print("| climb | added | below | seam | fixture | consumer | rung |")
    print("|---|---|---|---|---|---|---|")
    totals = {kind: 0 for kind in mg.BELOW}
    added_total = 0
    for climb in measured:
        if not climb.measured:
            print(f"| {label(climb.tag)} | not measured: {climb.decided_by} |")
            continue
        added_total += climb.added
        for kind in mg.BELOW:
            totals[kind] += climb.below(kind)
        cells = " | ".join(f"{climb.below(k) or '-'}" for k in mg.BELOW)
        print(
            f"| {label(climb.tag)} | {climb.added:,} | "
            f"{climb.below() or '-'} | {cells} |"
        )
    below_total = sum(totals.values())
    cells = " | ".join(f"{totals[k]}" for k in mg.BELOW)
    print(f"| **total** | **{added_total:,}** | **{below_total}** | {cells} |")

    if below_total:
        print("\n== share of the below-edits, by role ==\n")
        for kind in mg.BELOW:
            share = 100 * totals[kind] / below_total
            print(f"{kind:9} {totals[kind]:5}  {share:4.1f}%")

    print("\n== every line that landed in a rung module ==\n")
    for climb in measured:
        for change in climb.changes:
            if change.kind == "rung":
                print(
                    f"{label(climb.tag):8} {change.path:34} "
                    f"+{change.added} -{change.removed}"
                )

    print("\n== the Addition Test on the current tree ==\n")
    verdict = mg.addition_test(measured[-1] if measured else None)
    print(f"passed: {verdict.passed}")
    print(f"decided_by: {verdict.decided_by}")


if __name__ == "__main__":
    main()
