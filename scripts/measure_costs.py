"""Regenerate the cross-rung cost comparison table.

Used by Chapter 2's comparison table and Appendix B. The book's cost claims are measured,
not asserted, so this script is the source of those numbers -- and a reader can re-run it to
get today's. That turns a decay risk into a trust feature: the published table says what it
was measured against, and the command that regenerates it.

Absolute pricing is deliberately NOT computed here. Prose states ratios between rungs;
per-million-token rates live in Appendix B, so a price change is one edit in one place.

Usage:
    python -m scripts.measure_costs                 # print to stdout
    python -m scripts.measure_costs --out table.md  # write to a file
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Allow `python scripts/measure_costs.py` from the repo root as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from escalation_ladder.fixtures.incidents import load_incidents  # noqa: E402
from escalation_ladder.instrument import CostLedger  # noqa: E402
from escalation_ladder.rungs import load_all  # noqa: E402

COLUMNS: tuple[str, ...] = (
    "Rung",
    "Model calls",
    "Input tokens",
    "Output tokens",
    "p50 latency (ms)",
    "p99 latency (ms)",
)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Zero for an empty series."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[k]


def summarise_ledgers(rung: str, ledgers: list[CostLedger]) -> dict[str, object]:
    """Collapse one rung's per-incident ledgers into a single table row.

    Tokens and call counts are averaged per incident; latency is reported as percentiles,
    because the tail is what actually hurts in production and an average hides it.
    """
    if not ledgers:
        row: dict[str, object] = dict.fromkeys(COLUMNS, 0)
        row["Rung"] = rung
        return row
    n = len(ledgers)
    latencies = [led.total_latency_ms for led in ledgers]
    return {
        "Rung": rung,
        "Model calls": round(sum(led.model_calls for led in ledgers) / n),
        "Input tokens": round(sum(led.total_input_tokens for led in ledgers) / n),
        "Output tokens": round(sum(led.total_output_tokens for led in ledgers) / n),
        "p50 latency (ms)": round(percentile(latencies, 50), 1),
        "p99 latency (ms)": round(percentile(latencies, 99), 1),
    }


def render_markdown_table(rows: list[dict[str, object]]) -> str:
    """Render rows as a GitHub-flavoured markdown table. Headers are always present."""
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in COLUMNS) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the cross-rung cost table.")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the table here instead of stdout")
    args = parser.parse_args(argv)

    registry = load_all()
    incidents = load_incidents()

    rows: list[dict[str, object]] = []
    for rung, fn in registry.items():
        ledgers: list[CostLedger] = []
        for incident in incidents:
            try:
                ledgers.append(fn(incident))
            except Exception as exc:
                # A rung failing on an incident it cannot handle is the point of the book,
                # not a crash. Record it visibly and keep measuring the rest.
                print(f"  {rung} failed on {incident.incident_id}: {exc}", file=sys.stderr)
        rows.append(summarise_ledgers(rung, ledgers))

    table = render_markdown_table(rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} rungs)")
    else:
        print(table)

    if not rows:
        print("\nNo rungs registered yet. Rung modules land from Chapter 3 onward.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
