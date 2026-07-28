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
import inspect
import json
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

RECORDINGS = Path(__file__).resolve().parent.parent / "tests" / "recordings"
USAGE = RECORDINGS / "usage.json"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Zero for an empty series."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[k]


def summarise_ledgers(
    rung: str, ledgers: list[CostLedger], *, replayed: bool = False
) -> dict[str, object]:
    """Collapse one rung's per-incident ledgers into a single table row.

    Tokens and call counts are averaged per incident; latency is reported as percentiles,
    because the tail is what actually hurts in production and an average hides it.

    `replayed` blanks the two latency columns, and refusing to fill them is the
    point rather than a gap. A replay's elapsed time is the cost of reading a
    JSON file, so publishing it as p50 would say Level 6 answers in under a
    millisecond - a number that is real, reproducible, and a lie about the
    system. Tokens and call counts survive replay because they are properties of
    the request; latency is a property of the network, and the network is not
    here. Chapters 4 through 9 carry the measured latencies.

    A rung that made NO model calls keeps its latency even under replay, and
    that exception is not a special case for Level 0 - it is the same rule. The
    blanking exists because the network is missing; a rung that never touched
    the network is not missing anything. The Zero Row is measured on a replay
    for exactly the reason it is free.
    """
    if not ledgers:
        row: dict[str, object] = dict.fromkeys(COLUMNS, 0)
        row["Rung"] = rung
        return row

    # A rung that attempted calls and was billed for none of them measured nothing:
    # every call failed. Averaging that to zeros would seat it in the table looking
    # exactly like Level 0, which genuinely is free -- so a reader with no API key
    # would read "Level 1: 0 tokens, 0ms" as a result rather than as a missing key.
    attempted = sum(led.model_calls for led in ledgers)
    billed = sum(led.total_input_tokens for led in ledgers)
    if attempted and not billed:
        unmeasured: dict[str, object] = dict.fromkeys(COLUMNS, "not measured")
        unmeasured["Rung"] = rung
        return unmeasured

    n = len(ledgers)
    latencies = [led.total_latency_ms for led in ledgers]
    # `attempted` is recomputed rather than reused from the guard above, which
    # returns early: reaching here means the rung was billed, or made no calls
    # at all. Only the first case has a latency the replay cannot know.
    off_network = replayed and sum(led.model_calls for led in ledgers) > 0
    return {
        "Rung": rung,
        "Model calls": round(sum(led.model_calls for led in ledgers) / n),
        "Input tokens": round(sum(led.total_input_tokens for led in ledgers) / n),
        "Output tokens": round(sum(led.total_output_tokens for led in ledgers) / n),
        "p50 latency (ms)": (
            "not measured" if off_network else round(percentile(latencies, 50), 1)
        ),
        "p99 latency (ms)": (
            "not measured" if off_network else round(percentile(latencies, 99), 1)
        ),
    }


def replay_completer() -> object:
    """A completer that answers from the recordings and prices from the sidecar.

    Both files are committed, so this path needs no key, no network, and no
    spend - which is what lets a reader regenerate the table that the book's
    cost claims rest on rather than taking them on faith.
    """
    from escalation_ladder.llm import RecordedCompleter, Usage

    recordings: dict[str, str] = {}
    for path in sorted(RECORDINGS.glob("ch*.json")):
        recordings.update(json.loads(path.read_text(encoding="utf-8")))

    priced: dict[str, Usage] = {}
    if USAGE.exists():
        for key, entry in json.loads(USAGE.read_text(encoding="utf-8")).items():
            priced[key] = Usage(
                int(entry["input_tokens"]), int(entry["output_tokens"])
            )
    else:
        print(f"no {USAGE.name}; token columns will be zero. "
              "Run scripts/build_usage.py once to create it.", file=sys.stderr)

    return RecordedCompleter(recordings=recordings, usage_by_key=priced)


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
    parser.add_argument("--replay", action="store_true",
                        help="answer from the committed recordings: no key, no spend")
    args = parser.parse_args(argv)

    registry = load_all()
    incidents = load_incidents()
    completer = replay_completer() if args.replay else None

    rows: list[dict[str, object]] = []
    for rung, fn in registry.items():
        # Level 0 has no seam to pass a completer through, and that asymmetry is
        # the Zero Row being honest rather than an inconsistency to smooth over.
        takes_completer = "completer" in inspect.signature(fn).parameters
        ledgers: list[CostLedger] = []
        for incident in incidents:
            try:
                if completer is not None and takes_completer:
                    ledgers.append(fn(incident, completer))  # type: ignore[call-arg]
                else:
                    ledgers.append(fn(incident))
            except Exception as exc:
                # A rung failing on an incident it cannot handle is the point of the book,
                # not a crash. Record it visibly and keep measuring the rest.
                print(f"  {rung} failed on {incident.incident_id}: {exc}", file=sys.stderr)
        rows.append(summarise_ledgers(rung, ledgers, replayed=args.replay))

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
