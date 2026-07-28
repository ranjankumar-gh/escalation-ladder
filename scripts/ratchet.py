"""How steep is the Context Ratchet, and what decides its slope?

Chapter 8's named concept says that within one run, intermediate state only
accumulates - nothing is removed and nothing is retracted - so every step
re-buys every earlier step. That is a claim about cost, and this measures it
two ways.

The SHIPPED chain carries the original report, the surviving claims, and a list
of the tool calls already made. It does not re-send tool RESULTS. So its growth
across steps is roughly the size of one claim per step: real, and shallow.

The NAIVE chain - the one most people write first, because it is what a chat
transcript looks like - carries everything, results included. This script builds
both messages for the same recorded run and counts them, so the difference is a
measured ratio rather than an intuition about big-O.

The slope of the ratchet is a design decision, and this is the number that shows
which decision you made.

Counting uses the API's token counter, so it needs a key. It generates nothing
and costs nothing: no completion is requested, and the run being measured is
replayed from tests/recordings rather than re-run.

Usage:

    python scripts/ratchet.py
    python scripts/ratchet.py --incident INC-1044
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anthropic

from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import MODEL, RecordedCompleter
from escalation_ladder.reasoning import STEPS, SYSTEM, carried, reason

RECORDINGS = Path(__file__).parent.parent / "tests" / "recordings" / "ch08_reasoning.json"


def naive(message: str, results: list[str]) -> str:
    """The shipped message plus every tool result gathered so far.

    Built by APPENDING to the real message rather than by writing a second
    prompt, so the two arms cannot drift apart and the difference is exactly the
    results and nothing else.
    """
    if not results:
        return message
    block = "\n".join(f"<result>\n{text}\n</result>" for text in results)
    return f"{message}\n\n<everything read so far>\n{block}\n</everything read so far>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incident", default="INC-1044")
    args = parser.parse_args()

    if not RECORDINGS.exists():
        print(f"no recording at {RECORDINGS}")
        return 1

    incidents = {i.incident_id: i for i in load_incidents()}
    case = incidents[args.incident]
    replayed = reason(case, RecordedCompleter(recordings=json.loads(
        RECORDINGS.read_text(encoding="utf-8")
    )))

    client = anthropic.Anthropic()

    def count(text: str) -> int:
        response = client.messages.count_tokens(
            model=MODEL,
            system=SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens

    # Walk the recorded run again, rebuilding each step's message from the state
    # as it stood when that step ran.
    from escalation_ladder.reasoning import ChainState

    state = ChainState(incident=case)
    seen: list[str] = []
    rows = []
    by_id = {r.call_id: r for r in replayed.results}

    for index, record in enumerate(replayed.steps):
        step = STEPS[index]
        shipped_message = carried(state, step, index, len(replayed.steps))
        rows.append(
            {
                "step": step.name,
                "shipped_tokens": count(shipped_message),
                "naive_tokens": count(naive(shipped_message, seen)),
            }
        )
        state = ChainState(
            incident=case,
            records=state.records + (record,),
            calls=state.calls + record.calls,
        )
        for call in record.calls:
            result = by_id.get(call.call_id)
            if result is not None:
                seen.append(result.content)

    print(f"{args.incident}: {len(rows)} steps, tool results carried forward\n")
    print(f"{'step':<10}{'shipped':>10}{'naive':>10}{'ratio':>9}")
    for row in rows:
        ratio = row["naive_tokens"] / row["shipped_tokens"]
        print(f"{row['step']:<10}{row['shipped_tokens']:>10}"
              f"{row['naive_tokens']:>10}{ratio:>8.2f}x")

    shipped_total = sum(r["shipped_tokens"] for r in rows)
    naive_total = sum(r["naive_tokens"] for r in rows)
    print(f"\n{'total':<10}{shipped_total:>10}{naive_total:>10}"
          f"{naive_total / shipped_total:>8.2f}x")

    first, last = rows[0], rows[-1]
    print(
        f"\ngrowth across the chain: shipped "
        f"{last['shipped_tokens'] / first['shipped_tokens']:.2f}x, naive "
        f"{last['naive_tokens'] / first['naive_tokens']:.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
