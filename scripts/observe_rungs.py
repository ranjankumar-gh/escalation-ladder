"""Instrument every rung against the recordings and report what the record answers.

Free. No key, no network, no spend, and every figure in Chapter 14 regenerates
on a fresh clone:

    python -m scripts.observe_rungs

Like `scripts/evaluate_rungs.py` this registers no rung and appears in no
cross-rung cost table. It is a reporting harness, not an architecture.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Iterable

from escalation_ladder import evaluate as ev
from escalation_ladder import telemetry as tel
from escalation_ladder.agent import pursue
from escalation_ladder.classify import triage
from escalation_ladder.crew import resolve
from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.instrument import CostLedger
from escalation_ladder.llm import MeteredCompleter
from escalation_ladder.reasoning import reason
from escalation_ladder.retrieve import advise as advise_grounded
from escalation_ladder.router import ASKS, LADDERS, handle
from escalation_ladder.rules import route
from escalation_ladder.rules import run as rules_run
from escalation_ladder.tools import investigate
from escalation_ladder.workflow import advise as advise_pipeline

RUNGS: dict[int, Callable[..., Any]] = {
    1: triage,
    2: advise_grounded,
    3: advise_pipeline,
    4: investigate,
    5: reason,
    6: pursue,
    7: resolve,
}


def observe(level: int, incident: Incident) -> tuple[tel.Trace, CostLedger]:
    """Run one rung on one incident and keep both records of what happened.

    Three wrappers, and the nesting is the chapter's point. The replay answers,
    `MeteredCompleter` prices, `TracingCompleter` records provenance, and
    `witness` reads the decisions off the return value afterwards. No rung
    module knows that any of it is happening.
    """
    if level == 0:
        return tel.witness(route(incident), level=0), rules_run(incident)

    ledger = CostLedger()
    events: list[tel.Event] = []
    metered = MeteredCompleter(inner=ev.replay(), ledger=ledger, stage=f"L{level}")
    tracing = tel.TracingCompleter(
        inner=metered, events=events, incident_id=incident.incident_id, level=level
    )
    result = RUNGS[level](incident, tracing)
    own = getattr(result, "ledger", None)
    trace = tel.witness(result, level=level)
    if own is None:
        trace = trace + tel.from_ledger(
            ledger, incident_id=incident.incident_id, level=level
        )
    return trace + tel.Trace(incident.incident_id, tuple(events)), ledger


def sweep() -> tuple[dict[int, list[tel.Trace]], dict[int, list[CostLedger]]]:
    traces: dict[int, list[tel.Trace]] = {}
    ledgers: dict[int, list[CostLedger]] = {}
    for level in (0, *RUNGS):
        for incident in ev.corpus():
            trace, ledger = observe(level, incident)
            traces.setdefault(level, []).append(trace)
            ledgers.setdefault(level, []).append(ledger)
    return traces, ledgers


def composite_traces() -> list[tel.Trace]:
    out: list[tel.Trace] = []
    for ask in ASKS:
        for incident in ev.corpus():
            out.append(
                tel.witness(
                    handle(incident, ask, ev.replay()), level=-1, ladders=LADDERS
                )
            )
    return out


def ledger_traces(
    ledgers: dict[int, list[CostLedger]], *, billed_only: bool = False
) -> dict[int, list[tel.Trace]]:
    """The pre-Chapter-14 seams, promoted into traces so they can be asked the same questions.

    `billed_only` models what the vendor's own dashboard sees: requests that
    reached the API. It cannot see Level 0 at all, because Level 0 never calls
    anything - the Zero Row is invisible to vendor observability by
    construction, and that is not a gap in any product.
    """
    out: dict[int, list[tel.Trace]] = {}
    incidents = [i.incident_id for i in ev.corpus()]
    for level, batch in ledgers.items():
        for index, ledger in enumerate(batch):
            kept = CostLedger(
                [m for m in ledger.measurements if m.model_calls or not billed_only]
            )
            out.setdefault(level, []).append(
                tel.from_ledger(kept, incident_id=incidents[index], level=level)
            )
    return out


def cardinality(traces: Iterable[tel.Trace]) -> tuple[int, int]:
    reasons, kinds = set(), set()
    for trace in traces:
        for event in trace.events:
            if event.reason:
                reasons.add(event.reason)
            kinds.add(event.kind)
    return len(reasons), len(kinds)


def record_bytes(trace: tel.Trace) -> int:
    """One trace as newline-delimited JSON, which is how it would ship."""
    return sum(
        len(json.dumps({k: v for k, v in vars(event).items() if v not in ("", 0, None)}))
        + 1
        for event in trace.events
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", metavar="INCIDENT", help="print one trace and exit")
    parser.add_argument("--level", type=int, default=4, help="rung for --trace")
    args = parser.parse_args()

    if args.trace:
        incident = next(i for i in ev.corpus() if i.incident_id == args.trace)
        trace, _ = observe(args.level, incident)
        print(trace.spans())
        return

    traces, ledgers = sweep()
    every = [t for batch in traces.values() for t in batch]
    composite = composite_traces()

    print("== the counters ten chapters asked for ==\n")
    print(tel.counters(every + composite))

    print("\n== what each seam can localize ==\n")
    seams = {
        "vendor requests only": ledger_traces(ledgers, billed_only=True),
        "CostLedger (Ch3-Ch13)": ledger_traces(ledgers),
        "Event record (Ch14)": traces,
    }
    population = {q.level: tel.exercised(traces, q) for q in tel.QUESTIONS}
    print(
        "| Seam | "
        + " | ".join(f"L{q.level}" for q in tel.QUESTIONS)
        + " | Instrumentation Level |"
    )
    print("|---|" + "---|" * (len(tel.QUESTIONS) + 1))
    for name, seam in seams.items():
        marks = [
            "-"
            if not population[question.level]
            else ("yes" if tel.localizes(seam, question, population[question.level]) else "no")
            for question in tel.QUESTIONS
        ]
        level = tel.instrumentation_level(seam, traces)
        print(
            f"| {name} | "
            + " | ".join(marks)
            + f" | {'none' if level is None else level} |"
        )
    unexercised = [level for level, ids in population.items() if not ids]
    if unexercised:
        print(
            "\nnot exercised by this corpus (counted as not localized): "
            + ", ".join(f"L{level}" for level in sorted(unexercised))
        )

    print("\n== cardinality ==\n")
    reasons, kinds = cardinality(every + composite)
    print(f"distinct reason strings: {reasons}")
    print(f"distinct kinds:          {kinds}")

    print("\n== record size per request ==\n")
    print("| Rung | Events | Record bytes | Tokens | Record as share of model bytes |")
    print("|---|---:|---:|---:|---:|")
    for level in sorted(traces):
        batch = traces[level]
        events = sum(len(t.events) for t in batch) / len(batch)
        size = sum(record_bytes(t) for t in batch) / len(batch)
        tokens = sum(t.tokens for t in batch) / len(batch)
        # A rung whose recordings did not cover every incident reports a FLOOR,
        # not a measurement. Chapter 12's rule, and the fourth time this book
        # has had to apply it: the part that silently fails is the expensive
        # part, so a partial number does not look partial, it looks cheap.
        partial = any(
            event.kind == "unrecorded" for t in batch for event in t.events
        )
        # Four bytes per token, the usual English rule of thumb. Exact enough
        # for a ratio that spans three orders of magnitude.
        share = f"{100 * size / (tokens * 4):.0f}%" if tokens else "the whole cost"
        mark = "+" if partial else ""
        print(
            f"| L{level} | {events:.1f} | {size:,.0f} | {tokens:,.0f}{mark} | {share} |"
        )
    print("\n+ a floor: the recordings do not cover every incident at this rung.")

    print("\n== handoffs at Level 7 ==\n")
    for trace in traces.get(7, []):
        carried = [
            (e.index, int(dict(e.fields)["characters"]), dict(e.fields)["receiver"])
            for e in trace.events
            if e.kind == "handoff"
        ]
        if carried:
            print(f"{trace.incident_id}: " + ", ".join(
                f"#{i} -> {who} {n:,} chars" for i, n, who in carried
            ))


if __name__ == "__main__":
    main()
