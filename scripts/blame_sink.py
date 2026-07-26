"""Chapter 6's blame-sink experiment: make stage 1 wrong, see who gets blamed.

Over 100 healthy runs the pipeline never exhibited the blame sink, and the reason
is worth stating rather than hiding: on this corpus the only stage that ever
refused was stage 1, and a stage that refuses is attributed to itself correctly
by the crudest possible counter. The blame sink needs an upstream stage that is
CONFIDENTLY WRONG rather than honestly unsure - which is Chapter 4's Format
Illusion supplying the fuel for Chapter 6's failure mode.

So this script supplies one. It replaces stage 1's service with a plausible wrong
one, leaves every other stage untouched, and reports which stage stopped the run
against which stage caused it.

Two shapes come out, and they cost different amounts:

  * `retrieve` reports, `classify` caused it. Nothing cleared the floor, and no
    second call was made. The cheap shape.
  * `draft` reports, `classify` caused it. Passages were retrieved, they were
    about the wrong thing, the drafter noticed and refused. The expensive shape,
    and the one a dashboard misreads as a drafting problem.

Run:  python scripts/blame_sink.py            # replays, no key, no spend
      python scripts/blame_sink.py --live     # re-records, needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from escalation_ladder.classify import Classification
from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.llm import Completer, Completion, RecordedCompleter, prompt_key
from escalation_ladder.retrieve import default_index
from escalation_ladder.rules import route
from escalation_ladder.workflow import DRAFT_SYSTEM, advise

RECORDINGS = Path(__file__).resolve().parent.parent / "tests" / "recordings"
CH04 = RECORDINGS / "ch04_classify.json"
CH06 = RECORDINGS / "ch06_workflow.json"

T = TypeVar("T", bound=BaseModel)

# One plausible wrong service per incident. Plausible matters: a wrong answer
# nobody would give proves nothing about a failure mode built on confident ones.
INJECTED: dict[str, str] = {
    # the charge is real, and the checkout page is what the customer watched fail
    "INC-1042": "checkout-api",
    # "feels slow" is said about checkout far more often than about search
    "INC-1043": "checkout-api",
    # the report names the payment gateway in its own text
    "INC-1044": "payment-gateway",
    # an order confirmation email follows an order
    "INC-1045": "checkout-api",
}


@dataclass
class Misclassifier:
    """Answers stage 1 with a chosen wrong service, and passes stage 4 through.

    Patching the completer rather than the pipeline is deliberate: `advise` runs
    unmodified, so what this measures is the shipped code path and not a
    lookalike written for the experiment.
    """

    inner: Completer
    service: str

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "low",
    ) -> Completion[T]:
        completion = self.inner.parse(
            system=system, user=user, schema=schema, effort=effort
        )
        if schema is Classification and completion.parsed is not None:
            patched = completion.parsed.model_copy(update={"service": self.service})
            return Completion(patched, completion.usage, completion.failed)
        return completion


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def replaying() -> RecordedCompleter:
    """Chapter 4's and Chapter 6's recordings together. No key, no spend."""
    merged = {k: v for k, v in _load(CH04).items() if isinstance(v, str)}
    merged.update({k: v for k, v in _load(CH06).items() if isinstance(v, str)})
    return RecordedCompleter(recordings=merged)


class Recording:
    """Replays what is already recorded, and only calls live for what is not.

    Preferring the recording is not an optimization, it is a correctness
    requirement, and the first version of this script got it wrong in a way this
    chapter happens to be about. Calling stage 1 live re-recorded a
    classification whose summary differed slightly from the one on disk. Same
    prompt, same key, different value - so the query changed, so the retrieved
    passages changed, so every stage 4 recording downstream was keyed to a prompt
    that no longer occurs. One live call at stage 1 orphaned the whole chapter's
    stage 4 evidence, and the test suite said so.
    """

    def __init__(self, inner: Completer, recorded: dict[str, str]) -> None:
        self.inner = inner
        self.recorded = recorded
        self.captured: dict[str, str] = {}
        self.live_calls = 0

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "low",
    ) -> Completion[T]:
        key = prompt_key(system=system, user=user)
        if key in self.recorded:
            return RecordedCompleter(recordings=self.recorded).parse(
                system=system, user=user, schema=schema, effort=effort
            )
        self.live_calls += 1
        completion = self.inner.parse(
            system=system, user=user, schema=schema, effort=effort
        )
        if completion.parsed is not None:
            self.captured[key] = completion.parsed.model_dump_json()
        return completion


def incidents() -> list[Incident]:
    return [i for i in load_incidents() if not route(i).routed]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.live:
        from escalation_ladder.llm import AnthropicCompleter

        base: Completer = Recording(AnthropicCompleter(), replaying().recordings)
    else:
        base = replaying()

    print(
        f"{'incident':<10} {'injected':<20} {'reported by':<12} "
        f"{'origin':<10} outcome"
    )
    rows: list[tuple[str, str, str]] = []
    for incident in incidents():
        wrong = INJECTED[incident.incident_id]
        result = advise(incident, Misclassifier(base, wrong), default_index(incident))
        reported = result.reported_by or "-"
        origin = result.origin or "-"
        if not result.routed:
            outcome = "stopped"
        elif result.degraded:
            outcome = f"DEGRADED -> {result.decision.rota}"
        else:
            outcome = f"ROUTED -> {result.decision.rota}"
        rows.append((incident.incident_id, reported, origin))
        print(
            f"{incident.incident_id:<10} {wrong:<20} {reported:<12} "
            f"{origin:<10} {outcome}"
        )
        for record in result.trace:
            mark = "ok  " if record.ok else "stop"
            print(f"    {record.name:<9} {mark} {record.detail[:80]}")

    stopped = [r for r in rows if r[1] != "-"]
    diverged = [r for r in stopped if r[1] != r[2]]
    print(f"\nstopped runs           : {len(stopped)}/{len(rows)}")
    print(f"reported_by != origin  : {len(diverged)}/{len(stopped)}")
    if stopped:
        by_site: dict[str, int] = {}
        by_origin: dict[str, int] = {}
        for _, site, origin in stopped:
            by_site[site] = by_site.get(site, 0) + 1
            by_origin[origin] = by_origin.get(origin, 0) + 1
        print(f"by reporting stage     : {by_site}")
        print(f"by origin              : {by_origin}")

    if args.live and isinstance(base, Recording) and base.captured:
        existing = _load(CH06)
        existing.update(base.captured)
        CH06.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"\nrecorded {len(base.captured)} prompts into {CH06.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
