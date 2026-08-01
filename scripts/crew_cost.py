"""What does splitting one agent into three roles cost, before it answers anything?

This is the Break-Even Test, and it is the whole of Chapter 10's cost argument.

The investigator in `crew.py` is Chapter 9's agent, composed rather than
reimplemented, so its cost is already measured and does not need measuring
again. Everything a crew adds on top of it is a handoff, and a handoff is a
string this script can build and count without asking a model for anything. That
is the point worth taking away: the coordination cost of a multi-agent system is
computable before you build it, while the benefit is not computable at all.

WHAT RUNS HERE. Chapter 9's recorded investigations are replayed through
`agent.pursue` unchanged - real rounds, real tool results, no key and no spend -
and the real handoff strings are constructed from what came back. The default
report is in CHARACTERS, which is free. `--tokens` converts them with
`count_tokens`, which needs a key, generates nothing, and is billed for nothing.

WHAT DOES NOT RUN. The reviewer and the remediator are never called. Their INPUT
is exactly what this script counts, because their input is the handoff. Their
OUTPUT is not counted and cannot be, which makes every crew total below a FLOOR -
the same honest limit `scripts/build_usage.py` already carries for replayed
output, and for the same reason: thinking is billed as output and is in no
recording.

THE ARITHMETIC. A crew is worth its coordination cost only if it is more accurate
in proportion:

    accuracy_crew / accuracy_agent  >=  cost_crew / cost_agent

Call the right-hand side R. Since an accuracy cannot exceed 1.0, a crew can only
break even when the single agent it replaces is already scoring below 1/R. Above
that line the experiment is decided before it is run, and the money it would have
cost is the first thing this test saves you.

Usage:

    python scripts/crew_cost.py
    python scripts/crew_cost.py --tokens
    python scripts/crew_cost.py --carry claim
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from escalation_ladder.agent import pursue
from escalation_ladder.crew import (
    REMEDIATE_SYSTEM,
    REVIEW_SYSTEM,
    Review,
    _settled_verdict,
    remediate_user,
    review_user,
)
from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.llm import RecordedCompleter, Usage

RECORDINGS = Path(__file__).resolve().parent.parent / "tests" / "recordings"

# Chapter 9's published row, which this script does not re-measure and must not
# silently diverge from. `tests/test_crew.py` asserts these against the manifest
# figures, so a corpus change breaks a test before it invalidates the prose.
AGENT_TOKENS = 20_158
AGENT_ROUTED = 12
AGENT_RUNS = 15

# A stand-in review used only to BUILD the second handoff string. Nothing is
# asked of a model to produce it, and its content is deliberately typical rather
# than favourable: a shorter endorsement would understate the handoff it creates.
SPECIMEN_REVIEW = Review(
    endorsed=True,
    reason="The cited value appears in the tool output and matches the report.",
    quoted="placeholder",
)


@dataclass(frozen=True)
class Row:
    incident: str
    rounds: int
    routed: bool
    agent_requests: int
    crew_requests: int
    handoff_one: int
    handoff_two: int

    @property
    def coordination(self) -> int:
        return self.handoff_one + self.handoff_two


def _recorded() -> RecordedCompleter:
    """Chapter 9's recordings, with their real token counts where available."""
    recordings = json.loads((RECORDINGS / "ch09_agent.json").read_text())
    usage_path = RECORDINGS / "usage.json"
    by_key: dict[str, Usage] = {}
    if usage_path.exists():
        raw = json.loads(usage_path.read_text())
        for key, value in raw.items():
            if isinstance(value, dict) and "input_tokens" in value:
                by_key[key] = Usage(
                    input_tokens=int(value["input_tokens"]),
                    output_tokens=int(value["output_tokens"]),
                )
    return RecordedCompleter(recordings=recordings, usage_by_key=by_key)


@dataclass
class Spy:
    """Wraps the replay and keeps every prompt it was asked for.

    The prompts are the point. Each one that appears in `usage.json` carries a
    REAL `count_tokens` figure, measured once by `scripts/build_usage.py`, so a
    replayed run yields matched (characters, tokens) pairs for this exact corpus.
    That is what turns the handoff's character count into a token count without a
    key and without a request.
    """

    inner: RecordedCompleter
    seen: list[tuple[str, str]] = field(default_factory=list)

    def parse(self, **kwargs):  # type: ignore[no-untyped-def]
        return self.inner.parse(**kwargs)

    def invoke(self, *, system: str, user: str, **kwargs):  # type: ignore[no-untyped-def]
        self.seen.append((system, user))
        return self.inner.invoke(system=system, user=user, **kwargs)


def calibrate(seen: list[tuple[str, str]], priced: dict[str, dict]) -> float | None:
    """Characters per token, measured on this corpus rather than assumed.

    The folk ratio of four characters per token is wrong here by more than a
    factor of two: these prompts are tool schemas, service identifiers, and
    timestamps rather than English prose. Measuring it costs nothing, because
    `usage.json` already holds a real count for every prompt on the costed path.

    ONE HONEST BIAS, stated because it runs against this script's own
    conclusion. A priced token count includes the TOOL SCHEMAS sent with the
    request; the character count of the system and user text does not. So the
    ratio comes out lower than the true characters-per-token of the prose, and
    dividing by it OVERSTATES the handoff. The derived coordination cost below
    is therefore an upper bound, which makes the crew look more expensive than
    it is - the opposite of the direction a cost argument should round in, so it
    is named here rather than buried.
    """
    from escalation_ladder.llm import prompt_key

    pairs = [
        (len(system) + len(user), priced[key]["input_tokens"])
        for system, user in seen
        if (key := prompt_key(system=system, user=user)) in priced
    ]
    if not pairs:
        return None
    return pairs[-1][0] / pairs[-1][1]


def measure(incident: Incident, completer: RecordedCompleter, *, carry: str) -> Row | None:
    """Replay one investigation and build the handoffs it would have produced."""
    investigation = pursue(incident, completer)
    verdict = _settled_verdict(investigation)
    if verdict is None:
        # The investigator refused, so no handoff is ever built. That is a real
        # and cheap outcome rather than a gap in the data, and reporting it as
        # zero coordination would be the truth - but it would also average a
        # refusal into a row about what coordination costs, which is the
        # measurement-honesty defect this repo has now hit three times.
        return None

    first = review_user(incident, verdict, investigation, carry=carry)  # type: ignore[arg-type]
    second = remediate_user(verdict, SPECIMEN_REVIEW)
    return Row(
        incident=incident.incident_id,
        rounds=investigation.rounds,
        routed=investigation.routed,
        agent_requests=investigation.api_requests,
        # The investigator's requests, plus one `parse` for the reviewer and two
        # for the remediator's single `invoke`. See `Resolution.api_requests`.
        crew_requests=investigation.api_requests + 1 + 2,
        handoff_one=len(REVIEW_SYSTEM) + len(first),
        handoff_two=len(REMEDIATE_SYSTEM) + len(second),
    )


def _count_tokens(text: str) -> int:
    """Price one string with `count_tokens`. Needs a key; generates nothing."""
    import anthropic

    from escalation_ladder.llm import MODEL

    client = anthropic.Anthropic()
    reply = client.messages.count_tokens(
        model=MODEL, messages=[{"role": "user", "content": text}]
    )
    return int(reply.input_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carry",
        choices=("results", "claim"),
        default="results",
        help="what the investigator hands the reviewer",
    )
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="convert characters to tokens with count_tokens (needs a key, spends nothing)",
    )
    args = parser.parse_args()

    completer = _recorded()
    rows = [
        row
        for row in (
            measure(incident, completer, carry=args.carry)
            for incident in load_incidents()
        )
        if row is not None
    ]
    if not rows:
        print("not measured: no investigation reached a verdict")
        return

    print(f"carry={args.carry}, {len(rows)} incidents reached a handoff\n")
    print("| Incident | Rounds | Agent requests | Crew requests | Handoff 1 | Handoff 2 |")
    print("|:--|--:|--:|--:|--:|--:|")
    for row in rows:
        print(
            f"| {row.incident} | {row.rounds} | {row.agent_requests} | "
            f"{row.crew_requests} | {row.handoff_one:,} | {row.handoff_two:,} |"
        )

    mean_chars = sum(row.coordination for row in rows) / len(rows)
    mean_agent_req = sum(row.agent_requests for row in rows) / len(rows)
    mean_crew_req = sum(row.crew_requests for row in rows) / len(rows)
    print(f"\nmean coordination: {mean_chars:,.0f} characters")
    print(
        f"mean API requests: agent {mean_agent_req:.2f} -> crew {mean_crew_req:.2f} "
        f"({mean_crew_req / mean_agent_req:.2f}x)"
    )

    # No characters-to-tokens estimate is printed, and the omission is
    # deliberate. The folk ratio of four characters per token is wrong on this
    # corpus by a factor of two and a half - these prompts are tool schemas,
    # identifiers, and timestamps rather than English - and a book that measures
    # its numbers cannot publish one it guessed. Characters are exact and free;
    # tokens need `--tokens`, which needs a key and still spends nothing.
    print("\n--- the Break-Even Test ---")
    print(f"agent tokens per incident        {AGENT_TOKENS:,} (Chapter 9, measured)")
    added_input = (
        _priced(rows, args.carry) if args.tokens else _derived(args.carry)
    )
    if added_input is None:
        print("crew added input per incident    not measured (no priced prompts)")
        print("cost ratio R                     not measured")
        _headroom(None)
        return
    ratio = (AGENT_TOKENS + added_input) / AGENT_TOKENS
    agent_accuracy = AGENT_ROUTED / AGENT_RUNS
    how = "count_tokens" if args.tokens else "derived, free"
    print(f"crew added input per incident    {added_input:,.0f} ({how}; INPUT only)")
    print(f"cost ratio R                     {ratio:.3f}x")
    print(f"agent accuracy on this corpus    {agent_accuracy:.3f} ({AGENT_ROUTED}/{AGENT_RUNS})")
    required = agent_accuracy * ratio
    print(f"crew must reach                  {required:.3f} to break even")
    print(f"an agent must score below        {1.0 / ratio:.3f} for a crew to have headroom")

    # Back into runs. An accuracy is a fraction of a finite corpus, so the target
    # rounds UP to a whole run - and on a small corpus that rounding is most of
    # the answer. Reporting the fraction alone makes a target of 0.952 look like
    # a nudge when it is in fact a demand for a perfect score.
    import math

    needed = math.ceil(required * AGENT_RUNS - 1e-9)
    print(
        f"in runs                          {needed} of {AGENT_RUNS} "
        f"(agent got {AGENT_ROUTED}); {needed - AGENT_ROUTED} must convert"
    )
    if needed >= AGENT_RUNS:
        print("                                 -> that is a PERFECT score")
    _headroom(required)


# Which incidents the crew would have to convert, and which rung owns the
# evidence each one needs. Chapter 9 measured the first column; the second is
# read off Chapter 7's and Chapter 8's per-incident tables, which agree.
FAILURES: dict[str, str] = {
    "INC-1043": "Level 2 - the answer is in a runbook, and no role here has a retriever",
}


def _headroom(required: float | None) -> None:
    """Step two: name the failures a crew would have to convert, and ask if it can.

    A cost ratio tells you the accuracy to beat. It does not tell you whether
    beating it is reachable, and that question is answered by looking at WHICH
    requests are currently failing rather than at how many. This is the half of
    the Break-Even Test that a spreadsheet does not do for you, and it is
    usually where the answer is.
    """
    print("\n--- what the crew would have to convert ---")
    for incident, why in FAILURES.items():
        print(f"  {incident}: {why}")
    if required is not None and required > 1.0:
        print(
            "\nBreaking even needs an accuracy above 1.0. The experiment is decided "
            "before\nit is run."
        )
    print(
        "\nEvery failure on this corpus needs a capability no role in the crew has.\n"
        "Splitting authority does not add a retriever, so the accuracy the cost ratio\n"
        "demands is not reachable by this architecture at any price. Buy the crew for\n"
        "the authority boundary, or do not buy it."
    )


def _derived(carry: str) -> float | None:
    """Mean added INPUT tokens per incident, with no key and no request.

    Each incident's own last priced prompt supplies its characters-per-token, so
    the conversion is calibrated per incident rather than across the corpus - the
    ratio drifts from 1.51 to 1.73 as a working set accumulates, and a single
    corpus-wide constant would mis-price the long investigations most.
    """
    usage_path = RECORDINGS / "usage.json"
    if not usage_path.exists():
        return None
    priced = json.loads(usage_path.read_text())
    recordings = json.loads((RECORDINGS / "ch09_agent.json").read_text())

    totals: list[float] = []
    for incident in load_incidents():
        spy = Spy(RecordedCompleter(recordings=recordings))
        investigation = pursue(incident, spy)
        verdict = _settled_verdict(investigation)
        if verdict is None:
            continue
        ratio = calibrate(spy.seen, priced)
        if ratio is None:
            continue
        first = len(REVIEW_SYSTEM) + len(
            review_user(incident, verdict, investigation, carry=carry)  # type: ignore[arg-type]
        )
        second = len(REMEDIATE_SYSTEM) + len(remediate_user(verdict, SPECIMEN_REVIEW))
        totals.append((first + second) / ratio)
    if not totals:
        return None
    return sum(totals) / len(totals)


def _priced(rows: list[Row], carry: str) -> float | None:
    """Mean added INPUT tokens per incident, counted rather than estimated.

    Returns None on an empty ledger, exactly as `_derived` does. It used to
    return 0.0, which the caller's `is None` guard waved straight through: zero
    priced prompts printed `crew added input 0` and `cost ratio R 1.000x` - a
    complete Break-Even result with no measurement under it. That is this
    project's recurring defect arriving through one more door, in the file whose
    own docstrings claim to have shut it. A partial measurement does not look
    partial, it looks CHEAP.
    """
    completer = _recorded()
    totals: list[int] = []
    by_id = {incident.incident_id: incident for incident in load_incidents()}
    for row in rows:
        incident = by_id[row.incident]
        investigation = pursue(incident, completer)
        verdict = _settled_verdict(investigation)
        if verdict is None:
            continue
        first = review_user(incident, verdict, investigation, carry=carry)  # type: ignore[arg-type]
        second = remediate_user(verdict, SPECIMEN_REVIEW)
        totals.append(
            _count_tokens(REVIEW_SYSTEM + "\n" + first)
            + _count_tokens(REMEDIATE_SYSTEM + "\n" + second)
        )
    if not totals:
        return None
    return sum(totals) / len(totals)


if __name__ == "__main__":
    main()
