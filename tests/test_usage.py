"""The behaviours this chapter argues for, applied to the cost sidecar itself.

The book's cost claims regenerate from `tests/recordings/usage.json` rather than
from a live run, which moves the honesty problem rather than solving it: a
sidecar that quietly drifted from what was actually billed would produce a table
that looks measured and is not. These tests are the tripwire.

The published figures below are the ones the chapters print, so a reader holding
the book can check them against a replay without owning the measurement harness.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import RecordedCompleter, Usage
from escalation_ladder.rungs import load_all
from scripts.build_usage import as_tool, capture, load_recordings

RECORDINGS = Path(__file__).parent / "recordings"
SIDECAR = RECORDINGS / "usage.json"

# Mean input tokens per billed incident, as measured live and printed in each
# chapter. Chapter 4 p.cost, Chapter 5, Chapter 6, Chapter 7, Chapter 8,
# Chapter 9 respectively.
PUBLISHED: dict[str, int] = {
    "Level 1": 1098,
    "Level 2": 1719,
    "Level 3": 2735,
    "Level 4": 6055,
    "Level 5": 14803,
    "Level 6": 18154,
}

# How far a replayed row may sit from the live one before it stops being
# evidence. Two effects are inside this band and both are documented: pricing
# the structured-output schema as a strict tool overshoots by roughly ten
# percent, and Level 5's recorded path settles in fewer steps than its live mean
# took. A band this wide still catches the failure that motivated the test - not
# pricing the schema at all was a fifty-four percent error.
TOLERANCE = 0.25


def _replayed() -> dict[str, float]:
    """Mean input tokens per BILLED incident, replayed from the recordings.

    Per billed incident rather than per incident, because that is the
    denominator the published figures used. Dividing by the whole corpus would
    charge each rung for the incidents it never reached and make every row look
    cheaper than it was.
    """
    recordings = load_recordings()
    priced = {
        key: Usage(int(entry["input_tokens"]), int(entry["output_tokens"]))
        for key, entry in json.loads(SIDECAR.read_text(encoding="utf-8")).items()
    }
    incidents = load_incidents()

    rows: dict[str, float] = {}
    for rung, fn in load_all().items():
        takes_completer = "completer" in inspect.signature(fn).parameters
        total = billed = 0
        for incident in incidents:
            completer = RecordedCompleter(recordings=recordings, usage_by_key=priced)
            try:
                ledger = fn(incident, completer) if takes_completer else fn(incident)
            except Exception:
                # A rung with no recordings of its own cannot be replayed, and
                # from Chapter 10 there is one: the crew's investigator replays
                # from Chapter 9's recordings and its other two roles have none.
                # Skipping leaves it out of `rows`, so it is ABSENT from the
                # published comparison rather than present with a number it did
                # not earn.
                continue
            if ledger.total_input_tokens:
                total += ledger.total_input_tokens
                billed += 1
        if billed:
            rows[rung.split(":")[0]] = total / billed
    return rows


@pytest.mark.skipif(not SIDECAR.exists(), reason="run scripts/build_usage.py once")
def test_every_costed_prompt_has_a_price() -> None:
    """A missing price is a silent zero, which reads as a free rung."""
    priced = set(json.loads(SIDECAR.read_text(encoding="utf-8")))
    assert not set(capture()) - priced


@pytest.mark.skipif(not SIDECAR.exists(), reason="run scripts/build_usage.py once")
@pytest.mark.parametrize("rung", sorted(PUBLISHED))
def test_a_replayed_row_matches_what_the_chapter_published(rung: str) -> None:
    """The replay reproduces the measurement it stands in for, or it is not evidence."""
    replayed = _replayed()
    assert rung in replayed, f"{rung} produced no billed incident on replay"
    ratio = replayed[rung] / PUBLISHED[rung]
    assert abs(ratio - 1.0) <= TOLERANCE, (
        f"{rung} replayed {replayed[rung]:.0f} against a published "
        f"{PUBLISHED[rung]} ({ratio:.2f}x). Either the sidecar is stale or the "
        f"rung's path changed; rebuild with scripts/build_usage.py and re-read "
        f"the chapter's cost section before widening this band."
    )


def test_the_structured_output_schema_is_priced() -> None:
    """The defect this whole file exists to prevent, pinned directly.

    `count_tokens` accepts no `output_format`, so the schema a `messages.parse`
    call carries is invisible to it - and on Level 1 that schema was over half
    the request. Sending it as a strict tool is what closes the gap, and a
    schema that is not closed is rejected by the API rather than silently
    dropped, so both halves are asserted here.
    """
    from escalation_ladder.classify import Classification

    tool = as_tool(Classification)
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False
    for definition in tool["input_schema"].get("$defs", {}).values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False


@pytest.mark.skipif(not SIDECAR.exists(), reason="run scripts/build_usage.py once")
def test_replaying_without_the_sidecar_reports_nothing_rather_than_zero() -> None:
    """A cost file that is absent must not look like a rung that is free.

    Same rule `scripts/measure_costs.py` already applies to an all-failed rung,
    asserted here at the seam: with no prices loaded a replay bills zero, and
    zero is the Zero Row's honest answer and nobody else's.
    """
    recordings = load_recordings()
    incident = load_incidents()[1]
    from escalation_ladder.classify import run

    bare = run(incident, RecordedCompleter(recordings=recordings))
    assert bare.total_input_tokens == 0
    assert bare.model_calls > 0  # the call happened; only its price is unknown
