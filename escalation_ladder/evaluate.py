"""Scoring surfaces: one eval per place a rung can be wrong.

Chapter 13. Every rung chapter's cost table carried an `Eval difficulty` cell
written in words -- "unit tests only", "golden set and a judge", "trajectory, not
answer". This module turns that column into an integer by scoring each surface
separately, and the integer is the count of surfaces, not a rating.

Three rules, and each one exists because dropping it produces a number that
looks fine and means nothing:

1. A result is a RATE with an interval, never a pass or a fail. Chapter 4
   observed the same prompt returning two different services on two runs; a
   single assertion over a closed enum is a coin flip reported as a test.
2. A surface with no ground truth, or no sample, REFUSES. It does not score
   zero and it does not drop out of the table. Chapter 3's rule, applied to
   measurement: an eval that cannot decide must say so, because a missing row
   and a failing row lead to opposite actions.
3. Cost sits on the same row as score. An eval that reports only correctness
   can justify climbing and can never justify descending, and Chapter 16 is a
   descent.

Everything here runs against the committed recordings: no key, no network, no
spend. What it CANNOT do is measure a judge, and it says so rather than
substituting something cheaper and calling it the same number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Sequence

from escalation_ladder.classify import Classification, decide, evidence_supported
from escalation_ladder.fixtures.incidents import Incident, load_incidents
from escalation_ladder.fixtures.labels import GOLDEN, Golden
from escalation_ladder.instrument import CostLedger
from escalation_ladder.llm import (
    Completer,
    Completion,
    MeteredCompleter,
    RecordedCompleter,
    ToolRun,
    ToolSpec,
    Usage,
    prompt_key,
)

RECORDINGS = Path(__file__).resolve().parent.parent / "tests" / "recordings"
CONFIDENCE = 0.95

# The prefix `RecordedCompleter` uses when a prompt has no recording. Pinned by
# a test rather than trusted: a replay that quietly counted a missing recording
# as a refusal would report Level 4 as declining incidents it was never asked
# about, which is Chapter 12's partial measurement wearing a different hat.
MISS = "no recording"


# ---------------------------------------------------------------------------
# A rate, not an assertion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rate:
    """A pass rate over n samples, with the interval that makes n visible.

    The interval is Wilson rather than normal-approximation, because at the
    sample sizes an eval set actually reaches the normal interval is wrong in
    the direction that flatters you: at 4 of 4 it reports plus or minus zero.
    Wilson at 4 of 4 reports a lower bound near 0.51, which is the honest
    statement that four successes are consistent with a coin.
    """

    passes: int
    n: int

    @property
    def value(self) -> float:
        return self.passes / self.n if self.n else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        if not self.n:
            return (0.0, 1.0)
        z = NormalDist().inv_cdf(1 - (1 - CONFIDENCE) / 2)
        p, n = self.value, self.n
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
        return (max(0.0, centre - spread), min(1.0, centre + spread))

    @property
    def width(self) -> float:
        low, high = self.interval
        return high - low

    def render(self) -> str:
        low, high = self.interval
        return f"{self.passes}/{self.n} ({low:.2f}-{high:.2f})"


@dataclass(frozen=True)
class Score:
    """One surface's rate, why it was scored that way, and what it cost.

    `scored_by` is this file's `decided_by`. When `rate` is None it carries the
    reason the surface refused, so an undecided row is as legible as a decided
    one -- the whole point of not collapsing them into a zero.
    """

    surface: str
    level: int
    rate: Rate | None
    scored_by: str
    input_tokens: int = 0
    model_calls: int = 0

    @property
    def decided(self) -> bool:
        return self.rate is not None

    def render(self) -> str:
        body = self.rate.render() if self.rate is not None else "not measured"
        return f"L{self.level} {self.surface}: {body} - {self.scored_by}"


# ---------------------------------------------------------------------------
# Replay: every sample the recordings hold, and an exact count of what they miss
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_recordings() -> dict[str, tuple[str, ...]]:
    """Every recording, keyed by prompt, keeping ALL samples of the same prompt.

    `scripts/measure_costs.py` merges the same files with `dict.update`, so the
    last file wins. For a cost table that is harmless -- two samples of one
    prompt cost the same. For an eval it is the difference between n=1 and n=2,
    and this corpus has four prompts recorded twice: Chapter 6's pipeline
    re-issues Chapter 4's classify prompt verbatim, on a different day, so the
    repository has been holding a second draw of every Level 1 classification
    since Chapter 6 shipped.
    """
    samples: dict[str, list[str]] = {}
    for path in sorted(RECORDINGS.glob("ch*.json")):
        for key, body in json.loads(path.read_text(encoding="utf-8")).items():
            samples.setdefault(key, []).append(body)
    return {key: tuple(bodies) for key, bodies in samples.items()}


@lru_cache(maxsize=1)
def load_usage() -> dict[str, Usage]:
    path = RECORDINGS / "usage.json"
    if not path.exists():
        return {}
    return {
        key: Usage(int(entry["input_tokens"]), int(entry["output_tokens"]))
        for key, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


@dataclass
class Replay:
    """A recorded completer that counts what the recordings did not cover.

    Without the count, a rung asked about an incident nobody ever recorded
    returns a refusal indistinguishable from a rung that read the evidence and
    declined. Those are opposite findings and this repository contains both.
    """

    inner: RecordedCompleter
    misses: int = 0

    def parse(
        self, *, system: str, user: str, schema: Any, effort: str = "low"
    ) -> Completion[Any]:
        result = self.inner.parse(
            system=system, user=user, schema=schema, effort=effort
        )
        if result.failed is not None and result.failed.startswith(MISS):
            self.misses += 1
        return result

    def invoke(
        self,
        *,
        system: str,
        user: str,
        tools: tuple[ToolSpec, ...],
        execute: Any,
        schema: Any,
        effort: str = "low",
    ) -> ToolRun[Any]:
        result = self.inner.invoke(
            system=system,
            user=user,
            tools=tools,
            execute=execute,
            schema=schema,
            effort=effort,
        )
        if result.failed is not None and result.failed.startswith(MISS):
            self.misses += 1
        return result


LATEST = -1


def replay(pick: int = LATEST) -> Replay:
    """A completer answering from the `pick`-th recorded sample of each prompt.

    The default is the MOST RECENT draw, and that default is a decision rather
    than an obvious choice. Neither sample of a twice-recorded prompt is more
    correct than the other; they are two draws from the same distribution. What
    makes "latest" defensible is only that it is stated, so a score can be
    reproduced. A harness that picked whichever the filesystem happened to hand
    it first would produce two different tables from one command.

    Prompts with fewer samples than `pick` fall back to their last one, so a
    sweep over picks never invents a draw. `draws()` reports how many real
    samples a prompt has.
    """
    samples = load_recordings()
    chosen = {
        key: bodies[-1] if pick < 0 else bodies[min(pick, len(bodies) - 1)]
        for key, bodies in samples.items()
    }
    return Replay(RecordedCompleter(recordings=chosen, usage_by_key=load_usage()))


def draws(system: str, user: str) -> tuple[str, ...]:
    """Every recorded sample of one prompt. Length is the n an eval may claim."""
    return load_recordings().get(prompt_key(system=system, user=user), ())


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

# A check returns True (correct), False (wrong), or None (nothing to score).
# None is not a failure and not a zero; it is the refusal, and `score` keeps it
# out of both the numerator and the denominator.
Check = Callable[[Any, Golden], bool | None]


@dataclass(frozen=True)
class Surface:
    """A place a rung can be wrong independently of its final answer."""

    name: str
    level: int
    check: Check
    scored_by: str


@dataclass(frozen=True)
class Observation:
    """One rung's output for one incident, and what producing it cost."""

    incident_id: str
    result: Any
    ledger: CostLedger = field(default_factory=CostLedger)
    missing: int = 0


def score(surface: Surface, observations: Iterable[Observation]) -> Score:
    """Apply one surface's check to every observation that has something to say.

    An observation whose recordings were incomplete is dropped from n rather
    than counted wrong. That is the same rule `measure_costs.py` applies to a
    partially measured rung, for the same reason: a run that never happened is
    not evidence that it failed.
    """
    passes = 0
    n = 0
    skipped = 0
    tokens = 0
    calls = 0
    for observed in observations:
        golden = GOLDEN.get(observed.incident_id)
        if golden is None or observed.missing:
            skipped += 1
            continue
        verdict = surface.check(observed.result, golden)
        if verdict is None:
            skipped += 1
            continue
        n += 1
        passes += 1 if verdict else 0
        tokens += observed.ledger.total_input_tokens
        calls += observed.ledger.model_calls
    if not n:
        return Score(
            surface.name,
            surface.level,
            None,
            f"no sample ({skipped} unmeasured)",
        )
    note = surface.scored_by
    if skipped:
        note = f"{note}; {skipped} unmeasured"
    return Score(surface.name, surface.level, Rate(passes, n), note, tokens, calls)


# --- the checks -------------------------------------------------------------


def correct_service(result: Any, golden: Golden) -> bool | None:
    """A closed enum: right, wrong, or never answered."""
    claimed = getattr(result, "service", None)
    if claimed is None or claimed == "unknown":
        return None
    return claimed == golden.service


def correct_severity(result: Any, golden: Golden) -> bool | None:
    claimed = getattr(result, "severity", None)
    if claimed is None:
        return None
    return claimed == golden.severity


def routed_correctly(result: Any, golden: Golden) -> bool | None:
    """The end-to-end answer: did the right team get paged at the right grade?

    A refusal returns None rather than False, because Chapter 7's INC-1043 is a
    correct refusal and scoring it as a failed answer is precisely the harness
    defect this chapter exists to fix. Whether the refusal was RIGHT is a
    different surface, scored by `gave_up_correctly`.
    """
    decision = getattr(result, "decision", result)
    if not getattr(decision, "routed", False):
        return None
    return decision.severity == golden.severity


def probed_the_evidence(result: Any, golden: Golden) -> bool | None:
    """Did the run make the call whose output settles the incident?

    Scored on ARGUMENTS. INC-1042's investigation searched the correct log
    source with the pattern `error`, which matches nothing in a source whose
    decisive line is a WARN, and a name-level check calls that correct.
    """
    calls = getattr(result, "calls", None)
    if not calls:
        return None
    if not golden.probes:
        # Nothing in the tools settles this incident, so there is no correct
        # call to have made. Not a pass and not a fail.
        return None
    return all(
        any(probe.matches(call.name, call.arguments) for call in calls)
        for probe in golden.probes
    )


def gave_up_correctly(result: Any, golden: Golden) -> bool | None:
    """Was declaring the incident unreachable the right call?

    The surface Chapter 9 left unscored. `unreachable` is often the cheapest
    correct answer and nothing in this book has ever checked whether it was
    true, which means the one decision that saves the most money is the one
    with no eval behind it.

    Applicability is decided by whether the rung CAN give up, not by whether it
    did. Reading the value instead - `gave_up is None` means nothing to score -
    silently drops every run that correctly kept going, leaving a rate computed
    only over the runs that quit. That reads as 1/1 and is not a measurement.
    """
    if not hasattr(result, "gave_up") and not hasattr(result, "exhausted"):
        return None
    quit_early = bool(getattr(result, "gave_up", None)) or bool(
        getattr(result, "exhausted", False)
    )
    return quit_early is not golden.reachable


def trajectory_sound(result: Any, golden: Golden) -> bool | None:
    """Every step that ran, ran clean -- scored before the answer is read.

    A chain can reach a correct conclusion through a step that failed and was
    papered over downstream, and an answer-only harness cannot see the
    difference. Chapter 6's Downstream Veto is why that happens often enough
    to matter.
    """
    steps: Sequence[Any] | None = getattr(result, "steps", None)
    if steps is None:
        steps = getattr(result, "attempts", None)
    if not steps:
        return None
    return all(getattr(step, "ok", False) for step in steps)


def cited_its_evidence(result: Any, golden: Golden) -> bool | None:
    """The generation half of Level 2: is the quote actually in the corpus?

    Deterministic, and that is the argument. Groundedness is the one quality
    question on this rung that does not need a judge, so a team reaching for a
    judge here is paying a model to answer a question `in` answers.
    """
    quote = getattr(result, "quote", None)
    if quote is None:
        return None
    return getattr(result, "grounded", False)


# Level 1's two surfaces are scored on the model's CLAIM rather than on the
# routed decision, and the split matters: `service` and `severity` are what the
# model asserted, while `routing` is what the system did after Chapter 3's floor
# had its say. Collapsing them hides the only measurement that can tell you
# whether the floor is still earning its place.
CLAIM_SURFACES: tuple[Surface, ...] = (
    Surface("service", 1, correct_service, "closed enum against the label"),
    Surface("severity", 1, correct_severity, "closed enum against the label"),
)

SURFACES: tuple[Surface, ...] = (
    Surface("groundedness", 2, cited_its_evidence, "quote found in the cited document"),
    Surface("routing", 3, routed_correctly, "severity of the page actually raised"),
    Surface("tool choice", 4, probed_the_evidence, "decisive call made, by arguments"),
    Surface("trajectory", 5, trajectory_sound, "every step that ran, ran clean"),
    Surface("giving up", 6, gave_up_correctly, "quit exactly when unreachable"),
)


# ---------------------------------------------------------------------------
# Level 1 is the only rung this corpus can sample twice
# ---------------------------------------------------------------------------


def classification_draws(incident: Incident) -> tuple[Classification, ...]:
    """Every recorded classification of one incident, as objects.

    Not a re-run. These are the samples the repository already holds, read
    directly rather than replayed one at a time, which is what makes n honest.
    """
    from escalation_ladder.classify import SYSTEM, build_user

    bodies = draws(SYSTEM, build_user(incident))
    parsed: list[Classification] = []
    for body in bodies:
        try:
            parsed.append(Classification.model_validate_json(body))
        except Exception:
            continue
    return tuple(parsed)


def claims(incidents: Iterable[Incident]) -> list[Observation]:
    """One observation per recorded DRAW, not one per incident.

    This is where n stops being a formality. Four of this corpus's incidents
    were classified twice under the identical prompt - Chapter 6's pipeline
    re-issues Chapter 4's classify call verbatim - so the honest sample size for
    a Level 1 surface is eight, and one of those eight pairs disagrees.
    """
    observations: list[Observation] = []
    for incident in incidents:
        for claim in classification_draws(incident):
            observations.append(Observation(incident.incident_id, claim))
    return observations


def floor_effect(incidents: Iterable[Incident]) -> tuple[Rate, Rate]:
    """Severity before and after Chapter 3's one-directional floor.

    Chapter 4 argued the floor was worth its cost from one sample per incident.
    This is the same claim as a rate over every sample the corpus holds, which
    is the only form in which it can be argued with.
    """
    raw_passes = raw_n = floored_passes = floored_n = 0
    for incident in incidents:
        golden = GOLDEN.get(incident.incident_id)
        if golden is None:
            continue
        for claim in classification_draws(incident):
            raw_n += 1
            raw_passes += 1 if claim.severity == golden.severity else 0
            decision = decide(incident, claim)
            if decision.routed:
                floored_n += 1
                floored_passes += 1 if decision.severity == golden.severity else 0
    return Rate(raw_passes, raw_n), Rate(floored_passes, floored_n)


# ---------------------------------------------------------------------------
# The composite's surface: answered correctly, or answered too low
# ---------------------------------------------------------------------------


def re_ask(incidents: Iterable[Incident], ask: str, pick: int = LATEST) -> tuple[Rate, dict[int, int]]:
    """Share of a cascade's answers that were RIGHT at the rung it stopped on.

    Chapter 11's open surface. A Refusal Cascade climbs on a refusal, so a rung
    that answers confidently and wrongly ends the climb - the Format Illusion
    one level up, where the thing being trusted is not a schema but a rung's
    willingness to produce output at all. The complement of this rate is the
    re-ask rate, and it is the only number that distinguishes a request answered
    CORRECTLY at Level 1 from one answered PLAUSIBLY at Level 1 when it needed
    Level 4.
    """
    from escalation_ladder.router import handle

    passes = n = 0
    reached: dict[int, int] = {}
    for incident in incidents:
        golden = GOLDEN.get(incident.incident_id)
        if golden is None:
            continue
        completer = replay(pick)
        outcome = handle(incident, ask, completer)  # type: ignore[arg-type]
        if completer.misses or outcome.level_reached is None:
            continue
        reached[outcome.level_reached] = reached.get(outcome.level_reached, 0) + 1
        n += 1
        passes += 1 if outcome.decision.severity == golden.severity else 0
    return Rate(passes, n), reached


# ---------------------------------------------------------------------------
# How big does an eval set have to be?
# ---------------------------------------------------------------------------


def samples_needed(
    baseline: float, lift: float, *, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Cases per arm to detect `lift` over `baseline` in a two-proportion test.

    The arithmetic Chapter 10 needed and this book cannot afford. It is the
    standard normal-approximation sizing, and it is deliberately the crude
    version: the useful output is an order of magnitude, and every refinement
    moves the answer by less than the uncertainty in the effect size somebody
    guessed at the start.
    """
    if not 0.0 < baseline < 1.0 or lift <= 0.0 or baseline + lift >= 1.0:
        raise ValueError("baseline and baseline+lift must lie strictly inside (0, 1)")
    treated = baseline + lift
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    pooled = (baseline + treated) / 2
    numerator = (
        z_alpha * (2 * pooled * (1 - pooled)) ** 0.5
        + z_power * (baseline * (1 - baseline) + treated * (1 - treated)) ** 0.5
    ) ** 2
    return int(-(-numerator / (lift * lift) // 1))


def detectable_lift(n: int, baseline: float, **kwargs: Any) -> float:
    """The smallest lift `n` cases per arm can detect. The question in reverse.

    Asked this way the answer is usually disqualifying, which is the point: a
    team with 30 labeled cases discovers it is powered to detect a 30-point
    swing, and no architecture decision worth having turns on a swing that
    large.
    """
    low, high = 1e-4, 1.0 - baseline - 1e-4
    for _ in range(60):
        middle = (low + high) / 2
        if samples_needed(baseline, middle, **kwargs) > n:
            low = middle
        else:
            high = middle
    return high


# ---------------------------------------------------------------------------
# The judge, and the one honest thing to say about it
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You score one incident summary against the incident report it \
was written from.

Answer two questions independently. FAITHFUL: does every claim in the summary \
appear in the report? A summary that adds a cause the report does not state is \
not faithful, however plausible the cause is. COMPLETE: does the summary carry \
the detail an on-call engineer needs to act - the symptom, the scope, and \
whether anything is already known to be ruled out?

Quote the span that decides each answer. If you cannot quote one, the answer is \
no."""


def judge_user(report: str, summary: str) -> str:
    return f"<report>\n{report}\n</report>\n\n<summary>\n{summary}\n</summary>"


def judged(
    incident: Incident, summary: str, completer: Completer, schema: Any
) -> Completion[Any]:
    """Score a free-text summary with a model, because nothing else can.

    The last resort on the eval ladder and the only surface in this chapter
    that costs money to measure. It is shipped and it is UNMEASURED in this
    book: a judge prompt is new text, so it has no recording, and inventing a
    number for it would be the exact failure the chapter is about.

    Reach for it only after the checkable surfaces are green. A judge that
    disagrees with `evidence_supported` is not a second opinion, it is a more
    expensive way to be wrong about something a substring test already settled.
    """
    return completer.parse(
        system=JUDGE_SYSTEM,
        user=judge_user(incident.report, summary),
        schema=schema,
        effort="low",
    )


def checkable_first(incident: Incident, claim: Classification) -> bool:
    """The Floor Test, one layer up: can a check settle this without a model?

    Chapter 4 required a verbatim quote precisely so this question has an
    answer here. Where it does, the judge is not cheaper, faster, or more
    trustworthy -- it is none of the three.
    """
    return evidence_supported(claim, incident)


def observe(
    rung: Callable[..., Any], incidents: Iterable[Incident], pick: int = LATEST
) -> list[Observation]:
    """Run one rung over the corpus against the recordings, keeping the misses.

    Rungs that already carry a ledger keep theirs; Levels 1 and 2 return a
    decision with no ledger on it, so the seam is metered from outside. Both
    paths are the same measurement - `MeteredCompleter` is the wrapper Chapter 7
    introduced for exactly this, and using it here means no rung module needed
    an edit to become scoreable.
    """
    observations: list[Observation] = []
    for incident in incidents:
        completer = replay(pick)
        outer = CostLedger()
        metered = MeteredCompleter(inner=completer, ledger=outer, stage="eval")
        result = rung(incident, metered)
        own = getattr(result, "ledger", None)
        observations.append(
            Observation(
                incident.incident_id,
                result,
                own if own is not None else outer,
                completer.misses,
            )
        )
    return observations


def corpus() -> list[Incident]:
    return [i for i in load_incidents() if i.incident_id in GOLDEN]
