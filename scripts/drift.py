"""Does a later step overrule a wrong earlier one, and what makes it able to?

Chapter 6 measured the Downstream Veto across a fixed pipeline by breaking stage
one on purpose. This is the same experiment one rung up, where it matters more:
a pipeline's stages each see the request, but a chain's later steps are choosing
their tool arguments from what the earlier steps concluded. A wrong step one does
not merely pass an error along - it decides what step two goes and reads.

The injection is deliberately NOT an invented value. `reasoning` already discards
a finding whose cited value is in no tool result, so an invented claim never
enters the carried state at all and there would be nothing to measure. The claim
injected here quotes a REAL log line and draws the wrong conclusion from it,
which is the failure Chapter 5 identified and could not fix with a citation
check: the check catches invention and does not catch misreading.

Two arms, one variable. Both are shown the original report at every step, because
that is structural. What differs is whether the prompt says the carried claims
are fallible.

Usage (needs ANTHROPIC_API_KEY):

    python scripts/drift.py                  # 3 samples per incident
    python scripts/drift.py --samples 10
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from escalation_ladder import reasoning as rung
from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.llm import AnthropicCompleter
from escalation_ladder.reasoning import ChainState, StepRecord
from escalation_ladder.rules import owner_for
from escalation_ladder.tools import Finding

# The shipped prompt with the fallibility paragraph removed, and nothing else
# changed. Derived by cutting the real one so the two arms cannot drift apart;
# tests/test_reasoning.py asserts the cut still finds its text, so a prompt edit
# breaks the test rather than silently turning this into a comparison of the
# shipped prompt with itself.
VETO_PARAGRAPH = """Findings from earlier steps are CLAIMS, not facts. They were produced by this
same process against less evidence than you now have, and they are sometimes
wrong. You are shown the original report every step so that you can weigh a
claim against it. If a claim disagrees with what you read, say what you read."""

WITHHELD = rung.SYSTEM.replace(
    VETO_PARAGRAPH,
    "Findings from earlier steps are recorded below. Build on them.",
)


# One wrong claim per incident, each quoting a line that genuinely exists in
# that service's logs and drawing the wrong conclusion from it. `truth` is the
# service a correct chain names, so "corrected" means the run reached it despite
# being handed the claim.
#
# Scored on the SERVICE rather than the rota, and INC-1044 is why. Its correct
# answer is checkout-api and the injected claim says payment-gateway - two
# different services that this book's routing table sends to the same
# payments-oncall rota. Scored on the rota, inheriting that mistake and
# correcting it look identical, and the experiment would report a veto that
# never happened. It is also a production observation worth keeping: a
# misattribution inside a shared rota reaches the right pager for the wrong
# reason, so it is invisible to everyone except whoever reads the finding.
INJECTIONS: dict[str, tuple[Finding, str]] = {
    "INC-1044": (
        Finding(
            service="payment-gateway",
            severity="SEV1",
            cause=(
                "Payment gateway is the failing component; checkout is only "
                "showing the downstream effect."
            ),
            evidence_tool="search_logs",
            evidence_value="authorize ok acquirer=primary",
            next_step="Read payment-gateway metrics for the acquirer call.",
            needs_human=True,
        ),
        "checkout-api",
    ),
    "INC-1045": (
        Finding(
            service="search-api",
            severity="SEV2",
            cause="Search is the source of the duplicate delivery.",
            evidence_tool="search_logs",
            evidence_value="GET /search 200",
            next_step="Read search-api error rate.",
            needs_human=True,
        ),
        "notification-worker",
    ),
    "INC-1046": (
        Finding(
            service="checkout-api",
            severity="SEV3",
            cause="Checkout is driving the elevated error rate on search.",
            evidence_tool="search_logs",
            evidence_value="POST /orders 200",
            next_step="Read checkout-api throughput.",
            needs_human=True,
        ),
        "search-api",
    ),
}


def _rota(service: str) -> str | None:
    owner = owner_for(service)
    return owner.rota if owner is not None else None


@dataclass
class Arm:
    label: str
    runs: int = 0
    corrected: int = 0
    inherited: int = 0
    refused: int = 0
    failed: int = 0
    # Runs that inherited the wrong service and still paged the right rota,
    # because two services share one. Counted so the headline number is not
    # quietly flattered by the routing table.
    same_rota_as_truth: int = 0
    routed_to: list[str] = field(default_factory=list)

    def report(self) -> dict:
        if not self.runs:
            return {
                "arm": self.label,
                "corrected": "not measured",
                "api_failures": self.failed,
            }
        return {
            "arm": self.label,
            "runs": self.runs,
            "corrected": f"{self.corrected}/{self.runs}",
            "inherited the wrong claim": f"{self.inherited}/{self.runs}",
            "refused": f"{self.refused}/{self.runs}",
            "wrong service, right rota": self.same_rota_as_truth,
            "api_failures": self.failed,
        }


def run_arm(label: str, system: str, samples: int) -> Arm:
    incidents = {i.incident_id: i for i in load_incidents()}
    completer = AnthropicCompleter()
    arm = Arm(label=label)

    original = rung.SYSTEM
    rung.SYSTEM = system
    try:
        for incident_id, (claim, truth) in INJECTIONS.items():
            case = incidents[incident_id]
            for _ in range(samples):
                # The whole reason ChainState is a value: a walk can be started
                # from a state carrying a claim the model never made.
                planted = ChainState(
                    incident=case,
                    records=(
                        StepRecord(
                            name="observe",
                            ok=True,
                            detail="injected",
                            calls=(),
                            finding=claim,
                        ),
                    ),
                )
                result = rung.reason(
                    case, completer, steps=rung.STEPS[1:], initial=planted
                )
                why = result.decision.decided_by
                if not result.routed and why.startswith("api error"):
                    # Not a refusal and not a measurement. Counting a vendor
                    # failure as "refused to inherit the claim" would make a
                    # dead account look like a working veto.
                    arm.failed += 1
                    continue

                arm.runs += 1
                if not result.routed:
                    arm.refused += 1
                elif result.service == truth:
                    arm.corrected += 1
                else:
                    arm.inherited += 1
                arm.routed_to.append(result.service or "refused")
                if result.service == claim.service and result.routed:
                    arm.same_rota_as_truth += int(
                        result.decision.rota == _rota(truth)
                    )
    finally:
        rung.SYSTEM = original
    return arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    stated = run_arm("claims marked fallible", rung.SYSTEM, args.samples)
    print(json.dumps(stated.report(), indent=2))
    withheld = run_arm("claims presented as fact", WITHHELD, args.samples)
    print(json.dumps(withheld.report(), indent=2))

    print()
    if not stated.runs or not withheld.runs:
        print("NOTHING MEASURED in at least one arm. No comparison is available.")
        return
    print(
        f"corrected: {stated.corrected}/{stated.runs} stated vs "
        f"{withheld.corrected}/{withheld.runs} withheld"
    )


if __name__ == "__main__":
    main()
