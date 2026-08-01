"""The golden set: what a correct answer looks like, per incident, per surface.

Chapter 5 labeled one surface -- the answer spans retrieval had to put in front
of the model -- in `retrieval_labels.py`. This file is that idea carried to every
other surface a rung can be wrong on, and it deliberately IMPORTS Chapter 5's
labels rather than restating them. Service and severity are ground truth for the
whole book; two files claiming them independently is two files that can disagree.

The field worth arguing about is `probes`. A tool-choice label written as tool
NAMES ("a correct investigation calls search_logs") is satisfied by every
investigation in this corpus, including the ones that found nothing, because the
model always calls all three tools. A label written as a probe -- a tool, the
arguments, and the string its output must contain -- is a claim about the
fixtures that can be executed, so `tests/test_labels.py` runs every probe and
fails if the fixture corpus ever stops containing the evidence the label says is
there. A golden set that cannot rot silently is worth more than a larger one
that can.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from escalation_ladder.fixtures.retrieval_labels import LABELS


@dataclass(frozen=True)
class Probe:
    """A tool call whose output settles the incident, and the proof it does.

    `contains` is checked against the tool's real output, so the label is a
    testable statement about the fixtures rather than an assertion about them.
    """

    tool: str
    arguments: dict[str, Any]
    contains: str

    def matches(self, name: str, arguments: dict[str, Any]) -> bool:
        """Did an actual call reach this probe's evidence?

        Argument-level rather than name-level, and that is the whole point.
        INC-1042's investigation searched the right log source with the pattern
        `error`, which matches nothing in a source whose decisive line is a
        WARN. A name-level check scores that call correct.
        """
        if name != self.tool:
            return False
        return all(arguments.get(key) == value for key, value in self.arguments.items())


@dataclass(frozen=True)
class Golden:
    """Ground truth for one incident across every surface a rung can be wrong on."""

    incident_id: str
    service: str
    severity: str
    answers: tuple[str, ...]
    probes: tuple[Probe, ...]
    reachable: bool
    reachable_because: str


def _minutes(**kwargs: Any) -> dict[str, Any]:
    """Probe arguments omit `minutes`: the window is bound by the Toolbox.

    Chapter 7 put the window on the incident rather than in the tool schema, so
    a probe that pinned a minute count would be labeling the model's choice of
    lookback instead of its choice of evidence.
    """
    return kwargs


GOLDEN: dict[str, Golden] = {
    "INC-1041": Golden(
        incident_id="INC-1041",
        service=LABELS["INC-1041"].service,
        severity=LABELS["INC-1041"].severity,
        answers=LABELS["INC-1041"].answers,
        probes=(
            Probe(
                "query_metric",
                _minutes(service="checkout-api", metric="connection_pool_in_use"),
                contains="connection_pool_in_use on checkout-api",
            ),
            Probe(
                "recent_deploys",
                _minutes(service="checkout-api"),
                contains="Tune connection pool sizing",
            ),
        ),
        reachable=True,
        reachable_because="pool saturation is in the metric and the cause is in the deploy log",
    ),
    "INC-1042": Golden(
        incident_id="INC-1042",
        service=LABELS["INC-1042"].service,
        severity=LABELS["INC-1042"].severity,
        answers=LABELS["INC-1042"].answers,
        probes=(
            Probe(
                "search_logs",
                _minutes(source="payment-gateway", pattern="acquirer"),
                contains="under the 10000ms timeout",
            ),
        ),
        reachable=True,
        reachable_because="the acquirer latency line is in the payment-gateway log",
    ),
    "INC-1043": Golden(
        incident_id="INC-1043",
        service=LABELS["INC-1043"].service,
        severity=LABELS["INC-1043"].severity,
        answers=LABELS["INC-1043"].answers,
        probes=(),
        reachable=False,
        reachable_because=(
            "a missing composite index appears in no metric, no log, and no deploy; "
            "the answer is in a runbook, which is Level 2's surface and not Level 4's"
        ),
    ),
    "INC-1044": Golden(
        incident_id="INC-1044",
        service=LABELS["INC-1044"].service,
        severity=LABELS["INC-1044"].severity,
        answers=LABELS["INC-1044"].answers,
        probes=(
            Probe(
                "search_logs",
                _minutes(source="service-mesh", pattern="error"),
                contains="x509 certificate has expired",
            ),
        ),
        reachable=True,
        reachable_because="the expiry is in a log source the report never names",
    ),
    "INC-1045": Golden(
        incident_id="INC-1045",
        service=LABELS["INC-1045"].service,
        severity=LABELS["INC-1045"].severity,
        answers=LABELS["INC-1045"].answers,
        probes=(
            Probe(
                "search_logs",
                _minutes(source="notification-worker", pattern="ack"),
                contains="ack timeout, message returned to queue",
            ),
            Probe(
                "recent_deploys",
                _minutes(service="notification-worker"),
                contains="Switch to manual ack after send",
            ),
        ),
        reachable=True,
        reachable_because="the redelivery is in the log and the change that caused it is in the deploy log",
    ),
    "INC-1046": Golden(
        incident_id="INC-1046",
        service=LABELS["INC-1046"].service,
        severity=LABELS["INC-1046"].severity,
        answers=LABELS["INC-1046"].answers,
        probes=(
            Probe(
                "search_logs",
                _minutes(source="search-api", pattern="reindex"),
                contains="io wait high",
            ),
            Probe(
                "recent_deploys",
                _minutes(service="search-api"),
                contains="Increase reindex batch size",
            ),
        ),
        reachable=True,
        reachable_because="the reindex saturation is in the log and the batch-size change is in the deploy log",
    ),
}
