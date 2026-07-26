"""Level 0 - deterministic incident routing.

The floor of the ladder: rules, thresholds, and lookup tables, with no model
anywhere in the path. This module is the zero row of the cost table, and it is
still imported by the composite router in Chapter 11 - the cheap path only
stays available if it is never deleted.

Nothing here is probabilistic, and nothing here calls the network. It reads the
deploy log, which is a data source; reading a data source does not lift code off
the floor. What would lift it off the floor is an input whose correct output no
test can specify, and `route` refuses those rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from escalation_ladder.fixtures.deploys import recent_deploys
from escalation_ladder.fixtures.incidents import Incident
from escalation_ladder.instrument import CostLedger, measured
from escalation_ladder.rungs import register_rung

SEVERITIES: tuple[str, ...] = ("SEV1", "SEV2", "SEV3")


@dataclass(frozen=True)
class AlertSignal:
    """A machine alert parsed into the four fields the rules act on."""

    service: str
    metric: str
    threshold: float
    firing_minutes: int


@dataclass(frozen=True)
class Service:
    """Ownership and criticality for one service."""

    rota: str
    tier: str  # the severity floor for anything paging on this service


SERVICES: dict[str, Service] = {
    "checkout-api": Service(rota="payments-oncall", tier="SEV1"),
    "payment-gateway": Service(rota="payments-oncall", tier="SEV2"),
    "notification-worker": Service(rota="platform-oncall", tier="SEV2"),
    "search-api": Service(rota="search-oncall", tier="SEV3"),
}

# The value at which a metric is a SEV1 on its own, whatever the service tier.
# Units are the units the alert grammar uses, which is what the alert author
# already had to agree on with the monitoring system.
METRIC_SEV1_LINE: dict[str, float] = {
    "http_5xx_rate": 0.25,
    "p99_latency_ms": 3000.0,
    "cpu_utilization": 0.95,
    "connection_pool_in_use": 0.95,
}

# Minutes before an unacknowledged page escalates to the next rota up.
ESCALATION_SLA: dict[str, int] = {"SEV1": 5, "SEV2": 15, "SEV3": 60}

# `alert: http_5xx_rate > 0.4 for 5m on checkout-api`
#
# Anchored at both ends on purpose. An unanchored pattern would happily find
# this grammar inside a paragraph of prose a human wrote, route on the fragment
# it matched, and discard the rest of the sentence - which is exactly the silent
# misroute this module exists to avoid.
_ALERT = re.compile(
    r"^alert:\s+(?P<metric>[a-z0-9_]+)\s*>\s*(?P<threshold>\d+(?:\.\d+)?)\s+"
    r"for\s+(?P<minutes>\d+)m\s+on\s+(?P<service>[a-z0-9-]+)$"
)


def parse_alert(report: str) -> AlertSignal | None:
    """Parse a machine alert line. Return None for anything else.

    None is the load-bearing return value: it is how free-text prose leaves
    this function, and it is what makes the refusal in `route` possible.
    """
    match = _ALERT.match(report.strip())
    if match is None:
        return None
    return AlertSignal(
        service=match["service"],
        metric=match["metric"],
        threshold=float(match["threshold"]),
        firing_minutes=int(match["minutes"]),
    )


def owner_for(service_name: str) -> Service | None:
    """The rota that owns a service. None for a service nobody has claimed."""
    return SERVICES.get(service_name)


def severity_for(signal: AlertSignal, service: Service) -> str:
    """The service tier sets the floor; a metric past its SEV1 line escalates.

    The asymmetry is deliberate. A metric can raise severity and can never
    lower it, because a tier-1 service being only mildly degraded is still a
    tier-1 problem. A rule that averaged the two would page the wrong rota
    calmly, which is worse than paging it loudly.
    """
    line = METRIC_SEV1_LINE.get(signal.metric)
    if line is not None and signal.threshold >= line:
        return "SEV1"
    return service.tier


def escalation_after(severity: str) -> int:
    """Minutes before an unacknowledged page escalates."""
    return ESCALATION_SLA[severity]


@dataclass(frozen=True)
class RoutingDecision:
    """Where an incident goes, and the rule that sent it there.

    `decided_by` carries forward the convention from Chapter 2's
    `Recommendation`: every decision this book makes says which rule made it,
    so it can be argued with instead of merely trusted.
    """

    incident_id: str
    rota: str | None
    severity: str | None
    escalate_after_minutes: int | None
    already_overdue: bool
    recent_deploy: str | None
    needs_human: bool
    decided_by: str

    @property
    def routed(self) -> bool:
        return not self.needs_human

    def summary(self) -> str:
        if self.needs_human:
            return (
                f"{self.incident_id}\n"
                f"  UNROUTABLE - needs a human\n"
                f"  because: {self.decided_by}"
            )
        overdue = " (already past SLA)" if self.already_overdue else ""
        deploy = self.recent_deploy or "none on record"
        return (
            f"{self.incident_id}\n"
            f"  page {self.rota} at {self.severity}\n"
            f"  escalate after {self.escalate_after_minutes}m{overdue}\n"
            f"  last deploy: {deploy}\n"
            f"  decided by: {self.decided_by}"
        )


def _unroutable(incident: Incident, reason: str) -> RoutingDecision:
    return RoutingDecision(
        incident_id=incident.incident_id,
        rota=None,
        severity=None,
        escalate_after_minutes=None,
        already_overdue=False,
        recent_deploy=None,
        needs_human=True,
        decided_by=reason,
    )


def route(incident: Incident) -> RoutingDecision:
    """Route one incident, or refuse.

    There is no default branch. Every path out of this function either names
    the rule that decided the page or names the reason no rule could.
    """
    signal = parse_alert(incident.report)
    if signal is None:
        return _unroutable(incident, "the report is prose, not an alert line")

    service = owner_for(signal.service)
    if service is None:
        return _unroutable(incident, f"no rota owns service {signal.service!r}")

    severity = severity_for(signal, service)
    sla = escalation_after(severity)
    deploys = recent_deploys(signal.service)
    deploy = f"{deploys[0].deploy_id} {deploys[0].deployed_at}" if deploys else None

    return RoutingDecision(
        incident_id=incident.incident_id,
        rota=service.rota,
        severity=severity,
        escalate_after_minutes=sla,
        already_overdue=signal.firing_minutes >= sla,
        recent_deploy=deploy,
        needs_human=False,
        decided_by=(
            f"{signal.metric} > {signal.threshold} on {signal.service}"
            f" (tier {service.tier})"
        ),
    )


@register_rung("Level 0: Deterministic Code")
def run(incident: Incident) -> CostLedger:
    """Rung entry point. Measured, so the zero row is measured rather than assumed."""
    ledger = CostLedger()

    @measured("rules.route", ledger)
    def _run_once() -> RoutingDecision:
        return route(incident)

    _run_once()
    return ledger


if __name__ == "__main__":
    from escalation_ladder.fixtures.incidents import load_incidents

    decisions = [route(incident) for incident in load_incidents()]
    for decision in decisions:
        print(decision.summary())
        print()
    routed = sum(1 for d in decisions if d.routed)
    print(f"routed {routed}/{len(decisions)}; {len(decisions) - routed} need a human")
