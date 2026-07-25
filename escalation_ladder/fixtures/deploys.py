"""A synthetic deploy log. Fixed literals -- the deploy history never changes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Deploy:
    deploy_id: str
    service: str
    deployed_at: str        # ISO 8601, fixed
    version: str
    author: str
    summary: str


DEPLOYS: tuple[Deploy, ...] = (
    Deploy("D-9001", "checkout-api", "2026-01-12T15:30:00Z", "2026.1.12", "a.rao",
           "Tune connection pool sizing for the new pod shape."),
    Deploy("D-9002", "checkout-api", "2026-02-11T16:05:00Z", "2026.2.11", "s.iyer",
           "Add idempotency key to the order submit path."),
    Deploy("D-9003", "payment-gateway", "2026-01-18T10:12:00Z", "2026.1.18", "m.khan",
           "Bump acquirer SDK to 4.2.0."),
    Deploy("D-9004", "search-api", "2026-02-01T08:44:00Z", "2026.2.1", "j.park",
           "Enable nightly reindex job."),
    Deploy("D-9005", "search-api", "2026-03-20T19:15:00Z", "2026.3.20", "j.park",
           "Increase reindex batch size from 500 to 5000."),
    Deploy("D-9006", "notification-worker", "2026-03-02T12:00:00Z", "2026.3.2", "a.rao",
           "Switch to manual ack after send."),
)


def recent_deploys(service: str) -> list[Deploy]:
    """Deploys for one service, most recent first. Empty list for an unknown service."""
    matches = [d for d in DEPLOYS if d.service == service]
    return sorted(matches, key=lambda d: d.deployed_at, reverse=True)
