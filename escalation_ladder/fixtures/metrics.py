"""A fake metrics API returning deterministic time series.

Values are derived from a SHA-256 of (service, metric, absolute minute) -- never `random`, and
never the builtin `hash()`, which is salted per process and would return different numbers in
different runs. That distinction is what makes the book's measured cost numbers reproducible
on a reader's machine.

Two properties matter to Chapter 7, and neither was true before it.

The series is INCIDENT-AWARE. Every seeded incident has an episode window, and inside that
window the metrics it actually degraded leave their healthy band. Outside it, every series is
quiet. Before this, `query_metric` was uniform over the metric's full plausible range, which
meant a query during an incident looked exactly like a query during a calm Tuesday -- so a
tool built on it would return noise, a model would draw a confident conclusion from that
noise, and no test anywhere would fail. A fixture that cannot be wrong cannot demonstrate
anything.

The sample index is ABSOLUTE, not relative to the start of the window. Asking for 30 minutes
and asking for 60 minutes now agree on the 30 minutes they share. The earlier version indexed
the hash by position in the returned list, so the same wall-clock minute took a different
value depending on how much history you asked for -- which is fine until something compares
two windows, and Chapter 7 is where something does.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Fixed reference time. Reading the clock would make every example non-reproducible.
NOW = datetime(2026, 3, 21, 5, 0, 0, tzinfo=timezone.utc)

_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# The band a series sits in when the service is healthy. Narrow on purpose: a
# baseline as wide as the metric's plausible range is indistinguishable from an
# incident, and every alert threshold in `rules.METRIC_SEV1_LINE` would be
# crossed by ordinary noise.
#
# Units match the alert grammar in rules.py, NOT percent: an alert reading
# `http_5xx_rate > 0.4` and a sample from this series are directly comparable.
_BASELINE: dict[str, tuple[float, float]] = {
    "http_5xx_rate": (0.0005, 0.0060),        # fraction of requests
    "p99_latency_ms": (120.0, 260.0),
    "cpu_utilization": (0.18, 0.34),          # fraction of cores
    "connection_pool_in_use": (0.10, 0.30),   # fraction of pool
    "messages_sent": (900.0, 1100.0),         # count per minute
    "orders_created": (880.0, 1080.0),        # count per minute
}

# The band the same series sits in while an incident is degrading it.
_DEGRADED: dict[str, tuple[float, float]] = {
    "http_5xx_rate": (0.0700, 0.1900),
    "p99_latency_ms": (2600.0, 4400.0),
    "cpu_utilization": (0.82, 0.97),
    "connection_pool_in_use": (0.88, 0.99),
    "messages_sent": (3600.0, 4400.0),
    "orders_created": (880.0, 1080.0),
}

_DEFAULT_RANGE = (0.0, 100.0)


@dataclass(frozen=True)
class Episode:
    """A window in which one incident degraded a specific set of series.

    `metrics` is the point. An incident does not move every number - it moves
    the ones its root cause touches, and leaves the rest alone. INC-1045 is the
    clearest case: `messages_sent` quadruples and `orders_created` does not
    move at all, which is exactly what makes the ratio in the duplicates runbook
    diagnostic rather than decorative.
    """

    incident_id: str
    service: str
    metrics: tuple[str, ...]
    started_at: str
    ended_at: str | None  # None means still degraded at NOW


# Windows are wider than the incident record's `opened_at`, because a metric
# starts misbehaving before a human notices and files it.
EPISODES: tuple[Episode, ...] = (
    Episode("INC-1041", "checkout-api", ("http_5xx_rate", "connection_pool_in_use"),
            "2026-01-14T02:00:00Z", "2026-01-14T02:50:00Z"),
    Episode("INC-1042", "payment-gateway", ("p99_latency_ms",),
            "2026-01-19T10:40:00Z", "2026-01-19T12:30:00Z"),
    Episode("INC-1043", "search-api", ("p99_latency_ms",),
            "2026-02-02T09:20:00Z", "2026-02-02T11:05:00Z"),
    Episode("INC-1044", "checkout-api", ("http_5xx_rate",),
            "2026-02-11T21:50:00Z", "2026-02-11T23:40:00Z"),
    Episode("INC-1045", "notification-worker", ("messages_sent",),
            "2026-03-03T16:00:00Z", "2026-03-03T18:10:00Z"),
    # Still degrading at NOW. This is the one a tool can read live, which is why
    # it is Chapter 7's incident.
    Episode("INC-1046", "search-api", ("http_5xx_rate", "p99_latency_ms"),
            "2026-03-21T04:30:00Z", None),
)


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _unit_value(service: str, metric: str, minute: int) -> float:
    """A stable pseudo-random float in [0, 1) for one point in one series."""
    digest = hashlib.sha256(f"{service}|{metric}|{minute}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def degraded_at(service: str, metric: str, moment: datetime) -> str | None:
    """The incident degrading this series at this moment, or None if it is healthy."""
    for episode in EPISODES:
        if episode.service != service or metric not in episode.metrics:
            continue
        start = _parse(episode.started_at)
        end = _parse(episode.ended_at) if episode.ended_at is not None else None
        if moment >= start and (end is None or moment < end):
            return episode.incident_id
    return None


def known_metrics(service: str) -> tuple[str, ...]:
    """Every series this API can serve for a service.

    Every service reports the same catalogue except the two counters, which
    only the worker has. Chapter 7's tool returns this list in its error
    message rather than a bare failure, so a model that asked for the wrong
    series can correct itself without another round trip.
    """
    common = ("http_5xx_rate", "p99_latency_ms", "cpu_utilization",
              "connection_pool_in_use")
    if service == "notification-worker":
        return common + ("messages_sent", "orders_created")
    return common


def query_metric(
    service: str,
    metric: str,
    minutes: int,
    *,
    ending_at: str | None = None,
) -> list[tuple[str, float]]:
    """Return `minutes` one-minute samples ending at `ending_at`, oldest first.

    Each element is (ISO-8601 timestamp, value). `ending_at` defaults to NOW.
    It exists so a caller can read the window around an incident that is no
    longer live -- and it is deliberately NOT something the model chooses. See
    the tool schema in `tools.py`.
    """
    if minutes <= 0:
        return []
    end = _parse(ending_at) if ending_at is not None else NOW
    series: list[tuple[str, float]] = []
    for i in range(minutes):
        stamp = end - timedelta(minutes=minutes - 1 - i)
        table = (
            _DEGRADED
            if degraded_at(service, metric, stamp) is not None
            else _BASELINE
        )
        low, high = table.get(metric, _DEFAULT_RANGE)
        absolute = int((stamp - _EPOCH).total_seconds() // 60)
        value = low + _unit_value(service, metric, absolute) * (high - low)
        series.append((stamp.isoformat().replace("+00:00", "Z"), round(value, 4)))
    return series
