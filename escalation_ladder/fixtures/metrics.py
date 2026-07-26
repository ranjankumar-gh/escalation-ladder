"""A fake metrics API returning deterministic time series.

Values are derived from a SHA-256 of (service, metric, minute index) -- never `random`, and
never the builtin `hash()`, which is salted per process and would return different numbers in
different runs. That distinction is what makes the book's measured cost numbers reproducible
on a reader's machine.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

# Fixed reference time. Reading the clock would make every example non-reproducible.
NOW = datetime(2026, 3, 21, 5, 0, 0, tzinfo=timezone.utc)

# (floor, ceiling) per metric -- keeps generated values in a plausible range.
#
# Units match the alert grammar in rules.py, NOT percent: an alert reading
# `http_5xx_rate > 0.4` and a sample from this series are directly comparable.
# Chapter 3 has no reason to care, because Level 0 never queries this API - but
# Chapter 7 builds tools over it, and a tool that compares 0.4 against a series
# scaled 0-12 would report every incident as catastrophic.
#
# The series is deliberately NOT correlated with the seeded incidents yet: it is
# uniform over the range, so a query during an incident window looks like any
# other. Chapter 7 makes it incident-aware, which is where that behaviour is
# load-bearing.
_RANGES: dict[str, tuple[float, float]] = {
    "http_5xx_rate": (0.0, 0.6),             # fraction of requests
    "p99_latency_ms": (80.0, 4500.0),
    "cpu_utilization": (0.05, 0.99),         # fraction of cores
    "connection_pool_in_use": (0.0, 1.0),    # fraction of pool
}
_DEFAULT_RANGE = (0.0, 100.0)


def _unit_value(service: str, metric: str, index: int) -> float:
    """A stable pseudo-random float in [0, 1) for one point in one series."""
    digest = hashlib.sha256(f"{service}|{metric}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def query_metric(service: str, metric: str, minutes: int) -> list[tuple[str, float]]:
    """Return `minutes` one-minute samples ending at NOW, oldest first.

    Each element is (ISO-8601 timestamp, value).
    """
    if minutes <= 0:
        return []
    low, high = _RANGES.get(metric, _DEFAULT_RANGE)
    series: list[tuple[str, float]] = []
    for i in range(minutes):
        stamp = NOW - timedelta(minutes=minutes - 1 - i)
        value = low + _unit_value(service, metric, i) * (high - low)
        series.append((stamp.isoformat().replace("+00:00", "Z"), round(value, 4)))
    return series
