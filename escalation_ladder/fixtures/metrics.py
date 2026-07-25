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
_RANGES: dict[str, tuple[float, float]] = {
    "http_5xx_rate": (0.0, 12.0),            # percent
    "p99_latency_ms": (80.0, 4500.0),
    "cpu_utilization": (5.0, 95.0),
    "connection_pool_in_use": (0.0, 100.0),  # percent of pool
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
        series.append((stamp.isoformat().replace("+00:00", "Z"), round(value, 3)))
    return series
