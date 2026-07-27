"""A seeded application log, searchable by substring.

Generated from templates rather than written out line by line, because the
interesting property is density over a window and a literal file long enough to
have density is a file nobody reads. Everything here is deterministic: the same
query returns the same lines on any machine, forever.

The design constraint that matters is what each source does NOT say.
`checkout-api` logs the symptom it can see -- its sidecar refused a connection --
and nothing about why, because a service genuinely cannot see the reason its own
sidecar failed a handshake. The reason is in `service-mesh`, which is in the
catalogue and reachable at any time. Nothing hides it. It is simply not
findable by anyone who has not already read the checkout logs, which is what
makes INC-1044 need a second round and INC-1046 not need one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from escalation_ladder.fixtures.metrics import NOW, _parse

# Log sources. A superset of `rules.SERVICES` on purpose: infrastructure emits
# logs and owns no rota, and a tool whose enum was the routing table would make
# the mesh unreachable rather than merely unobvious.
LOG_SOURCES: tuple[str, ...] = (
    "checkout-api",
    "payment-gateway",
    "notification-worker",
    "search-api",
    "service-mesh",
)


@dataclass(frozen=True)
class LogLine:
    source: str
    at: str  # ISO 8601
    level: str  # INFO | WARN | ERROR
    message: str

    def render(self) -> str:
        return f"{self.at} {self.level:<5} {self.source} {self.message}"


@dataclass(frozen=True)
class LogEpisode:
    """Lines emitted at a fixed cadence while something is wrong."""

    source: str
    started_at: str
    ended_at: str | None
    every_minutes: int
    messages: tuple[tuple[str, str], ...]  # (level, text)


# One routine line per source, emitted every five minutes forever. Without these
# a search that matches nothing is indistinguishable from a source that is not
# logging at all, and those are different production problems.
ROUTINE: dict[str, str] = {
    "checkout-api": "POST /orders 200 in 41ms",
    "payment-gateway": "authorize ok acquirer=primary in 220ms",
    "notification-worker": "delivered order_confirmation batch=32",
    "search-api": "GET /search 200 in 38ms",
    "service-mesh": "sidecar healthy peers=12",
}

EPISODES: tuple[LogEpisode, ...] = (
    # INC-1046, still running at NOW. Everything needed to explain it is here
    # and in the deploy log, which is what makes it a one-round incident.
    LogEpisode(
        "search-api", "2026-03-21T04:30:00Z", None, 2,
        (
            ("INFO", "reindex started batch_size=5000 shard=0"),
            ("ERROR", "query timeout after 3000ms during reindex"),
            ("ERROR", "GET /search 503 upstream_timeout"),
            ("WARN", "io wait high, reindex holding write lock"),
        ),
    ),
    # INC-1044. The symptom, and only the symptom. checkout-api cannot see why
    # its sidecar refused the connection, so neither can anything reading these.
    LogEpisode(
        "checkout-api", "2026-02-11T21:50:00Z", "2026-02-11T23:40:00Z", 2,
        (
            ("ERROR", "POST /orders 503 upstream connect error before headers"),
            ("ERROR", "no healthy upstream for cluster=inventory"),
        ),
    ),
    # The cause of INC-1044, in the source nobody queries first.
    LogEpisode(
        "service-mesh", "2026-02-11T21:48:00Z", "2026-02-11T23:40:00Z", 3,
        (
            ("ERROR",
             "tls handshake failed peer=inventory: x509 certificate has expired"),
            ("WARN", "certificate NotAfter=2026-02-11T21:47:00Z serial=4a91c2"),
        ),
    ),
    LogEpisode(
        "notification-worker", "2026-03-03T16:00:00Z", "2026-03-03T18:10:00Z", 2,
        (
            ("WARN", "redelivering message order_id=A-8821 attempt=3"),
            ("WARN", "ack timeout, message returned to queue"),
        ),
    ),
    LogEpisode(
        "payment-gateway", "2026-01-19T10:40:00Z", "2026-01-19T12:30:00Z", 2,
        (("WARN", "acquirer call slow: 8400ms, under the 10000ms timeout"),),
    ),
)


def _lines_between(source: str, start: datetime, end: datetime) -> list[LogLine]:
    """Every line one source emitted in [start, end], oldest first."""
    lines: list[LogLine] = []

    routine = ROUTINE.get(source)
    if routine is not None:
        minute = start.replace(second=0, microsecond=0)
        while minute <= end:
            if int(minute.timestamp()) // 60 % 5 == 0:
                lines.append(
                    LogLine(source, _stamp(minute), "INFO", routine)
                )
            minute += timedelta(minutes=1)

    for episode in EPISODES:
        if episode.source != source:
            continue
        ep_start = _parse(episode.started_at)
        ep_end = _parse(episode.ended_at) if episode.ended_at else NOW
        minute = max(ep_start, start.replace(second=0, microsecond=0))
        step = 0
        while minute <= min(ep_end, end):
            elapsed = int((minute - ep_start).total_seconds() // 60)
            if elapsed >= 0 and elapsed % episode.every_minutes == 0:
                level, message = episode.messages[step % len(episode.messages)]
                lines.append(LogLine(source, _stamp(minute), level, message))
                step += 1
            minute += timedelta(minutes=1)

    return sorted(lines, key=lambda line: (line.at, line.message))


def _stamp(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def search_logs(
    source: str,
    pattern: str,
    minutes: int,
    *,
    ending_at: str | None = None,
    limit: int = 12,
) -> list[LogLine]:
    """Case-insensitive substring search over one source's recent lines.

    Substring rather than regex, and that is a blast-radius decision rather than
    a laziness one: a regex field accepts a catastrophically backtracking
    pattern from a model that has no idea it wrote one, and the tool would hang
    inside a request whose deadline is a page.

    The match runs over the RENDERED line - timestamp, level, source, message -
    rather than over the message alone, because that is what `grep` does to a
    log file and therefore what anyone searching one expects. Matching the
    message only was a real bug in this fixture: a search for "error" returned
    nothing while ERROR-level lines sat in the window, and a tool that answers
    "no matches" when there are matches is worse than a tool that is missing.
    """
    if minutes <= 0 or source not in LOG_SOURCES:
        return []
    end = _parse(ending_at) if ending_at is not None else NOW
    start = end - timedelta(minutes=minutes)
    needle = pattern.lower().strip()
    matches = [
        line
        for line in _lines_between(source, start, end)
        if needle in line.render().lower()
    ]
    return matches[-limit:]
