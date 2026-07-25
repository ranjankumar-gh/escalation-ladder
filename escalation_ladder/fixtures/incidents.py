"""The seeded past-incident corpus.

The six incidents are chosen so that each rung of the ladder has a case it handles cleanly
and a case it cannot handle at all. They are the raw material for every Failure Receipt in
the book, so changing them changes the book's arguments -- edit with care.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DB = Path(__file__).parent / "incidents.db"


@dataclass(frozen=True)
class Incident:
    incident_id: str
    opened_at: str          # ISO 8601, fixed -- never generated from the clock
    service: str
    severity: str           # SEV1 | SEV2 | SEV3
    title: str
    report: str             # free-text on-call report, as a human typed it
    root_cause: str
    resolution: str


SEED_INCIDENTS: tuple[Incident, ...] = (
    # Machine-generated alert text. A rule can classify this; Level 0 handles it.
    Incident(
        incident_id="INC-1041",
        opened_at="2026-01-14T02:14:00Z",
        service="checkout-api",
        severity="SEV1",
        title="Checkout 5xx rate above 40%",
        report="alert: http_5xx_rate > 0.4 for 5m on checkout-api",
        root_cause="Connection pool exhausted after deploy 2026.1.12 halved max_connections.",
        resolution="Rolled back deploy 2026.1.12; restored max_connections to 200.",
    ),
    # Free-text prose with no alert. Defeats Level 0; needs language understanding.
    Incident(
        incident_id="INC-1042",
        opened_at="2026-01-19T11:02:00Z",
        service="payment-gateway",
        severity="SEV2",
        title="Intermittent payment timeouts, customer reports only",
        report=(
            "Three separate customers emailed support saying their card was charged but the "
            "order page spun forever. No alert fired. Support escalated to us because it "
            "sounded like the thing that happened before Christmas."
        ),
        root_cause="Upstream acquirer added 8s latency; our timeout was 10s so no alert tripped.",
        resolution="Lowered timeout to 5s, added a p99 latency alert on the acquirer call.",
    ),
    # Resolution lives in a runbook, not in the report. Needs retrieval (Level 2).
    Incident(
        incident_id="INC-1043",
        opened_at="2026-02-02T09:41:00Z",
        service="search-api",
        severity="SEV3",
        title="Search slow for a subset of tenants",
        report="Search feels slow for some enterprise tenants. Not sure which. No dashboard shows it.",
        root_cause="Missing composite index on tenant_id+created_at for tenants above 1M documents.",
        resolution="Added the composite index; p99 fell from 4.2s to 310ms.",
    ),
    # Open-ended: no knowable completion path up front. This is the Level 6 case.
    Incident(
        incident_id="INC-1044",
        opened_at="2026-02-11T22:07:00Z",
        service="checkout-api",
        severity="SEV1",
        title="Checkout down, cause unknown",
        report=(
            "Everything is on fire. Checkout is returning 500s, the dashboard is red across "
            "the board, and the last deploy was six hours ago so I do not think it is us. "
            "Payment gateway looks fine. I have no idea where to start."
        ),
        root_cause="Certificate for the internal service mesh expired; mTLS handshakes failed.",
        resolution="Renewed the certificate and added a 30-day expiry alert.",
    ),
    # Fixed multi-stage shape: classify, extract, retrieve, draft. The Level 3 case.
    Incident(
        incident_id="INC-1045",
        opened_at="2026-03-03T16:20:00Z",
        service="notification-worker",
        severity="SEV2",
        title="Duplicate order confirmation emails",
        report="Customers are getting the same confirmation email 3-7 times.",
        root_cause="Consumer crashed after send but before ack, so the message was redelivered.",
        resolution="Made the send idempotent on an order_id key with a 24h dedupe window.",
    ),
    # Needs a live metric value no document contains. The Level 4 case.
    Incident(
        incident_id="INC-1046",
        opened_at="2026-03-21T04:55:00Z",
        service="search-api",
        severity="SEV3",
        title="Elevated error rate during nightly reindex",
        report="alert: http_5xx_rate > 0.05 for 10m on search-api",
        root_cause="Reindex saturated disk IO; queries timed out while the index rebuilt.",
        resolution="Throttled the reindex and moved it behind a read replica.",
    ),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    opened_at   TEXT NOT NULL,
    service     TEXT NOT NULL,
    severity    TEXT NOT NULL,
    title       TEXT NOT NULL,
    report      TEXT NOT NULL,
    root_cause  TEXT NOT NULL,
    resolution  TEXT NOT NULL
)
"""


def build_db(path: Path | None = None) -> Path:
    """Create (or refresh) the incident database. Idempotent."""
    db_path = Path(path) if path is not None else DEFAULT_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)
        conn.executemany(
            "INSERT OR REPLACE INTO incidents "
            "(incident_id, opened_at, service, severity, title, report, root_cause, resolution) "
            "VALUES (:incident_id, :opened_at, :service, :severity, :title, :report, "
            ":root_cause, :resolution)",
            [asdict(i) for i in SEED_INCIDENTS],
        )
    return db_path


def load_incidents(path: Path | None = None) -> list[Incident]:
    """Load every incident, building the database first if it does not exist."""
    db_path = Path(path) if path is not None else DEFAULT_DB
    if not db_path.exists():
        build_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM incidents ORDER BY incident_id").fetchall()
    return [Incident(**dict(row)) for row in rows]
