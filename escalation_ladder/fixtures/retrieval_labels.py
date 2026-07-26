"""What the corpus says that would have helped, per incident.

This is the golden set for the retrieval half of Chapter 5, and it is labeled by
hand because there is no other way to get the number.

The labels are ANSWER SPANS - exact substrings of the corpus - rather than
passage ids. That choice is the whole reason the chapter can compare chunking
strategies at all. Labeling relevant passage ids bakes one chunking into the
ground truth, so every rival chunking scores badly by construction and the
measurement answers a question nobody asked. A span is chunking-independent: any
retriever, cut any way, either put that text in front of the model or did not.

`answers` is what the on-call engineer needed. `service` is the routing answer,
kept here so retrieval quality and routing quality can be scored from one file.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    """Ground truth for one incident."""

    service: str
    severity: str
    answers: tuple[str, ...]


LABELS: dict[str, Label] = {
    "INC-1041": Label(
        service="checkout-api",
        severity="SEV1",
        answers=(
            "Sustained values near 100 mean pool exhaustion",
            "Compare current `max_connections` against the value before the "
            "last deploy",
        ),
    ),
    "INC-1042": Label(
        service="payment-gateway",
        severity="SEV2",
        answers=(
            "A timeout set above the acquirer's degraded latency makes this "
            "failure invisible to alerting",
            "look for a step change, not a spike",
        ),
    ),
    "INC-1043": Label(
        service="search-api",
        severity="SEV3",
        answers=(
            "Verify the composite index on `tenant_id, created_at` exists",
            "Segment `p99_latency_ms` by tenant size",
        ),
    ),
    "INC-1044": Label(
        service="checkout-api",
        severity="SEV1",
        answers=(
            "Read the mesh sidecar logs for handshake failures",
            "Check certificate expiry across the mesh",
        ),
    ),
    "INC-1045": Label(
        service="notification-worker",
        severity="SEV2",
        answers=(
            "Check the consumer for crashes between send and acknowledgment",
            "Confirm the dedupe key is set on the send path",
        ),
    ),
    "INC-1046": Label(
        service="search-api",
        severity="SEV3",
        answers=(
            "Confirm the reindex job is running and check its progress",
            "It saturates disk IO on the index hosts",
        ),
    ),
}

# Queries this corpus cannot answer. They are the calibration set for the
# relevance floor: the floor has to sit above the best score any of these
# achieves, or the retriever will hand the model three passages for a question
# about the coffee machine and the model will do its best with them.
#
# Real ones are better than invented ones. Every line below is the kind of thing
# that lands in an on-call channel and is not an incident.
UNANSWERABLE: tuple[str, ...] = (
    "The office wifi is down and nobody can reach the VPN concentrator.",
    "Can someone approve my expense report from the Berlin offsite?",
    "The coffee machine on floor three is leaking again.",
    "What is our policy on carrying over unused vacation days?",
    "Reminder: laptop refresh forms are due before the end of the quarter.",
    "Does anyone know the wifi password for the guest network?",
)
