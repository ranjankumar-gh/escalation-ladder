# Runbook: search-api nightly reindex

## Symptoms

Elevated error rate or timeouts on `search-api` between 02:00 and 05:00 UTC, resolving without
intervention by the time anyone looks.

## First checks

1. Confirm the reindex job is running and check its progress. It saturates disk IO on the
   index hosts for the duration.
2. Read `http_5xx_rate` for the same window on previous nights. A recurring shape means the
   job, not an incident.
3. Check whether queries are being served from the replica or from the host being rebuilt.

## Known trap

The window closes before the page is acknowledged, so the incident is repeatedly opened and
auto-resolved. Recurring self-resolving alerts are a capacity problem being reported as an
availability problem.

## Escalation

SEV3. Route to the ticket queue and fix the schedule, not the symptom.
