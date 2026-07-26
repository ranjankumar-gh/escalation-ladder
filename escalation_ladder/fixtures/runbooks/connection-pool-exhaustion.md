# Runbook: database connection pool exhaustion

## Symptoms

`connection_pool_in_use` sustained near 1.0, with rising latency followed by errors. Affects
whichever service holds the pool, most often `checkout-api`.

## First checks

1. Compare current `max_connections` against the value before the last deploy. Pool sizing
   regressions have shipped twice on this system.
2. Look for a slow query holding connections. One unindexed query is enough to exhaust a pool
   sized for normal work.
3. Check whether a retry storm is multiplying demand. Retries acquire connections too.

## Known trap

Raising the pool size resolves the symptom and moves the bottleneck to the database. Confirm
the database can serve the larger pool before raising it.

## Escalation

Match the tier of the affected service.
