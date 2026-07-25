# Runbook: search-api latency

## Symptoms

Slow search, usually reported per tenant rather than globally.

## First checks

1. Segment `p99_latency_ms` by tenant size. Tenants above 1M documents degrade first.
2. Check whether the nightly reindex is running - it saturates disk IO.
3. Verify the composite index on `tenant_id, created_at` exists.

## Known trap

A global p99 stays healthy while a handful of large tenants are unusable. Segment before
concluding there is no problem.
