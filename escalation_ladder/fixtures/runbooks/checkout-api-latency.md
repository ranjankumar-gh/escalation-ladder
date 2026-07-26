# Runbook: checkout-api elevated latency

## Symptoms

`p99_latency_ms` above 1500 on `checkout-api` while the error rate stays flat. Customers
describe the order page as hanging or spinning rather than failing.

## First checks

1. Split latency by downstream call. Checkout fans out to payment authorization, inventory,
   and the notification publish, and only the first of those is on the critical path.
2. Compare against `connection_pool_in_use`. Latency rises before errors do when the pool is
   saturating.
3. Check `recent_deploys("checkout-api")` for a change to timeout or pool configuration.

## Known trap

A downstream service degrading slowly does not trip an error-rate alert, because our own
timeout is generous enough to absorb it. The failure surfaces as customer reports before it
surfaces as monitoring.

## Escalation

SEV2 while orders still complete. SEV1 once the checkout success rate falls below 90%.
