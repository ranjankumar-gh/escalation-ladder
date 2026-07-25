# Runbook: payment-gateway timeouts

## Symptoms

Customers report a charged card with no order confirmation. Often no alert fires, because
the acquirer's added latency stays under our timeout.

## First checks

1. `query_metric("payment-gateway", "p99_latency_ms", 60)` - look for a step change, not a spike.
2. Check the acquirer status page before assuming the fault is ours.

## Known trap

A timeout set above the acquirer's degraded latency makes this failure invisible to alerting,
so it arrives through support instead. Alert on p99 latency, not only on errors.
