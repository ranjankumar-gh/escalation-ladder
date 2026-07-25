# Runbook: checkout-api elevated 5xx rate

## Symptoms

`http_5xx_rate` above 5% for more than 5 minutes on `checkout-api`.

## First checks

1. Compare against `connection_pool_in_use`. Sustained values near 100 mean pool exhaustion.
2. Check `recent_deploys("checkout-api")`. Pool sizing regressions have shipped twice.
3. Confirm the service mesh certificate is valid. An expired certificate presents as blanket
   500s with healthy upstreams, which is the most misleading version of this failure.

## Escalation

SEV1 if the checkout success rate is below 90%. Page the payments on-call immediately.
