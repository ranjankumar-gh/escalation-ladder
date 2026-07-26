# Runbook: notification-worker duplicate sends

## Symptoms

Recipients receive several copies of the same transactional message. Copies arrive within a
few minutes of each other and carry identical bodies.

## First checks

1. Compare `messages_sent` against `orders_created` for the same window. A ratio above 1.1
   means redelivery rather than a producer bug.
2. Check the consumer for crashes between send and acknowledgment. The queue is at-least-once,
   so an unacknowledged message is redelivered by design.
3. Confirm the dedupe key is set on the send path and that its window is longer than the
   consumer's visibility timeout.

## Known trap

The send itself is not idempotent unless a dedupe key is supplied. Retrying a send without one
produces a second delivery that looks identical to a duplicate producer event, and the two are
indistinguishable in the send log.

## Escalation

SEV2. Customer-visible but not revenue blocking. Do not page overnight.
