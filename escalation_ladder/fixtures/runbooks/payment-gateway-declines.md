# Runbook: payment-gateway elevated declines

## Symptoms

Authorization decline rate above baseline with no change in traffic mix. Orders fail at the
payment step and customers are not charged.

## First checks

1. Segment declines by issuer and by decline code. A single issuer moving is the acquirer's
   problem; a uniform shift is usually ours.
2. Check whether a recent deploy changed the fields sent on authorization. A dropped or
   reformatted address field raises soft declines immediately.
3. Compare against the acquirer status page before opening a ticket with them.

## Known trap

Declines and timeouts are different failures with the same customer-facing outcome. A decline
means the acquirer answered. Confirm which one you have before following this runbook.

## Escalation

SEV1. Revenue path, and the loss is immediate and unrecoverable.
