# Runbook: notification-worker delivery backlog

## Symptoms

Transactional messages arrive late, minutes to hours after the event that triggered them. No
errors are logged; throughput is simply below the arrival rate.

## First checks

1. Read queue depth and consumer lag. A depth that grows linearly means the consumer pool is
   undersized rather than broken.
2. Check whether a provider is rate limiting the send path. Providers return 429 and the
   client retries silently, so the symptom is latency rather than failure.
3. Verify no poison message is blocking the head of a partition.

## Known trap

Backlog and duplicate delivery look the same to a recipient, because a backlog draining after a
consumer restart delivers a burst. Read the send log timestamps before concluding which one
this is.

## Escalation

SEV2 during business hours, SEV3 overnight unless the backlog covers password resets.
