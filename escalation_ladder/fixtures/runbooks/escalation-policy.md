# Escalation policy

| Condition | Action |
|---|---|
| Revenue-path service, success rate below 90% | SEV1, page immediately |
| Revenue-path service, degraded but serving | SEV2, notify on-call in hours |
| Non-revenue service degraded | SEV3, ticket |
| Classifier confidence below 0.7 | Route to a human. Do not guess. |

The last row is the floor: below the confidence threshold, no model decides.
