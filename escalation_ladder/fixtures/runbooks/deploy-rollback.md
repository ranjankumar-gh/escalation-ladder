# Runbook: rolling back a deploy

## Symptoms

Any regression that correlates with a deploy timestamp. Correlation is enough to roll back;
causation can be established afterwards.

## First checks

1. Read `recent_deploys(service)` and take the most recent entry within the incident window.
2. Confirm the deploy is reversible. Schema migrations and data backfills are not, and rolling
   back the application without reverting them makes the outage worse.
3. Roll back first and diagnose second. The rollback is the mitigation; the diagnosis is the
   follow-up.

## Known trap

A deploy six hours old is still a suspect if the regression needs traffic to surface. Deploy
time is not incident time.

## Escalation

No page of its own. Follow the tier of the incident that triggered it.
