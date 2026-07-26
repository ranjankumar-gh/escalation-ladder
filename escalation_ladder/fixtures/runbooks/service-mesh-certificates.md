# Runbook: service mesh certificate expiry

## Symptoms

Blanket 500s across one or more services while every upstream reports healthy. Dashboards go
red simultaneously rather than progressively, and no deploy correlates with the start.

## First checks

1. Read the mesh sidecar logs for handshake failures. Expired mTLS certificates fail the
   handshake and the application layer reports a generic upstream error.
2. Check certificate expiry across the mesh, not only for the service that paged.
3. Confirm the control plane is issuing certificates. A failed rotation job is silent until
   the existing certificate expires.

## Known trap

This presents as an application failure and every application-level check passes. Teams spend
the first hour reading their own code, because nothing in the error points at transport.

## Escalation

SEV1 across any revenue-path service.
