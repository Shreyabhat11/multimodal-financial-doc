#!/bin/sh
# ============================================================
# Runs Alembic migrations before starting the main process (CMD, passed as "$@").
#
# Why here and not a separate one-off "migrate" compose service that runs once:
# for a single-instance deployment (this docker-compose.yml), running the migration
# as the container's own entrypoint is simpler and self-contained — no ordering
# dependency between two separate services to get wrong. A real multi-replica
# production deployment SHOULD run migrations as a separate release step (e.g. a
# Kubernetes Job, or a CI/CD pipeline step) before rolling out new API replicas,
# rather than having every replica race to run "alembic upgrade head" on startup —
# noted here rather than pretended away.
# ============================================================
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting: $@"
exec "$@"
