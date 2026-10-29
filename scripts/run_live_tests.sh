#!/usr/bin/env bash
# Live tier runner: the integration tests of tests/test_integration_live.py
# against real model hosts (audit 29.10.2026, release 3.4.1).
#
# Why this exists: the live tier used to default to http://localhost:11434
# (REQUIREMENTS §4 — the deploy host), so on a workstation all 13 tests skipped
# on every run and the tier was dead weight: it caught nothing and was never
# run on purpose. This script is the runner; the hosts come from the
# environment or from an OPTIONAL local env file next to this script
# (run_live_tests.env, gitignored) — host addresses never enter the repository.
# Defaults below are the honest localhost ones; see docker-compose.test.yml of
# the test contour for the wiring it mirrors.
#
# Usage:
#   scripts/run_live_tests.sh                     # the whole live tier
#   scripts/run_live_tests.sh -k summary          # pass pytest args through
#   LIVE_EMBEDDING_URL=... scripts/run_live_tests.sh   # override a host
#
# Requirements: the dev venv (.venv) with the project installed; reachability
# of the model hosts (unreachable hosts keep their honest SKIP, not a failure).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Optional local overrides (gitignored): model hosts, model names, dimension.
ENV_FILE="$(dirname "$0")/run_live_tests.env"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

# Embedding host + model served there (keep EMBEDDING_DIM and the tier's
# LIVE_EMBEDDING_DIM assertion in sync — the dimension is model-specific).
export LIVE_EMBEDDING_URL="${LIVE_EMBEDDING_URL:-http://localhost:11434}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding:4b}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-2560}"
export LIVE_EMBEDDING_DIM="${LIVE_EMBEDDING_DIM:-2560}"
# Generation host + model for the summary/judge live fixtures.
export LIVE_SUMMARY_URL="${LIVE_SUMMARY_URL:-http://localhost:11434}"
export LIVE_SUMMARY_MODEL="${LIVE_SUMMARY_MODEL:-flint:latest}"

echo "live tier: embedding=${LIVE_EMBEDDING_URL} summary=${LIVE_SUMMARY_URL} model=${LIVE_SUMMARY_MODEL}"

cd "$ROOT"
exec .venv/bin/python -m pytest tests/test_integration_live.py "$@"
