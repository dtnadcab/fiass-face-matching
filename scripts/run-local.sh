#!/usr/bin/env bash
# Local fiass-face-matching (no Docker): Postgres on localhost, in-memory Redis via REDIS_FAKE=1.
# Usage:
#   ./scripts/run-local.sh
#   PORT=18089 FACE_MIN_BLUR_VARIANCE=35 ./scripts/run-local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -q -r requirements.txt
export REDIS_FAKE="${REDIS_FAKE:-1}"
export FACE_MIN_BLUR_VARIANCE="${FACE_MIN_BLUR_VARIANCE:-38}"
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5432}"
export DB_NAME="${DB_NAME:-fiass_e2e}"
export DB_USER="${DB_USER:-$(whoami)}"
export DB_PASS="${DB_PASS:-}"
export PORT="${PORT:-18089}"
createdb "$DB_NAME" 2>/dev/null || true
exec uvicorn main:app --host 127.0.0.1 --port "$PORT"
