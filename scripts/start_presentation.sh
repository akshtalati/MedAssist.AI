#!/usr/bin/env bash
# Start FastAPI + static UI on a free port; export MEDASSIST_API_BASE for CLI scripts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing venv at $PY — create it and pip install -r requirements.txt"
  exit 1
fi
HOST="${HOST:-127.0.0.1}"
if [[ -n "${PORT:-}" ]]; then
  CHOSEN="$PORT"
  if lsof -nP -iTCP:"$CHOSEN" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: PORT=$CHOSEN is already in use. Run: lsof -nP -iTCP:$CHOSEN -sTCP:LISTEN"
    exit 1
  fi
else
  CHOSEN=""
  for p in 8000 8001 8002 8003 8080; do
    if ! lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
      CHOSEN=$p
      break
    fi
  done
  if [[ -z "${CHOSEN}" ]]; then
    echo "ERROR: No free port in list 8000 8001 8002 8003 8080. Set PORT explicitly."
    exit 1
  fi
fi
export MEDASSIST_API_BASE="http://${HOST}:${CHOSEN}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Starting API + UI (first start may take 10–30s while imports load)."
echo " Wait for: Uvicorn running on http://${HOST}:${CHOSEN}"
echo " UI:    ${MEDASSIST_API_BASE}/"
echo " Copy:  export MEDASSIST_API_BASE=${MEDASSIST_API_BASE}"
echo " Smoke: MEDASSIST_API_BASE=${MEDASSIST_API_BASE} ./scripts/smoke_presentation.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exec "$PY" -m uvicorn api.main:app --host "$HOST" --port "$CHOSEN"
