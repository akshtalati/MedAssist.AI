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
export STREAMLIT_PUBLIC_URL="http://${HOST}:8501"

cleanup() {
  if [[ -n "${UVICORN_PID:-}" ]]; then
    kill "$UVICORN_PID" 2>/dev/null || true
    wait "$UVICORN_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " API + OpenAPI: ${MEDASSIST_API_BASE}  |  GET ${MEDASSIST_API_BASE}/ → redirects to Streamlit"
echo " Starting Uvicorn in the background, then Streamlit (single doctor UI)."
echo " Open: ${STREAMLIT_PUBLIC_URL}  (full console: KG, assessments, follow-up)"
echo " Copy:  export MEDASSIST_API_BASE=${MEDASSIST_API_BASE}"
echo " Smoke: MEDASSIST_API_BASE=${MEDASSIST_API_BASE} ./scripts/smoke_presentation.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

"$PY" -m uvicorn api.main:app --host "$HOST" --port "$CHOSEN" &
UVICORN_PID=$!
# Brief wait so Streamlit’s first API calls succeed
sleep 2

# Doctor console: Streamlit (must match features in streamlit_app.py)
"$PY" -m streamlit run "$ROOT/streamlit_app.py" \
  --server.port 8501 \
  --server.address "$HOST" \
  --browser.gatherUsageStats false
