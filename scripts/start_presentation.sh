#!/usr/bin/env bash
# Canonical one-command demo: FastAPI + Streamlit.
# Usage: ./scripts/start_presentation.sh
# Env:   .env (optional), PORT / STREAMLIT_PORT, HOST=127.0.0.1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing venv at $PY — run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
fi

export JWT_SECRET="${JWT_SECRET:-dev-local-change-in-dotenv}"
export AUTH_DISABLED="${AUTH_DISABLED:-1}"
export MEDASSIST_BOOTSTRAP_DOCTOR_USER="${MEDASSIST_BOOTSTRAP_DOCTOR_USER:-doctor}"
export MEDASSIST_BOOTSTRAP_DOCTOR_PASSWORD="${MEDASSIST_BOOTSTRAP_DOCTOR_PASSWORD:-doctor123}"
export MEDASSIST_BOOTSTRAP_ADMIN_USER="${MEDASSIST_BOOTSTRAP_ADMIN_USER:-admin}"
export MEDASSIST_BOOTSTRAP_ADMIN_PASSWORD="${MEDASSIST_BOOTSTRAP_ADMIN_PASSWORD:-admin123}"

HOST="${HOST:-127.0.0.1}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

port_in_use() {
  local p="$1"
  command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1
}

if [[ -n "${PORT:-}" ]]; then
  CHOSEN="$PORT"
  if port_in_use "$CHOSEN"; then
    echo "ERROR: PORT=$CHOSEN is already in use."
    exit 1
  fi
else
  CHOSEN=""
  for p in 8000 8001 8002 8003 8080; do
    if ! port_in_use "$p"; then
      CHOSEN=$p
      break
    fi
  done
  if [[ -z "${CHOSEN}" ]]; then
    echo "ERROR: No free port in list 8000 8001 8002 8003 8080. Set PORT explicitly."
    exit 1
  fi
fi

if port_in_use "$STREAMLIT_PORT"; then
  echo "ERROR: Streamlit port $STREAMLIT_PORT is already in use. Set STREAMLIT_PORT to a free port."
  exit 1
fi

export MEDASSIST_API_BASE="http://${HOST}:${CHOSEN}"
export STREAMLIT_PUBLIC_URL="http://${HOST}:${STREAMLIT_PORT}"
export PYTHONPATH="$ROOT"

UVICORN_PID=""
ST_PID=""

cleanup() {
  echo ""
  echo "Stopping background services..."
  for pid in ${ST_PID:-} ${UVICORN_PID:-}; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_for_http() {
  local url="$1"
  local deadline=$((SECONDS + 45))
  while [[ $SECONDS -lt $deadline ]]; do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.4
  done
  return 1
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " MedAssist.AI — API + Streamlit"
echo " API (OpenAPI): ${MEDASSIST_API_BASE}/docs"
echo " Streamlit:     ${STREAMLIT_PUBLIC_URL}"
echo " AUTH_DISABLED: ${AUTH_DISABLED} (set 0 in .env to require JWT)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

"$PY" -m uvicorn api.main:app --host "$HOST" --port "$CHOSEN" &
UVICORN_PID=$!

if ! wait_for_http "${MEDASSIST_API_BASE}/health"; then
  echo "ERROR: API did not become healthy at ${MEDASSIST_API_BASE}/health"
  exit 1
fi

"$PY" -m streamlit run "$ROOT/streamlit_app.py" \
  --server.port "$STREAMLIT_PORT" \
  --server.address "$HOST" \
  --browser.gatherUsageStats false &
ST_PID=$!
sleep 1

echo ""
echo "All services running. Press Ctrl+C to stop."
wait
