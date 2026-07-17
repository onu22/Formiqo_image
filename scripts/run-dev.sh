#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DOCS_URL="http://127.0.0.1:8000/docs#/"
UI_URL="http://127.0.0.1:5173/"
HEALTH_URL="http://127.0.0.1:8000/docs"
UI_HEALTH_URL="http://127.0.0.1:5173/"
PORT=8000
FE_PORT=5173
VENV_DIR=".venv"
STAMP_FILE="${VENV_DIR}/.requirements-sha256"
PID_FILE="${VENV_DIR}/uvicorn.pid"
LOG_FILE="${VENV_DIR}/uvicorn.log"
FE_PID_FILE="${VENV_DIR}/vite.pid"
FE_LOG_FILE="${VENV_DIR}/vite.log"
FE_STAMP_FILE="${VENV_DIR}/.frontend-package-sha256"
FRONTEND_DIR="frontend"

stop_servers() {
  local stopped=0
  local pid

  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      echo "Stopped uvicorn (PID $pid)."
      stopped=1
    fi
    rm -f "$PID_FILE"
  fi

  if [[ -f "$FE_PID_FILE" ]]; then
    pid="$(cat "$FE_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      # Vite often spawns children; kill the process group when possible.
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      echo "Stopped Vite (PID $pid)."
      stopped=1
    fi
    rm -f "$FE_PID_FILE"
  fi

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null || true
    echo "Stopped process on port ${PORT} (PID $pid)."
    stopped=1
  done < <(lsof -i ":${PORT}" -sTCP:LISTEN -t 2>/dev/null || true)

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null || true
    echo "Stopped process on port ${FE_PORT} (PID $pid)."
    stopped=1
  done < <(lsof -i ":${FE_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)

  if pkill -f "uvicorn app.main:app" 2>/dev/null; then
    echo "Stopped remaining uvicorn processes."
    stopped=1
  fi

  if pkill -f "vite" 2>/dev/null; then
    echo "Stopped remaining Vite processes."
    stopped=1
  fi

  if [[ "$stopped" -eq 0 ]]; then
    echo "No backend or frontend dev server running."
  else
    echo "Dev servers stopped."
  fi
}

port_pid() {
  local port="$1"
  lsof -i ":${port}" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local attempt
  for attempt in $(seq 1 40); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for ${label} at $url" >&2
  return 1
}

open_firefox() {
  local url="$1"
  if ! open -a Firefox "$url" 2>/dev/null; then
    echo "Firefox is required but could not be opened. Install Firefox or check the app name." >&2
    exit 1
  fi
}

ensure_venv() {
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11 is required but was not found on PATH." >&2
    exit 1
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    python3.11 -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

ensure_dependencies() {
  local req_hash
  req_hash="$(shasum -a 256 requirements.txt | awk '{print $1}')"

  if [[ ! -f "$STAMP_FILE" ]] || [[ "$(cat "$STAMP_FILE")" != "$req_hash" ]] || ! python -c "import uvicorn" >/dev/null 2>&1; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    echo "$req_hash" > "$STAMP_FILE"
  else
    echo "Python dependencies up to date."
  fi
}

ensure_frontend_dependencies() {
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required but was not found on PATH." >&2
    exit 1
  fi

  local pkg_hash
  pkg_hash="$(shasum -a 256 "${FRONTEND_DIR}/package-lock.json" | awk '{print $1}')"

  if [[ ! -d "${FRONTEND_DIR}/node_modules" ]] || [[ ! -f "$FE_STAMP_FILE" ]] || [[ "$(cat "$FE_STAMP_FILE")" != "$pkg_hash" ]]; then
    echo "Installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
    echo "$pkg_hash" > "$FE_STAMP_FILE"
  else
    echo "Frontend dependencies up to date."
  fi
}

start_backend() {
  local existing_pid
  existing_pid="$(port_pid "$PORT")"

  if [[ -n "$existing_pid" ]]; then
    echo "Backend already running on port ${PORT} (PID ${existing_pid})."
    echo "$existing_pid" > "$PID_FILE"
    return 0
  fi

  echo "Starting uvicorn on port ${PORT}..."
  "${VENV_DIR}/bin/python" - <<'PY'
import subprocess
from pathlib import Path

root = Path.cwd()
venv = root / ".venv"
log_file = venv / "uvicorn.log"
pid_file = venv / "uvicorn.pid"

with log_file.open("a", encoding="utf-8") as log:
    process = subprocess.Popen(
        [
            str(venv / "bin" / "python"),
            "-m",
            "uvicorn",
            "app.main:app",
            "--reload",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

pid_file.write_text(str(process.pid), encoding="utf-8")
print(process.pid)
PY

  sleep 1
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Uvicorn failed to start. Check ${LOG_FILE}." >&2
    tail -20 "$LOG_FILE" >&2 || true
    exit 1
  fi
}

start_frontend() {
  local existing_pid
  existing_pid="$(port_pid "$FE_PORT")"

  if [[ -n "$existing_pid" ]]; then
    echo "Frontend already running on port ${FE_PORT} (PID ${existing_pid})."
    echo "$existing_pid" > "$FE_PID_FILE"
    return 0
  fi

  echo "Starting Vite on port ${FE_PORT}..."
  # Detach into a new session (like uvicorn) so Vite survives when this script exits.
  # A plain `&` in the script's shell is killed by SIGHUP on macOS when run-dev.sh returns.
  "${VENV_DIR}/bin/python" - <<PY
import subprocess
from pathlib import Path

root = Path.cwd()
frontend = root / "${FRONTEND_DIR}"
venv = root / "${VENV_DIR}"
log_file = venv / "vite.log"
pid_file = venv / "vite.pid"

with log_file.open("a", encoding="utf-8") as log:
    process = subprocess.Popen(
        [
            "npm",
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            "${FE_PORT}",
        ],
        cwd=frontend,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

pid_file.write_text(str(process.pid), encoding="utf-8")
print(process.pid)
PY

  sleep 1
  if ! kill -0 "$(cat "$FE_PID_FILE")" 2>/dev/null; then
    echo "Vite failed to start. Check ${FE_LOG_FILE}." >&2
    tail -20 "$FE_LOG_FILE" >&2 || true
    exit 1
  fi
}

run_dev() {
  ensure_venv
  ensure_dependencies
  ensure_frontend_dependencies
  start_backend
  start_frontend
  wait_for_url "$HEALTH_URL" "backend"
  wait_for_url "$UI_HEALTH_URL" "frontend"
  open_firefox "$UI_URL"
  open_firefox "$DOCS_URL"

  echo
  echo "Formiqo backend + frontend ready."
  echo "  UI:       $UI_URL"
  echo "  API docs: $DOCS_URL"
  echo "  Backend PID:  $(cat "$PID_FILE")"
  echo "  Frontend PID: $(cat "$FE_PID_FILE")"
  echo "  Backend logs:  $LOG_FILE"
  echo "  Frontend logs: $FE_LOG_FILE"
}

case "${1:-run}" in
  run)
    run_dev
    ;;
  stop)
    stop_servers
    ;;
  *)
    echo "Usage: $0 [run|stop]" >&2
    exit 1
    ;;
esac
