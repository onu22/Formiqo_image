#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DOCS_URL="http://127.0.0.1:8000/docs#/"
HEALTH_URL="http://127.0.0.1:8000/docs"
PORT=8000
VENV_DIR=".venv"
STAMP_FILE="${VENV_DIR}/.requirements-sha256"
PID_FILE="${VENV_DIR}/uvicorn.pid"
LOG_FILE="${VENV_DIR}/uvicorn.log"

stop_server() {
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

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null || true
    echo "Stopped process on port ${PORT} (PID $pid)."
    stopped=1
  done < <(lsof -i ":${PORT}" -sTCP:LISTEN -t 2>/dev/null || true)

  if pkill -f "uvicorn app.main:app" 2>/dev/null; then
    echo "Stopped remaining uvicorn processes."
    stopped=1
  fi

  if [[ "$stopped" -eq 0 ]]; then
    echo "No dev server running on port ${PORT}."
  else
    echo "Dev server stopped."
  fi
}

port_pid() {
  lsof -i ":${PORT}" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true
}

wait_for_server() {
  local attempt
  for attempt in $(seq 1 30); do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for server at $HEALTH_URL" >&2
  return 1
}

open_firefox() {
  if ! open -a Firefox "$DOCS_URL" 2>/dev/null; then
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
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "$req_hash" > "$STAMP_FILE"
  else
    echo "Dependencies up to date."
  fi
}

start_server() {
  local existing_pid
  existing_pid="$(port_pid)"

  if [[ -n "$existing_pid" ]]; then
    echo "Server already running on port ${PORT} (PID ${existing_pid})."
    echo "$existing_pid" > "$PID_FILE"
    return 0
  fi

  echo "Starting uvicorn on port ${PORT}..."
  "${VENV_DIR}/bin/python" - <<'PY'
import subprocess
import sys
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

run_dev() {
  ensure_venv
  ensure_dependencies
  start_server
  wait_for_server
  open_firefox

  echo
  echo "Formiqo dev server ready."
  echo "  URL:  $DOCS_URL"
  echo "  PID:  $(cat "$PID_FILE")"
  echo "  Logs: $LOG_FILE"
}

case "${1:-run}" in
  run)
    run_dev
    ;;
  stop)
    stop_server
    ;;
  *)
    echo "Usage: $0 [run|stop]" >&2
    exit 1
    ;;
esac
