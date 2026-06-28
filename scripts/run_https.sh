#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -f certs/cert.pem || ! -f certs/key.pem ]]; then
  "${ROOT}/scripts/generate_dev_certs.sh"
fi

source .venv/bin/activate
exec uvicorn app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-keyfile certs/key.pem \
  --ssl-certfile certs/cert.pem
