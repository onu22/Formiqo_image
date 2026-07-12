---
name: run
description: Start the Formiqo FastAPI dev server (venv, deps, uvicorn) and open Swagger docs in Firefox.
disable-model-invocation: true
---

# Run Formiqo Dev Server

Start the local FastAPI development environment and open the Swagger UI in Firefox.

## Steps

1. From the repository root, run:

```bash
chmod +x scripts/run-dev.sh
./scripts/run-dev.sh
```

2. If the script fails, report the exact error. Common causes:
   - `python3.11` not installed
   - `pip install` failure
   - port 8000 conflict
   - Firefox not installed

3. Confirm success with:
   - Server URL: `http://127.0.0.1:8000/docs#/`
   - PID from the script output
   - Firefox opened to the docs page

4. Do **not** start a second uvicorn process if the script reports the server is already running on port 8000.

## Stop server

Use `/stop` in Cursor chat, or run `./scripts/run-dev.sh stop`.
