---
name: run
description: Start Formiqo backend (uvicorn) and frontend (Vite) together, then open the UI and Swagger in Firefox.
disable-model-invocation: true
---

# Run Formiqo Dev Servers

Start the local FastAPI backend and Vite frontend together.

## Steps

1. From the repository root, run:

```bash
chmod +x scripts/run-dev.sh
./scripts/run-dev.sh
```

2. If the script fails, report the exact error. Common causes:
   - `python3.11` not installed
   - `npm` not installed
   - `pip install` or `npm install` failure
   - port 8000 or 5173 conflict
   - Firefox not installed

3. Confirm success with:
   - UI: `http://127.0.0.1:5173/`
   - API docs: `http://127.0.0.1:8000/docs#/`
   - Backend and frontend PIDs from the script output
   - Firefox opened to both URLs

4. Do **not** start duplicate processes if the script reports a server is already running on port 8000 or 5173.

## Stop servers

Use `/stop` in Cursor chat, or run `./scripts/run-dev.sh stop`.
