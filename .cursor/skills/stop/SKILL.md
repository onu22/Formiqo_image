---
name: stop
description: Stop the Formiqo FastAPI dev server running on port 8000.
disable-model-invocation: true
---

# Stop Formiqo Dev Server

Stop the local uvicorn development server.

## Steps

1. From the repository root, run:

```bash
./scripts/run-dev.sh stop
```

2. If the script fails, report the exact error.

3. Confirm success with:
   - Script output indicating the server was stopped, or that nothing was running on port 8000
   - Port 8000 is no longer in use

4. Do **not** delete `.venv` or uninstall dependencies — only stop the running server.
