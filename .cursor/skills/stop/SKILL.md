---
name: stop
description: Stop the Formiqo FastAPI backend and Vite frontend started by /run.
disable-model-invocation: true
---

# Stop Formiqo Dev Servers

Stop the local uvicorn backend and Vite frontend.

## Steps

1. From the repository root, run:

```bash
./scripts/run-dev.sh stop
```

2. If the script fails, report the exact error.

3. Confirm success with:
   - Script output indicating servers were stopped, or that nothing was running
   - Ports 8000 and 5173 are no longer in use

4. Do **not** delete `.venv`, `node_modules`, or uninstall dependencies — only stop the running servers.
