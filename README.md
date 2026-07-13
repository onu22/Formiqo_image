## MVP build harness

Multi-agent operating system for delivering [`docs/PRD.md`](docs/PRD.md), inspired by
[10Legs/freelance-developer-harness](https://github.com/10Legs/freelance-developer-harness).

| Doc | Purpose |
|-----|---------|
| [`AGENTS.md`](AGENTS.md) | Team roster and governance |
| [`harness/README.md`](harness/README.md) | Harness quick start |
| [`harness/OPERATING-MODEL.md`](harness/OPERATING-MODEL.md) | Gates, pipeline, rules |
| [`harness/sprints/CURRENT`](harness/sprints/CURRENT) | Active sprint backlog |

**Cursor commands:** `/run-mvp` · `/pm` · `/epic E1` · `/gate G1` · `/sprint-plan` · `/run` · `/stop`

**Status:** `./scripts/harness-status.sh` · **Next action:** `./scripts/harness-next.sh`

**Hands-off MVP:** `/run-mvp` — autonomous conductor to G4 ship ([`harness/RUN-MVP.md`](harness/RUN-MVP.md))

## Quick start (Cursor)

Type `/run` in Cursor chat to:

- create `.venv` if missing
- install dependencies when needed
- start `uvicorn` on port 8000
- open Swagger docs in Firefox at `http://127.0.0.1:8000/docs#/`

Or run directly:

```bash
./scripts/run-dev.sh
```

Stop the server with `/stop` in Cursor chat, or:

```bash
./scripts/run-dev.sh stop
```

## Manual install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: http://127.0.0.1:8000/docs#/

- openai: `gpt-4o or gpt-5.5`
- anthropic: `claude-opus-4-6 or claude-opus-4-7`

### Grounding (optional env)


