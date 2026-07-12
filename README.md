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


