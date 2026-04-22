## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

```bash
pip3 install -r requirements-dev.txt
pytest
```

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Field Grounding POC

Run with defaults from env:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/jobs/<job_id>/ground-fields"
```
{
  "provider": "anthropic",
  "model": "claude-opus-4-7"
}

{
  "provider": "openai",
  "model": "gpt-4o"
}
