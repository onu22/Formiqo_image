## Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

deactivate
```

- OpenAI: `gpt-5.5`
- Anthropic: `claude-opus-4-7`

Stamping **checkbox** and **radio** fields draws a **vector checkmark** (two strokes inside the bbox), not a text glyph—consistent appearance across fonts and PDF viewers.


