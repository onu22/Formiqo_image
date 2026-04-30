## Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

openai
gpt-5.5
---
anthropic
claude-opus-4-7


Set these optional env vars in `.env` (prefix is `FORMIQO_`):



