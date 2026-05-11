## Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


http://127.0.0.1:8000/docs#/

- OpenAI: `gpt-5.5`
- Anthropic: `claude-opus-4-7`



