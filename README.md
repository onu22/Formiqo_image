## Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


http://127.0.0.1:8000/docs#/

- openai: `gpt-4o or gpt-5.5`
- anthropic: `claude-opus-4-6 or claude-opus-4-7`

### Grounding geometry (optional env)

- `FORMIQO_GROUNDING_LINE_PADDING_PX` (default `3`) — line intersection padding in validation
- `FORMIQO_GROUNDING_STAMP_INSET_PX` (default `2`) — inset when snapping bboxes to cells/bands



