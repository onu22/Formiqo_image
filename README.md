## Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
./scripts/run_https.sh
```

The dev server uses a local self-signed certificate (`certs/`). Your browser will warn about the cert on first visit — that is expected for local HTTPS. Regenerate with `./scripts/generate_dev_certs.sh` if needed.

https://127.0.0.1:8000/docs#/

- openai: `gpt-4o or gpt-5.5`
- anthropic: `claude-opus-4-6 or claude-opus-4-7`

### Grounding (optional env)



