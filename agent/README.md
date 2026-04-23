# Voice Agent API (minimal)

This folder contains a minimal FastAPI scaffold to expose STT and TTS inferences from this repository.

Endpoints
- `POST /asr` - multipart file upload (wav/mp3/m4a/flac). Returns JSON `{ "text": "..." }`.
- `POST /tts` - JSON `{ "text": "..." }`. Returns `audio/wav` bytes.
- `POST /agent` - multipart file upload; returns audio reply (echo-based) using ASR -> TTS pipeline.

Quick start

Install dependencies (prefer inside a venv):
```bash
pip install -r agent/requirements.txt
```

Run server:
```bash
uvicorn agent.server:app --host 0.0.0.0 --port 8000
```

Notes
- The wrappers try to call functions present in existing `STT/` and `TTS/` modules. If those functions are unavailable, the service will raise an error. Edit `agent/asr.py` and `agent/tts.py` to point to your actual inference functions or model paths.
