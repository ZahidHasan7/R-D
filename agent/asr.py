import os
import tempfile
from typing import Optional

def transcribe(file_path: str) -> str:
    """Transcribe audio using OpenAI Whisper."""
    try:
        import whisper
        model = whisper.load_model("small")
        res = model.transcribe(str(file_path), language="bn")
        return res.get("text", "")
    except Exception as e:
        return f"[ERROR] ASR failed: {e}"
