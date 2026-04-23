import os
import tempfile
from typing import Optional

def transcribe(file_path: str) -> str:
    """Transcribe audio using OpenAI Whisper (tiny model - fast on CPU)."""
    try:
        import whisper
        # Use "tiny" model for speed on CPU (65MB vs 461MB for "small")
        model = whisper.load_model("tiny")
        res = model.transcribe(str(file_path), language="bn")
        return res.get("text", "")
    except Exception as e:
        return f"[ERROR] ASR failed: {e}"
