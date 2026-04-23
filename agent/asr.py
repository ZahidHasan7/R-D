import os
import tempfile
from typing import Optional

def transcribe(file_path: str) -> str:
    """Transcribe audio using OpenAI Whisper (base model - better accuracy)."""
    try:
        import whisper
        # Use "base" model for better accuracy (140MB, ~10sec per minute on CPU)
        model = whisper.load_model("base")
        res = model.transcribe(str(file_path), language="bn")
        return res.get("text", "")
    except Exception as e:
        return f"[ERROR] ASR failed: {e}"
