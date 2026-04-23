import os
import tempfile
from typing import Optional

_whisper_model = None

def _load_whisper(name: str = "small"):
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model(name)
        except Exception:
            _whisper_model = None
    return _whisper_model


def transcribe(file_path: str) -> str:
    """Transcribe an audio file using Whisper (preferred) or repository scripts.
    Returns transcribed text or error message.
    """
    # Preferred: use local whisper model (fast to call)
    model = _load_whisper("small")
    if model:
        try:
            res = model.transcribe(str(file_path), language="bn")
            return res.get("text", "")
        except Exception:
            pass

    # Fallback: try calling approach_2A_large implementation if available
    try:
        from STT import approach_2A_large
        # approach_2A_large uses whisper internally; reuse its model if needed
        # but it doesn't expose a single-file API; so fall back to subprocess if present
    except Exception:
        pass

    return "[ERROR] No usable STT inference available. Install 'whisper' or modify agent/asr.py to call your model."
