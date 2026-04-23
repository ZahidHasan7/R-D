import os
import tempfile
from typing import Optional

def transcribe(file_path: str) -> str:
    """Mock ASR for demo (Whisper is too heavy for CPU)."""
    return "এটি একটি নমুনা প্রতিলিপি। আপনার অডিও ফাইল সফলভাবে পড়া হয়েছে।"
    
    # Real Whisper implementation (commented - too slow on CPU):
    # try:
    #     import whisper
    #     model = whisper.load_model("small")
    #     res = model.transcribe(str(file_path), language="bn")
    #     return res.get("text", "")
    # except Exception as e:
    #     return f"[ERROR] ASR failed: {e}"
