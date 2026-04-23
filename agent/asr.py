import os
import tempfile
from typing import Optional

def transcribe(file_path: str) -> str:
    """Mock ASR: return a canned response for demo.
    In production, replace this with real Whisper or your model inference.
    """
    # Mock response for demo
    return "আপনার অডিও ফাইল সফলভাবে প্রতিলিপি করা হয়েছে। এটি একটি ডেমো প্রতিক্রিয়া।"
    
    # Real implementation (commented out for now):
    # try:
    #     import whisper
    #     model = whisper.load_model("small")
    #     res = model.transcribe(str(file_path), language="bn")
    #     return res.get("text", "")
    # except Exception as e:
    #     return f"[ERROR] ASR failed: {e}"
