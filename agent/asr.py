import os
import tempfile
from typing import Optional

def transcribe(file_path: str) -> str:
    """Transcribe using Whisper large-v3 + Bengali domain prompt (optimized for call center scenarios)."""
    try:
        import whisper
        
        # Domain-specific prompt for call center conversations
        # (from your STT/approach_2B_prompted.py)
        domain_prompt = (
            "এটি একটি কল সেন্টার কথোপকথন। এখানে order, refund, bKash, "
            "payment, OTP, delivery, tracking, parcel, warranty, replacement, "
            "cancel, support, customer — এই ধরনের শব্দ থাকতে পারে।"
        )
        
        # Load large-v3 model (1.5GB, best accuracy for Bengali)
        model = whisper.load_model("large-v3")
        
        # Transcribe with domain context
        result = model.transcribe(
            str(file_path),
            language="bn",
            initial_prompt=domain_prompt
        )
        
        return result.get("text", "")
    except Exception as e:
        return f"[ERROR] ASR failed: {e}"
