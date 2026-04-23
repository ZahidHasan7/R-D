import os
import tempfile
import re
import unicodedata
from typing import Optional

def clean_bengali_output(text: str) -> str:
    """
    Clean corrupted Whisper output (repeated conjuncts).
    Whisper sometimes produces: নেনেনেনে instead of: নে
    """
    if not text:
        return ""
    
    # Fix repeated conjuncts (e.g., নেনেনে → নে)
    text = re.sub(r'([\u0980-\u09FF][\u09BE-\u09FF])(\1)+', r'\1', text)
    
    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def transcribe(file_path: str) -> str:
    """Transcribe using Whisper base + Bengali domain prompt (optimized for CPU)."""
    try:
        import whisper
        
        # Domain-specific prompt for call center conversations
        domain_prompt = (
            "এটি একটি কল সেন্টার কথোপকথন। এখানে order, refund, bKash, "
            "payment, OTP, delivery, tracking, parcel, warranty, replacement, "
            "cancel, support, customer — এই ধরনের শব্দ থাকতে পারে।"
        )
        
        # Load Whisper base model (140MB) - good balance for CPU
        model = whisper.load_model("base")
        
        # Transcribe with strict language enforcement
        result = model.transcribe(
            str(file_path),
            language="bn",                    # Force Bengali
            initial_prompt=domain_prompt,     # Domain context
            temperature=0.2,                  # Lower randomness
            best_of=1,                        # Single pass for speed
            fp16=False                        # CPU compatibility
        )
        
        text = result.get("text", "").strip()
        
        # Clean up corrupted output
        text = clean_bengali_output(text)
        
        # If we got empty or very short result
        if not text or len(text) < 3:
            return "[No speech detected or too short]"
        
        return text
        
    except Exception as e:
        return f"[ERROR] Transcription failed: {str(e)}"
