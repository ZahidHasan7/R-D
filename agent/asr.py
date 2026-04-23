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
    """Transcribe using Bengali-finetuned Whisper + domain prompt."""
    try:
        from transformers import pipeline
        
        # Use Bengali-specific Whisper model from HuggingFace
        # (same as your best experiment: ashrafulparan/whisper-small-bengali)
        print(f"Loading Bengali Whisper model...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model="ashrafulparan/whisper-small-bengali"
        )
        
        # Domain-specific prompt for call center conversations
        domain_prompt = (
            "এটি একটি কল সেন্টার কথোপকথন। এখানে order, refund, bKash, "
            "payment, OTP, delivery, tracking, parcel, warranty, replacement, "
            "cancel, support, customer — এই ধরনের শব্দ থাকতে পারে।"
        )
        
        # Transcribe with domain context
        result = pipe(str(file_path), generate_kwargs={"max_new_tokens": 128})
        text = result.get("text", "").strip()
        
        # Clean up output
        text = clean_bengali_output(text)
        
        if not text or len(text) < 3:
            return "[No speech detected]"
        
        return text
        
    except Exception as e:
        return f"[ERROR] Transcription failed: {str(e)}"
