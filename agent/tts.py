import os
import sys
from io import BytesIO
from gtts import gTTS

_normalizer = None

def _init_vits():
    """Lazily initialize the TextNormalizer (Kept name for backward API compatibility)."""
    global _normalizer
    if _normalizer is None:
        # Add TTS root to sys.path to allow imports from src.*
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tts_root = os.path.abspath(os.path.join(current_dir, '..', 'TTS'))
        if tts_root not in sys.path:
            sys.path.append(tts_root)
        
        try:
            from src.normalizer import TextNormalizer
            
            print("Loading Text Normalizer pipeline...")
            _normalizer = TextNormalizer(use_ml=False)
            print("Online gTTS pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing TTS pipeline: {e}")
            raise RuntimeError(f"Could not load normalizer: {e}")

def synthesize(text: str) -> bytes:
    """
    Online Bangla TTS synthesis using Google TTS (gTTS) and custom normalizer.
    Returns MP3 bytes.
    """
    if not text:
        return b''
        
    _init_vits()
    
    # 1. Pipeline Front-End: Multi-stage Normalization
    norm_text = _normalizer.normalize(text)
    
    # 2. Online Audio Generation
    tts = gTTS(text=norm_text, lang='bn', slow=False)
    
    # 3. Packing into memory
    bio = BytesIO()
    tts.write_to_fp(bio)
    return bio.getvalue()
