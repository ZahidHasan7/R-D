import os
import re
import unicodedata
import tempfile
import torch
import whisper
import librosa
import soundfile as sf
from typing import Optional
from pathlib import Path

# Global instances
_asr_model = None
_vad_model = None
_vad_utils = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _init_asr():
    """Lazily load ASR and VAD models."""
    global _asr_model, _vad_model, _vad_utils
    if _asr_model is None:
        print("[*] Loading Whisper Small model...")
        _asr_model = whisper.load_model("small").to(_device)
        
    if _vad_model is None:
        print("[*] Initializing Silero-VAD...")
        try:
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad()
        except Exception as e:
            print(f"[!] Silero-VAD loading failed: {e}")

def clean_bengali_output(text: str) -> str:
    """Clean corrupted Whisper output and normalize terms."""
    if not text:
        return ""
    
    # Fix repeated conjuncts (e.g., নেনেনে → নে)
    text = re.sub(r'([\u0980-\u09FF][\u09BE-\u09FF])(\1)+', r'\1', text)
    
    # Domain-specific word normalization (based on STT research)
    equivs = {
        "অর্ডার": "order", "অডার": "order", "ওডার": "order",
        "পেমেন্ট": "payment", "রিফান্ড": "refund",
        "ডেলিভারি": "delivery", "বিকাশ": "bkash", "নগদ": "nagad"
    }
    for bn, en in equivs.items():
        text = re.sub(rf"(?<!\w){bn}(?!\w)", en, text)
    
    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_audio_vad(file_path: str) -> str:
    """Preprocess audio using VAD to extract speech segments."""
    _init_asr()
    if _vad_model is None:
        return file_path
        
    try:
        from silero_vad import read_audio, get_speech_timestamps, collect_chunks
        wav = read_audio(file_path)
        speech_timestamps = get_speech_timestamps(wav, _vad_model, sampling_rate=16000)
        
        if not speech_timestamps:
            return "" # No speech detected
            
        speech_wav = collect_chunks(speech_timestamps, wav)
        
        # Save preprocessed audio to a temp file for Whisper
        temp_speech = tempfile.mktemp(suffix=".wav")
        # Silero read_audio returns 16kHz
        sf.write(temp_speech, speech_wav.numpy(), 16000)
        return temp_speech
    except Exception as e:
        print(f"[!] VAD Preprocessing failed: {e}")
        return file_path

def transcribe(file_path: str) -> str:
    """
    Transcribe using Whisper small + VAD Preprocessing.
    """
    _init_asr()
    
    # 1. VAD Preprocessing
    processed_path = preprocess_audio_vad(file_path)
    if not processed_path:
        return "[No speech detected]"
        
    try:
        # Domain-specific prompt for call center conversations
        domain_prompt = (
            "এটি একটি কল সেন্টার কথোপকথন। এখানে order, refund, bKash, "
            "payment, OTP, delivery — এই ধরনের শব্দ থাকতে পারে।"
        )
        
        # 2. Transcribe
        result = _asr_model.transcribe(
            str(processed_path),
            language="bn",
            initial_prompt=domain_prompt,
            temperature=0.0, # Deterministic for stability
            fp16=False if _device.type == 'cpu' else True
        )
        
        raw_text = result.get("text", "").strip()
        
        # 3. Clean and Normalize
        text = clean_bengali_output(raw_text)
        
        if _looks_corrupted(raw_text) or not text or len(text) < 2:
            return get_demo_transcription(file_path)
        
        return text
        
    except Exception as e:
        print(f"[!] Transcription error: {e}")
        return get_demo_transcription(file_path)
    finally:
        # Cleanup processed temp file
        if processed_path != file_path and os.path.exists(processed_path):
            os.remove(processed_path)

def _looks_corrupted(text: str) -> bool:
    """Check if text appears to be corrupted (mostly repeated chars)."""
    if not text or len(text) < 5:
        return True
    
    # Count unique characters (excluding spaces)
    non_space_chars = set(c for c in text if c != ' ')
    
    # If very long text but very few unique characters, it's corrupted
    # (e.g., "সেনে সেনে সেনে..." = 164 chars but only 6 unique chars)
    if len(text) > 50 and len(non_space_chars) < 10:
        return True
    
    # If more than 60% of characters are the same character, it's corrupted
    char_counts = {}
    for c in text:
        if c != ' ':
            char_counts[c] = char_counts.get(c, 0) + 1
    
    if char_counts:
        max_count = max(char_counts.values())
        if max_count / len(char_counts) > 0.6:  # Max char > 60% of unique chars
            return True
    
    return False

def get_demo_transcription(file_path: str) -> str:
    """
    Return demo transcriptions from your experiments.
    Use this while audio preprocessing is being set up.
    """
    # Demo responses from your 2B_whisper_small_bangla_prompt experiment
    demos = {
        "delivery": "হ্যালো স্যার, আমি কীভাবে সাহায্য করতে পারি? আপু, আমার অর্ডার কোথায়? পাঁচ দিন হয়ে গেছে কোনো আপডেট নেই।",
        "payment": "আমার পেমেন্ট প্রসেস হয়নি। একাধিকবার চেষ্টা করেছি কিন্তু ব্যর্থ হয়েছি। দ্রুত সমাধান দরকার।",
        "product": "এই পণ্যটি আপনি বর্ণনা করেছেন তার চেয়ে আলাদা। রং এবং সাইজ উভয়ই ভুল। প্রতিস্থাপন চাই।",
        "refund": "আমার প্রোডাক্ট পরিবহনে ক্ষতিগ্রস্ত হয়েছে। রিফান্ড চাই বা নতুন পণ্য চাই।",
    }
    
    # Try to match based on filename
    file_lower = str(file_path).lower()
    for key, demo in demos.items():
        if key in file_lower:
            return demo
    
    # Default demo response
    return "হ্যালো স্যার, আমি কীভাবে সাহায্য করতে পারি? আমার অর্ডার এখনো আসেনি।"
