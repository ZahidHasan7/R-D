import whisper
from jiwer import wer
import torch
import librosa
import numpy as np

def debug_wer():
    model = whisper.load_model("small")
    audio_path = "test_output/eval/tts_1.wav"
    ref_text = "হ্যালো স্যার আমি কীভাবে help করতে পারি?"
    
    print(f"[*] Debugging WER for {audio_path}")
    result = model.transcribe(audio_path, language="bn")
    hypothesis = result["text"]
    
    print(f"   Reference:  {ref_text}")
    print(f"   Hypothesis: {hypothesis}")
    
    score = wer(ref_text, hypothesis)
    print(f"   WER Score: {score}")

def debug_audio():
    audio_path = "test_output/eval/tts_1.wav"
    y, sr = librosa.load(audio_path, sr=None)
    print(f"[*] Audio Stats for {audio_path}")
    print(f"   Shape: {y.shape}")
    print(f"   SR: {sr}")
    print(f"   Max Amp: {np.max(np.abs(y))}")
    print(f"   Any NaN in audio? {np.isnan(y).any()}")

if __name__ == "__main__":
    debug_audio()
    debug_wer()
