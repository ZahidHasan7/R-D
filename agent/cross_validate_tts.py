import os
import sys
import json

# Add project root to path
sys.path.append(os.getcwd())

from agent.asr import transcribe
from jiwer import wer

# Test Cases
test_cases = [
    {"id": 1, "text": "হ্যালো স্যার আমি কীভাবে help করতে পারি?"},
    {"id": 41, "text": "payment হয়ে গেছে কিন্তু order confirm দেখাচ্ছে না"}
]

print("[*] Running Domain-Specific Cross-Validation...")
results = []

for case in test_cases:
    audio_path = f"test_output/eval/tts_{case['id']}.wav"
    ref_text = case['text']
    
    if not os.path.exists(audio_path):
        print(f"[!] skipping {audio_path}, not found.")
        continue
        
    print(f"[*] Transcribing {audio_path}...")
    transcription = transcribe(audio_path)
    
    # Calculate WER with normalized transcript
    error = wer(ref_text, transcription)
    
    results.append({
        "id": case['id'],
        "reference": ref_text,
        "transcription": transcription,
        "WER": round(error * 100, 2)
    })
    print(f"    - Ref: {ref_text}")
    print(f"    - ASR: {transcription}")
    print(f"    - WER: {round(error * 100, 2)}%")

# Save report
with open("test_output/domain_validation.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n[!] Validation Done. Check test_output/domain_validation.json")
