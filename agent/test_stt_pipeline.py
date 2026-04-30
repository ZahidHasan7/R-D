import sys, os
import torch

# Add project root and TTS dir to path
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('TTS'))

from agent.asr import transcribe

def test_stt():
    # Test with ref_speaker.wav
    audio_path = "ref_speaker.wav"
    if not os.path.exists(audio_path):
        print(f"[!] {audio_path} not found.")
        return
        
    print(f"[*] Transcribing {audio_path}...")
    try:
        text = transcribe(audio_path)
        print(f"[+] Transcript: {text}")
        
        # Test cleaning/normalization logic
        test_phrases = ["আমি অডার করতে চাই", "বিকাশ পেমেন্ট", "রিফান্ড চাই"]
        from agent.asr import clean_bengali_output
        print("\n[*] Testing Normalization Logic:")
        for p in test_phrases:
            clean = clean_bengali_output(p)
            print(f"RAW: {p} -> CLEAN: {clean}")
            
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    test_stt()
