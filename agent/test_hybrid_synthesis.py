import sys, os
import torch
import numpy as np
import wave

# Add project root and TTS dir to path
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('TTS'))

from agent.tts import synthesize

def test_hybrid():
    test_cases = [
        "হ্যালো! হ্যালো Zahid বলছি from OpenAI। Actually আজ সকাল এ আমাদের final deployment ছিল।",
        "The quick brown fox jumps over the lazy dog.",
        "আমি ১৯৫২ সালের ভাষা আন্দোলনের শহীদদের প্রতি শ্রদ্ধা জানাই।"
    ]
    
    if not os.path.exists("test_output"):
        os.makedirs("test_output")
        
    for i, text in enumerate(test_cases):
        print(f"[*] Testing Case {i+1}: {text}")
        try:
            audio_bytes = synthesize(text)
            if audio_bytes:
                filename = f"test_output/hybrid_test_{i+1}.wav"
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                print(f"[+] Success! Audio saved to {filename}")
            else:
                print(f"[!] Failed to generate audio for Case {i+1}")
        except Exception as e:
            print(f"[!] Error in Case {i+1}: {e}")
        print("-" * 30)

if __name__ == "__main__":
    test_hybrid()
