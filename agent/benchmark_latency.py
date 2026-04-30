import time
import os
import sys
import torch
import numpy as np

# Add project root and TTS dir to path
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('TTS'))

from agent.asr import transcribe, _init_asr
from agent.tts import synthesize, _init_vits

def benchmark_pipeline():
    test_cases = [
        {"text": "হ্যালো! হ্যালো Zahid বলছি from OpenAI। Actually আজ সকাল এ আমাদের final deployment ছিল।", "audio": "ref_speaker.wav"},
        {"text": "The quick brown fox jumps over the lazy dog.", "audio": None},
        {"text": "আমি ১৯৫২ সালের ভাষা আন্দোলনের শহীদদের প্রতি শ্রদ্ধা জানাই।", "audio": None}
    ]
    
    # Warup
    print("[*] Warming up engines...")
    _init_asr()
    _init_vits()
    
    results = []
    
    for i, case in enumerate(test_cases):
        text = case["text"]
        audio_in = case["audio"]
        print(f"\n[Benchmarking Case {i+1}]")
        
        case_data = {"id": i+1, "text": text}
        
        # 1. ASR Benchmark (if audio provided)
        if audio_in and os.path.exists(audio_in):
            start_asr = time.time()
            _ = transcribe(audio_in)
            asr_time = time.time() - start_asr
            case_data["asr_latency"] = asr_time
            print(f" - ASR Latency: {asr_time:.2f}s")
        
        # 2. TTS Benchmark
        start_tts = time.time()
        audio_out = synthesize(text)
        tts_time = time.time() - start_tts
        case_data["tts_latency"] = tts_time
        print(f" - TTS Latency: {tts_time:.2f}s")
        
        # Calculate RTF for TTS
        if audio_out:
            import wave
            import io
            with wave.open(io.BytesIO(audio_out), 'rb') as wav:
                duration = wav.getnframes() / wav.getframerate()
                rtf = tts_time / duration if duration > 0 else 0
                case_data["duration"] = duration
                case_data["rtf"] = rtf
                print(f" - Audio Duration: {duration:.2f}s")
                print(f" - TTS RTF: {rtf:.2f}x")
        
        results.append(case_data)

    print("\n" + "="*40)
    print("Latency Summary")
    print("="*40)
    for r in results:
        print(f"Case {r['id']}: TTS={r['tts_latency']:.2f}s", end="")
        if "asr_latency" in r:
            print(f", ASR={r['asr_latency']:.2f}s", end="")
        if "rtf" in r:
            print(f", RTF={r['rtf']:.2f}x", end="")
        print()

if __name__ == "__main__":
    benchmark_pipeline()
