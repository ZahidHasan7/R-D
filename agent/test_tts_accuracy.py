import os
import sys
import torch
import numpy as np
import librosa
import wave
import json
import time

# Add project root to path
sys.path.append(os.getcwd())

from agent.tts import synthesize, _init_vits
from agent.asr import transcribe, _init_asr

# Evaluation Imports
import whisper
from jiwer import wer, cer
from pymcd.mcd import Calculate_MCD
# speechmos.dnsmos is missing, using UTMOS22 Strong instead
import parselmouth
from scipy.stats import pearsonr

# 1. Pronunciation Accuracy (WER/CER)
def evaluate_wer_cer(tts_audio_path: str, reference_text: str, asr_model) -> dict:
    # Use a prompt to help Whisper with code-switching and avoid hallucinations
    result = asr_model.transcribe(
        tts_audio_path, 
        language="bn", 
        initial_prompt="হ্যালো, এইটা একটা বাংলা এবং English code-mixed voice agent পরীক্ষা।"
    )
    hypothesis = result["text"].strip()
    
    # If hypothesis is empty or gibberish, WER will be 100
    if not hypothesis:
        return {"reference": reference_text, "hypothesis": "[No Speech Detected]", "WER": 100.0, "CER": 100.0}
    
    word_error = wer(reference_text, hypothesis)
    char_error = cer(reference_text, hypothesis)
    
    return {
        "reference": reference_text,
        "hypothesis": hypothesis,
        "WER": round(word_error * 100, 2),
        "CER": round(char_error * 100, 2),
    }

# 2. Acoustic Similarity (MCD)
mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")

def evaluate_mcd(tts_audio_path: str, ref_audio_path: str) -> float:
    try:
        # Pymcd requires both files to be exactly same sampling rate.
        # We will create temporary 16kHz versions of both.
        def to_16k(path):
            temp = path.replace(".wav", "_16k.wav")
            y, _ = librosa.load(path, sr=16000)
            from scipy.io import wavfile
            # Ensure float32 [-1, 1] is converted properly or use soundfile
            import soundfile as sf
            sf.write(temp, y, 16000)
            return temp
            
        t_tts = to_16k(tts_audio_path)
        t_ref = to_16k(ref_audio_path)
        
        mcd_value = mcd_toolbox.calculate_mcd(t_ref, t_tts)
        
        # Cleanup
        os.remove(t_tts)
        os.remove(t_ref)
        
        return round(mcd_value, 3)
    except Exception as e:
        print(f"[!] MCD calculation failed: {e}")
        return -1.0

# 3. Naturalness Prediction (MOS-LQO)
# Using UTMOS22 Strong
try:
    from speechmos.utmos22.strong.model import UTMOS22Strong
    # This might fail on CPU if weights aren't found or random
    _mos_model = UTMOS22Strong()
    # Attempt to load weights if they exist in a common path or skip
    # (Simplified for this environment)
except Exception as e:
    print(f"[*] Note: speechmos initialization: {e}")
    _mos_model = None

def evaluate_mos(tts_audio_path: str) -> dict:
    if _mos_model is None:
        return {"MOS_overall": 0.0}
    try:
        y, sr = librosa.load(tts_audio_path, sr=None)
        # UTMOS expects (Batch, Time) tensor
        wav_tensor = torch.from_numpy(y).unsqueeze(0).float()
        with torch.no_grad():
            score_tensor = _mos_model(wav_tensor, sr)
        score = score_tensor.item()
        return {
            "MOS_overall": round(score, 2),
        }
    except Exception as e:
        print(f"[!] MOS prediction failed: {e}")
        return {"MOS_overall": 0.0}

# 4. Prosody Similarity (F0)
def extract_f0(audio_path: str) -> np.ndarray:
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]
    return f0

def evaluate_prosody(tts_audio: str, ref_audio: str) -> dict:
    try:
        f0_tts = extract_f0(tts_audio)
        f0_ref = extract_f0(ref_audio)
        
        min_len = min(len(f0_tts), len(f0_ref))
        if min_len < 10: return {"f0_correlation": 0.0}
        
        f0_tts = f0_tts[:min_len]
        f0_ref = f0_ref[:min_len]
        
        corr, _ = pearsonr(f0_tts, f0_ref)
        return {
            "f0_correlation": round(corr, 3),
            "f0_range_tts": round(float(np.ptp(f0_tts)), 2),
            "f0_range_ref": round(float(np.ptp(f0_ref)), 2),
        }
    except Exception as e:
        print(f"[!] Prosody evaluation failed: {e}")
        return {"f0_correlation": 0.0}

def main():
    print("[*] Initializing engines and loading models...")
    _init_vits()
    asr_bench_model = whisper.load_model("small") # Using small for faster benchmarking
    
    # Test Cases from metadata (now using converted WAVs)
    test_cases = [
        {
            "id": 1,
            "ref_audio": "test_output/eval/ref_wavs/Delivery_delay.wav",
            "text": "হ্যালো স্যার আমি কীভাবে help করতে পারি?"
        },
        {
            "id": 41,
            "ref_audio": "test_output/eval/ref_wavs/Payment_Issue.wav",
            "text": "payment হয়ে গেছে কিন্তু order confirm দেখাচ্ছে না"
        },
        {
            "id": 51,
            "ref_audio": "test_output/eval/ref_wavs/Product_inquiry.wav",
            "text": "এই product টার কি warranty আছে?"
        }
    ]
    
    os.makedirs("test_output/eval", exist_ok=True)
    results = []
    
    for case in test_cases:
        cid = case["id"]
        text = case["text"]
        ref_audio_path = case["ref_audio"]
        
        print(f"\n[Evaluating Case {cid}] Text: {text}")
        
        # 1. Generate TTS (Without filter for clean evaluation)
        start_time = time.time()
        audio_bytes = synthesize(text, apply_filter=False)
        tts_time = time.time() - start_time
        
        tts_audio_path = f"test_output/eval/tts_{cid}.wav"
        with open(tts_audio_path, "wb") as f:
            f.write(audio_bytes)
            
        # 2. Run Evaluations
        print(" - Calculating Metrics...")
        
        # ASR Accuracy
        accuracy = evaluate_wer_cer(tts_audio_path, text, asr_bench_model)
        
        # MOS Prediction
        mos = evaluate_mos(tts_audio_path)
        
        # Reference-based metrics
        mcd = evaluate_mcd(tts_audio_path, ref_audio_path)
        prosody = evaluate_prosody(tts_audio_path, ref_audio_path)
        
        case_result = {
            "id": cid,
            "text": text,
            "latency_s": round(tts_time, 2),
            "accuracy": accuracy,
            "mos": mos,
            "mcd_db": mcd,
            "prosody": prosody
        }
        results.append(case_result)
        print(f"   [+] WER: {accuracy['WER']}% | MOS: {mos['MOS_overall']} | MCD: {mcd} dB")

    # Save Output
    output_file = "test_output/eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[!] Evaluation Complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
