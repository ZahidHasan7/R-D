import os
import torch
import jiwer
import argparse
import numpy as np
import wave
import warnings
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, VitsModel, AutoTokenizer
except ImportError:
    print("Error: transformers library is missing.")
    exit(1)

try:
    from src.normalizer import TextNormalizer
except ImportError:
    TextNormalizer = None

def evaluate_vits2_accuracy(test_file):
    print("==================================================")
    print("     Objective TTS Acoustic Evaluation (ASR)      ")
    print("==================================================")
    
    # 1. Load Sentences
    tests = []
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # If it's a word-list CSV, stitch them into readable sentence chunks of 15 words
            if test_file.endswith('.csv') and len(lines) > 2 and len(lines[1].split()) == 1:
                lines = lines[1:] if lines[0].lower() == 'word' else lines
                chunks = [" ".join(lines[i:i+15]) for i in range(0, len(lines), 15)]
                tests = chunks
            else:
                tests = lines
                
    # Limit max runs to 10 chunks to avoid 19-hour CPU exhaustion
    tests = tests[:10]
    
    if not tests:
        tests = [
            "আজকের ট্রানজ্যাকশন আইডি হল তিন দশমিক এক চার",
            "ডক্টর রহমান পঞ্চাশ কেজি চাল কিনেছেন",
            "আমি আপনাকে কিভাবে সাহায্য করতে পারি"
        ]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Backbone: {device}\n")
    
    # 2. Load Models Once (Heavy Step)
    print("[INIT] Loading VITS2 TTS (facebook/mms-tts-ben)...")
    tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-ben").to(device)
    
    print("[INIT] Loading Whisper ASR (openai/whisper-tiny)...")
    try:
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
    except Exception as e:
        print(f"Error loading whisper pipeline: {e}")
        return
        
    norm = TextNormalizer() if TextNormalizer else None
    
    transcriptions = []
    truths = []
    
    for idx, text in enumerate(tests):
        print(f"\n[{idx+1}/{len(tests)}] Original Input: {text}")
        
        # A. Normalize Text for acoustic model constraint
        input_text = str(norm.normalize(text)) if norm else text
        print(f"      Normalized Target: {input_text}")
        
        # B. Synthesize Audio
        inputs = tts_tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = tts_model(**inputs).waveform
            
        audio_data = output[0].cpu().numpy()
        audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data_16 = (audio_data * 32767).astype(np.int16)
        
        wav_path = "temp_eval.wav"
        with wave.open(wav_path, 'w') as f:
            f.setnchannels(1) # mono channel required for whisper
            f.setsampwidth(2)
            f.setframerate(tts_model.config.sampling_rate)
            f.writeframes(audio_data_16.tobytes())
            
        # C. Transcribe Audio mapping
        try:
            # Whisper handles .wav format natively through the pipeline
            result = asr_pipe(wav_path, generate_kwargs={"language": "bengali"})
            predicted = result['text'].strip()
        except Exception as e:
            print(f"ASR Decoding Failed: {e}")
            predicted = ""
            
        print(f"      ASR Translated   : {predicted}")
        
        truths.append(input_text.lower())
        transcriptions.append(predicted.lower())
        
    if os.path.exists("temp_eval.wav"):
        os.remove("temp_eval.wav")
        
    # D. Evaluate Output Accuracy
    wer_score = jiwer.wer(truths, transcriptions)
    cer_score = jiwer.cer(truths, transcriptions)
    
    print("\n" + "="*50)
    print(f"          FINAL TTS ACOUSTIC WER SCORE  : {wer_score * 100:.2f}%")
    print(f"          FINAL TTS ACOUSTIC CER SCORE  : {cer_score * 100:.2f}%")
    print("="*50)
    print("Note: Lower is better. A low WER implies the TTS synthesized clear,")
    print("human-intelligible audio that the ASR model decoded flawlessly.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_test = os.path.join("tests", "cases", "test_cases.txt")
    parser.add_argument("--test_file", type=str, default=default_test, help="Provide a txt file with lines of text")
    args = parser.parse_args()
    
    evaluate_vits2_accuracy(args.test_file)
