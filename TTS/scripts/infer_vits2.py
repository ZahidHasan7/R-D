import os
import torch
import wave
import struct
import argparse
import numpy as np
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from transformers import VitsModel, AutoTokenizer
except ImportError:
    print("Error: transformers library is missing.")
    exit(1)

# Import the project's powerful hybrid front-end normalizer
try:
    from src.normalizer import TextNormalizer
except ImportError:
    print("Warning: src.normalizer.py not found. Raw text will be used without formatting.")
    TextNormalizer = None

def synthesize(text, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- VITS INFERENCE PIPELINE ---")
    print(f"Device: {device}")
    
    # 1. Front-End: Text Normalization
    # This proves the value of the pipeline! It translates English abbreviations, 
    # weights, and numbers into valid phonetical Bangla strings.
    if TextNormalizer:
        norm = TextNormalizer()
        input_text = str(norm.normalize(text))
        print(f"\n[1] Normalized Input: {input_text}")
    else:
        input_text = text
        print(f"\n[1] Raw Input (No Normalizer): {input_text}")
        
    print("\n[2] Connecting to Hugging Face: Loading 'facebook/mms-tts-ben'...")
    # Load model and tokenizer (automatically downloads the ~130MB checkpoint if missing)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    model = VitsModel.from_pretrained("facebook/mms-tts-ben").to(device)
    
    print("[3] Synthesizing physical audio waves...")
    # Tokenize normalized string (Bangla characters -> tokens)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model(**inputs).waveform
        
    # Standardize neural tensor format
    audio_data = output[0].cpu().numpy()
    
    # Normalize volume and convert to 16-bit PCM for WAV packing
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data = (audio_data * 32767).astype(np.int16)
    
    sample_rate = model.config.sampling_rate
    print(f"[4] Saving exact audio stream to: {output_path} ({sample_rate} Hz)")
    
    # Use native wave (no extra soundfile dependency needed!)
    with wave.open(output_path, 'w') as f:
        f.setnchannels(1) # Mono channel
        f.setsampwidth(2) # 16 bit
        f.setframerate(sample_rate)
        f.writeframes(audio_data.tobytes())
        
    print("Success! You can now listen to the generated file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Complex test case to prove normalization + TTS works end-to-end
    parser.add_argument("--text", type=str, default="জনাব রহমান 3.5kg আপেল এবং 10cm দড়ি কিনেছেন।", help="Text to convert")
    default_output = os.path.join("results", "outputs", "output_test.wav")
    parser.add_argument("--output", type=str, default=default_output, help="Path to save the audio file")
    args = parser.parse_args()
    
    synthesize(args.text, args.output)
