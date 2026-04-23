import os
import csv
import torch
import wave
import numpy as np
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import VitsModel, AutoTokenizer
from src.normalizer import TextNormalizer
from src.prosody_formatter import ProsodyFormatter
import warnings
warnings.filterwarnings('ignore')

import re

def reconstruct_sentences(file_path):
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
        
    # Legacy check for the original 'bangla_words.csv' 1-word-per-line format
    if content.lower().startswith("word\n") or content.lower().startswith("word,"):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            words = [row.get('word', '').strip() for row in reader if row.get('word', '').strip()]
        content = " ".join(words)
        
    # Split text into sentences using Bangla boundaries (Dari, ?, !) and newlines
    raw_sentences = re.split(r'[৷?!]|\n+', content)
    for s in raw_sentences:
        s = s.strip()
        if s:
            sentences.append(s)
            
    return sentences

def process(input_csv):
    output_dir = os.path.join('results', 'outputs', 'story_outputs_vits')
    batch_size = 5 # Smaller chunking so tensors do not overflow RAM
    
    os.makedirs(output_dir, exist_ok=True)
        
    # Enforce ML translation (use_ml=True) precisely so English dialogue
    # (e.g., "Sure... I’d like that") is translated into Bengali script.
    # Otherwise, the VITS model will silently skip all non-Bengali characters!
    normalizer = TextNormalizer(use_ml=True)
    prosody = ProsodyFormatter()
    
    print("Loading VITS2 Pipeline (facebook/mms-tts-ben)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    model = VitsModel.from_pretrained("facebook/mms-tts-ben").to(device)
    
    sentences = reconstruct_sentences(input_csv)
    total_sentences = len(sentences)
    print(f"Loaded {total_sentences} sentences.")
    
    for i in range(0, total_sentences, batch_size):
        batch_num = (i // batch_size) + 1
        batch_sentences = sentences[i:i + batch_size]
        
        # Normalize sentence strictly one by one so the ML translation model
        # (BanglaT5) doesn't hallucinate or summarize the entire paragraph!
        normalized_sentences = [normalizer.normalize(s) for s in batch_sentences]
        normalized_text = " ".join(normalized_sentences)
        formatted_text = prosody.format(normalized_text)
        
        # Clean formatting tokens since Huggingface pipelines do not natively map <pause>
        clean_text = formatted_text.replace('<pause>', ' ').replace('<break>', ' ').replace('<breath>', ' ')
        
        # CRITICAL FIX: The MMS-TTS model silently drops text if it contains unmapped symbols
        # (like English double quotes ", smart quotes, or ellipsis …).
        import re
        clean_text = re.sub(r'[^\u0980-\u09FF\s?!।]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        filename = f"story_vits_{batch_num:03d}.wav"
        filepath = os.path.join(output_dir, filename)
        
        print(f"Synthesizing Batch {batch_num} -> {filename}...")
        
        try:
            inputs = tokenizer(clean_text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**inputs).waveform
                
            audio_data = output[0].cpu().numpy()
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-9)
            audio_data_16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filepath, 'w') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(model.config.sampling_rate)
                f.writeframes(audio_data_16.tobytes())
        except Exception as e:
            print(f"Failed parsing block: {e}")
            
    print(f"\nVITS processing complete. Target outputs populated safely inside: {output_dir}/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    default_input = os.path.join('data', 'datasets', 'sampledata.csv')
    parser.add_argument("--input", type=str, default=default_input, help="Input text or csv story file")
    args = parser.parse_args()
    
    process(args.input)
