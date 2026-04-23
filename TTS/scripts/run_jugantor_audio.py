import os
import time
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalizer import TextNormalizer
from src.tts_engine import TTSEngine

def generate_audio():
    input_file = os.path.join('tests', 'cases', 'jugantor_test_cases.txt')
    output_dir = os.path.join('results', 'outputs', 'jugantor_outputs')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    normalizer = TextNormalizer(use_ml=True) # Use ML for more natural audio phrasing
    tts = TTSEngine()
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"Synthesizing {len(lines)} cases from {input_file}...")
    
    for i, raw_text in enumerate(lines):
        print(f"Processing [{i+1}/{len(lines)}]: {raw_text[:50]}...")
        
        # 1. Normalize
        normalized_text = normalizer.normalize(raw_text)
        
        # 2. Synthesize
        filename = f"jugantor_{i+1:02d}.mp3"
        filepath = os.path.join(output_dir, filename)
        
        tts.generate_audio(normalized_text, filepath)
        
    print(f"\nAll audio files saved to: {output_dir}/")

if __name__ == '__main__':
    generate_audio()
