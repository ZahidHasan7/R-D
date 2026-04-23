import os
import csv
import re
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalizer import TextNormalizer
from src.tts_engine_v2 import TTSEngineV2, TTSBackend
from src.prosody_formatter import ProsodyFormatter

def reconstruct_sentences(csv_path):
    """
    Reads word-per-line CSV and joins words into sentences based on punctuation.
    """
    sentences = []
    current_sentence = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word'].strip()
            if not word: continue
            
            current_sentence.append(word)
            
            # Check for sentence enders: ৷ (Bangla Dari), ?, !
            # Also force split if the sentence gets too long (e.g. 25 words)
            if any(ender in word for ender in ['৷', '?', '!']) or len(current_sentence) >= 25:
                sentences.append(" ".join(current_sentence))
                current_sentence = []
    
    # Add any remaining words as the last sentence
    if current_sentence:
        sentences.append(" ".join(current_sentence))
        
    return sentences

def process():
    input_csv = os.path.join('data', 'datasets', 'bangla_words.csv')
    output_dir = os.path.join('results', 'outputs', 'story_outputs_optimized')
    batch_size = 50 # sentences per audio file
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    normalizer = TextNormalizer(use_ml=False)
    prosody = ProsodyFormatter()
    tts = TTSEngineV2(lang='bn', backend=TTSBackend.GTTS)
    
    print(f"Loading words from {input_csv}...")
    sentences = reconstruct_sentences(input_csv)
    total_sentences = len(sentences)
    print(f"Reconstructed {total_sentences} sentences.")
    
    # Process in batches
    for i in range(0, total_sentences, batch_size):
        batch_num = (i // batch_size) + 1
        batch_sentences = sentences[i:i + batch_size]
        
        # 1. Join, Normalize, and Format Prosody
        batch_text = " ".join(batch_sentences)
        print(f"Normalizing Batch {batch_num}...")
        normalized_text = normalizer.normalize(batch_text)
        
        # Apply prosody formatting (which replaces tokens with markers)
        # and synthesis will replace markers with minimalist spaces for shorter pauses
        formatted_text = prosody.format(normalized_text)
        
        # 2. Synthesize
        filename = f"story_optimized_{batch_num:03d}.mp3"
        filepath = os.path.join(output_dir, filename)
        
        print(f"Synthesizing {filename}...")
        try:
            # TTSEngineV2 will strip markers and replace with single spaces for fast flow
            tts.generate_audio(formatted_text, filepath, strip_markers=True)
        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            continue

        if batch_num >= 3: # Smaller demo for fast verification
            break

    print(f"\nProcessing complete. Check segments in: {output_dir}/")

if __name__ == '__main__':
    process()
