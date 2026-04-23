from process_story_vits import reconstruct_sentences
import sys
import os

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalizer import TextNormalizer
from src.prosody_formatter import ProsodyFormatter
import re

sentences = reconstruct_sentences(os.path.join('data', 'datasets', 'sampledata.csv'))
normalizer = TextNormalizer(use_ml=True)
prosody = ProsodyFormatter()

batch_size = 5
for i in range(0, len(sentences), batch_size):
    batch_num = (i // batch_size) + 1
    batch_sentences = sentences[i:i + batch_size]
    batch_text = " ".join(batch_sentences)
    
    if batch_num == 4:
        print(f"\n--- BATCH {batch_num} ---")
        print("RAW BATCH TEXT:", batch_text)
        
        normalized_text = normalizer.normalize(batch_text)
        print("NORMALIZED TEXT:", normalized_text)
        
        formatted_text = prosody.format(normalized_text)
        clean_text = formatted_text.replace('<pause>', ' ').replace('<break>', ' ').replace('<breath>', ' ')
        clean_text = re.sub(r'[^\u0980-\u09FF\s?!।]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        print("CLEAN TEXT TO VITS:", clean_text)
