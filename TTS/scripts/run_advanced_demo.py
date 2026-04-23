"""
run_advanced_demo.py — Executes the Production Bangla TTS pipeline on the Advanced Demo Dataset.
"""

import sys
import os

# Add root to sys.path to allow imports from src/ and scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate import evaluate
from scripts.main import build_pipeline, run_pipeline

# Paths
TEST_FILE  = os.path.join('tests', 'cases', 'demo_v2_test_cases.txt')
TRUTH_FILE = os.path.join('tests', 'ground_truth', 'demo_v2_ground_truth.txt')
OUTPUT_DIR = os.path.join('results', 'outputs', 'demo_outputs')

def main():
    print("="*65)
    print("  PRODUCTION BANGLA TTS — ADVANCED DEMO (v2)")
    print("="*65)
    
    if not os.path.exists(TEST_FILE) or not os.path.exists(TRUTH_FILE):
        print("Error: Demo files not found. Please ensure demo_v2_test_cases.txt and demo_v2_ground_truth.txt exist.")
        return

    # 1. Run Evaluation Report
    print("\nPhase 1: Running Accuracy Report...")
    metrics = evaluate(test_file=TEST_FILE, truth_file=TRUTH_FILE, verbose=True)
    
    # 2. Generate Demo Audio for top cases
    print("\nPhase 2: Generating Audio for Visual/Audio Verification...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build pipeline components
    normalizer, detector, ner, g2p, prosody, tts, log = build_pipeline()
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        cases = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Process all cases
    for i, raw_text in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] Processing: {raw_text}")
        norm, prose, before_p, after_p = run_pipeline(
            raw_text, normalizer, detector, ner, g2p, prosody, tts, log, i
        )
        
        # Override output paths for the demo folder
        new_before = os.path.join(OUTPUT_DIR, f'demo_before_{i}.mp3')
        new_after = os.path.join(OUTPUT_DIR, f'demo_after_{i}.mp3')
        os.replace(before_p, new_before)
        os.replace(after_p, new_after)
        
        print(f"    Normalized : {norm}")
        print(f"    Prosody    : {prose}")
        print(f"    Audio saved: {new_before} | {new_after}")

    print("\n" + "="*65)
    print(f"DEMO COMPLETE! Aggregate Accuracy: {metrics.get('avg_token_overlap', 0)*100:.1f}%")
    print(f"Check the '{OUTPUT_DIR}/' directory for audio comparisons.")
    print("="*65)

if __name__ == '__main__':
    main()
