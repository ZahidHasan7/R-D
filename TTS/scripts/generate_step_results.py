import json
import time
import os
import jiwer
import re
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalizer import TextNormalizer
from src.prosody_formatter import ProsodyFormatter
from src.g2p_engine import G2PEngine

def _compute_cer(references, hypotheses):
    try:
        return jiwer.cer(references, hypotheses)
    except AttributeError:
        total_chars, total_errors = 0, 0
        for ref, hyp in zip(references, hypotheses):
            ref_chars = list(ref.replace(' ', ''))
            hyp_chars = list(hyp.replace(' ', ''))
            total_chars += len(ref_chars)
            total_errors += int(jiwer.wer([ref], [hyp]) * len(ref_chars))
        return total_errors / max(total_chars, 1)

def main():
    test_file = os.path.join('tests', 'cases', 'demo_v2_test_cases.txt')
    truth_file = os.path.join('tests', 'ground_truth', 'demo_v2_ground_truth.txt')
    
    with open(test_file, 'r', encoding='utf-8') as f:
        tests = [x.strip() for x in f if x.strip() and not x.startswith('#')]
    with open(truth_file, 'r', encoding='utf-8') as f:
        truths = [x.strip() for x in f if x.strip() and not x.startswith('#')]
        
    print("Loading Text Normalizer (Rule-based)...")
    norm_rules = TextNormalizer(use_ml=False)
    
    print("Loading Text Normalizer (ML Mode)...")
    norm_ml = TextNormalizer(use_ml=True)
    
    print("Loading Prosody Formatter...")
    prosody_formatter = ProsodyFormatter()
    
    print("Loading G2P Engine...")
    g2p_engine = G2PEngine()
    
    predictions = []
    
    print(f"Evaluating {len(tests)} cases...")
    for i, (raw, truth) in enumerate(zip(tests, truths)):
        print(f"Processing case {i+1}/{len(tests)}...")
        
        # Step 1: Rule Based only Mode
        t0 = time.time()
        step1_out = norm_rules.normalize(raw)
        dt1 = (time.time() - t0) * 1000
        
        # Step 2: Full ML Pipeline Mode
        t0 = time.time()
        step2_out = norm_ml.normalize(raw)
        dt2 = (time.time() - t0) * 1000
        
        # Step 3: Prosody Formatting
        t0 = time.time()
        step3_out = prosody_formatter.format(step2_out)
        dt3 = (time.time() - t0) * 1000
        
        # Step 4: G2P Phonetic Injection
        t0 = time.time()
        # We run G2P on the stripped prosody text to show clean phonemes
        # Or run it on prosody text itself if the G2P handles markers (it handles tokens)
        step4_out = g2p_engine.process_text(step2_out)
        dt4 = (time.time() - t0) * 1000
        
        # Structural Integrity check
        is_structurally_valid = prosody_formatter.strip_markers(step3_out).strip() == step2_out.strip()
        
        predictions.append({
            "id": i + 1,
            "raw_input": raw,
            "expected_ground_truth": truth,
            "step1_rule_based_output": step1_out,
            "step2_ml_translation_output": step2_out,
            "step3_prosody_output": step3_out,
            "step4_g2p_output": step4_out,
            "time_ms_step1": round(dt1, 2),
            "time_ms_step2": round(dt2, 2),
            "time_ms_step3": round(dt3, 2),
            "time_ms_step4": round(dt4, 2),
            "exact_match_step1": step1_out.strip() == truth.strip(),
            "exact_match_step2": step2_out.strip() == truth.strip(),
            "prosody_structural_integrity": is_structurally_valid,
            "pause_count": len(re.findall(r'<pause:\w+>', step3_out))
        })
        
    print("\nGenerating prediction.json...")
    pred_file = os.path.join('results', 'metrics', 'prediction.json')
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
        
    print("Generating result.js...")
    s1_texts = [p["step1_rule_based_output"] for p in predictions]
    s2_texts = [p["step2_ml_translation_output"] for p in predictions]
    s3_texts = [p["step3_prosody_output"] for p in predictions]
    
    metrics = {
        "wer_raw": round(jiwer.wer(truths, tests) * 100, 2),
        "wer_step1_rule_based": round(jiwer.wer(truths, s1_texts) * 100, 2),
        "wer_step2_ml_translation": round(jiwer.wer(truths, s2_texts) * 100, 2),
        "cer_step2": round(_compute_cer(truths, s2_texts) * 100, 2),
        "prosody_coverage": round(sum(1 for p in predictions if p["pause_count"] > 0) / len(tests) * 100, 2),
        "prosody_integrity": round(sum(1 for p in predictions if p["prosody_structural_integrity"]) / len(tests) * 100, 2),
        "avg_pauses": round(sum(p["pause_count"] for p in predictions) / len(tests), 2),
        "total_cases": len(tests),
        "exact_matches_step2": sum(1 for p in predictions if p["exact_match_step2"])
    }
    
    from scripts.measure_g2p_accuracy import evaluate_g2p
    phonetic_truth_file = os.path.join('tests', 'ground_truth', 'phonetic_ground_truth.txt')
    if os.path.exists(phonetic_truth_file):
        with open(phonetic_truth_file, 'r', encoding='utf-8') as f:
            g2p_truths = [x.strip() for x in f if x.strip()]
        g2p_results = evaluate_g2p(predictions, g2p_truths)
        metrics["g2p_per"] = g2p_results["avg_phoneme_error_rate"]
        metrics["g2p_exact_match"] = g2p_results["exact_match_rate"]
    
    result_js_file = os.path.join('results', 'metrics', 'result.js')
    with open(result_js_file, 'w', encoding='utf-8') as f:
        f.write("export const evaluationMetrics = " + json.dumps(metrics, ensure_ascii=False, indent=4) + ";\n\n")
        f.write("export const stepPredictions = " + json.dumps(predictions, ensure_ascii=False, indent=4) + ";\n\n")
        
    print("Done! Files saved.")

if __name__ == '__main__':
    main()
