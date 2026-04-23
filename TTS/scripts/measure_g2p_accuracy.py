import json
import jiwer
import os

def phoneme_error_rate(ref: str, hyp: str) -> float:
    """PER = edit distance between phoneme sequences / ref length."""
    # Split by spaces to treat each phoneme as a 'word' for jiwer.wer
    ref_ph = ref.strip().split()
    hyp_ph = hyp.strip().split()
    if not ref_ph:
        return 1.0 if hyp_ph else 0.0
    return jiwer.wer(' '.join(ref_ph), ' '.join(hyp_ph))

def evaluate_g2p(predictions: list[dict], ground_truth_phonemes: list[str]) -> dict:
    pers = []
    exact_matches = 0
    errors = []

    for i, (pred_data, ref) in enumerate(zip(predictions, ground_truth_phonemes)):
        hyp = pred_data['step4_g2p_output']
        per = phoneme_error_rate(ref, hyp)
        pers.append(per)
        
        if hyp.strip() == ref.strip():
            exact_matches += 1
            
        if per > 0.1:
            errors.append({
                "id": pred_data['id'],
                "input": pred_data['raw_input'],
                "pred": hyp,
                "ref": ref,
                "per": round(per, 4)
            })

    avg_per = sum(pers) / len(pers) if pers else 0.0
    return {
        "avg_phoneme_error_rate": round(avg_per * 100, 2),
        "exact_match_rate": round((exact_matches / len(predictions)) * 100, 2),
        "error_count": len(errors),
        "detailed_errors": errors
    }

if __name__ == "__main__":
    # Example usage / test
    import json
    pred_file = os.path.join('results', 'metrics', 'prediction.json')
    truth_file = os.path.join('tests', 'ground_truth', 'phonetic_ground_truth.txt')
    
    if not os.path.exists(pred_file):
        print(f"Error: {pred_file} not found.")
        return

    with open(pred_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    with open(truth_file, 'r', encoding='utf-8') as f:
        truths = [x.strip() for x in f if x.strip()]
        
    results = evaluate_g2p(preds, truths)
    print(json.dumps(results, indent=4))
