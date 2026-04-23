import os
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalizer import TextNormalizer
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# CER helper (jiwer >= 3.x exposes jiwer.cer; fall back for older versions)
# ---------------------------------------------------------------------------

def _compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """Computes Character Error Rate between reference and hypothesis lists."""
    try:
        return jiwer.cer(references, hypotheses)
    except AttributeError:
        # jiwer < 3.0 fallback: compute manually at character level
        total_chars, total_errors = 0, 0
        for ref, hyp in zip(references, hypotheses):
            ref_chars = list(ref.replace(' ', ''))
            hyp_chars = list(hyp.replace(' ', ''))
            total_chars += len(ref_chars)
            # Simple edit distance via jiwer word-level on chars
            total_errors += int(jiwer.wer([ref], [hyp]) * len(ref_chars))
        return total_errors / max(total_chars, 1)


# ---------------------------------------------------------------------------
# Pronunciation accuracy proxy
# ---------------------------------------------------------------------------

def _token_overlap(ref: str, hyp: str) -> float:
    """
    Computes token overlap ratio between reference and hypothesis.
    Acts as a pronunciation accuracy proxy (higher = better).
    0.0 = no overlap, 1.0 = perfect match.
    """
    ref_tokens = set(ref.strip().split())
    hyp_tokens = set(hyp.strip().split())
    if not ref_tokens:
        return 1.0
    intersection = ref_tokens & hyp_tokens
    return len(intersection) / len(ref_tokens)


# ---------------------------------------------------------------------------
# MOS simulation stub
# ---------------------------------------------------------------------------

def _mos_simulation_stub(normalized_text: str, ground_truth: str) -> float:
    """
    MOS (Mean Opinion Score) simulation placeholder.

    A real MOS estimation would:
      1. Synthesize audio from normalized_text
      2. Run a neural MOS predictor (e.g. MOSNet, UTMOS) on the wav
      3. Return a score in range [1.0, 5.0]

    This stub returns a proxy score based on token overlap:
    score = 1.0 + 4.0 * token_overlap  (maps 0→1 overlap to 1.0→5.0 MOS)
    """
    overlap = _token_overlap(ground_truth, normalized_text)
    return round(1.0 + 4.0 * overlap, 2)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    test_file: str = os.path.join('tests', 'cases', 'test_cases.txt'),
    truth_file: str = os.path.join('tests', 'ground_truth', 'ground_truth.txt'),
    verbose: bool = True,
) -> dict:
    """
    Runs the full evaluation suite and returns a metrics dict.

    Returns:
        {
            'wer_before': float,  # WER of raw input vs ground truth
            'wer_after':  float,  # WER of normalized output vs ground truth
            'cer_before': float,  # CER of raw input vs ground truth
            'cer_after':  float,  # CER of normalized output vs ground truth
            'wer_improvement_pct': float,
            'cer_improvement_pct': float,
            'avg_token_overlap': float,  # pronunciation accuracy proxy
            'avg_mos_proxy': float,      # simulated MOS score
            'n_cases': int,
        }
    """
    normalizer = TextNormalizer()
    if verbose:
        print("Running evaluation suite...\n")

    with open(test_file, 'r', encoding='utf-8') as f:
        tests  = [x.strip() for x in f if x.strip() and not x.startswith('#')]
    with open(truth_file, 'r', encoding='utf-8') as f:
        truths = [x.strip() for x in f if x.strip() and not x.startswith('#')]

    if len(tests) != len(truths):
        print(f"Error: test_cases ({len(tests)}) and ground_truth ({len(truths)}) "
              f"must have the same number of non-empty lines.")
        return {}

    normalized_texts = []
    per_case_results = []
    total_time_ms = 0.0

    for raw, truth in zip(tests, truths):
        t0   = time.perf_counter()
        norm = normalizer.normalize(raw)
        dt   = (time.perf_counter() - t0) * 1000

        total_time_ms += dt
        normalized_texts.append(norm)

        overlap  = _token_overlap(truth, norm)
        mos_pred = _mos_simulation_stub(norm, truth)
        match    = norm.strip() == truth.strip()

        per_case_results.append({
            'raw': raw, 'truth': truth, 'norm': norm,
            'overlap': overlap, 'mos': mos_pred,
            'match': match, 'time_ms': dt,
        })

    # Aggregate metrics
    wer_before = jiwer.wer(truths, tests)
    wer_after  = jiwer.wer(truths, normalized_texts)
    cer_before = _compute_cer(truths, tests)
    cer_after  = _compute_cer(truths, normalized_texts)

    wer_imp = ((wer_before - wer_after) / max(wer_before, 1e-9)) * 100
    cer_imp = ((cer_before - cer_after) / max(cer_before, 1e-9)) * 100

    avg_overlap = sum(r['overlap'] for r in per_case_results) / len(per_case_results)
    avg_mos     = sum(r['mos']     for r in per_case_results) / len(per_case_results)
    exact_match = sum(1 for r in per_case_results if r['match'])

    if verbose:
        _print_report(
            tests, truths, normalized_texts,
            wer_before, wer_after, wer_imp,
            cer_before, cer_after, cer_imp,
            avg_overlap, avg_mos, exact_match,
            per_case_results, total_time_ms,
        )

    metrics = {
        'wer_before': round(wer_before, 4),
        'wer_after':  round(wer_after,  4),
        'cer_before': round(cer_before, 4),
        'cer_after':  round(cer_after,  4),
        'wer_improvement_pct': round(wer_imp, 2),
        'cer_improvement_pct': round(cer_imp, 2),
        'avg_token_overlap': round(avg_overlap, 4),
        'avg_mos_proxy': round(avg_mos, 2),
        'n_cases': len(tests),
        'exact_matches': exact_match,
        'avg_time_ms': round(total_time_ms / len(tests), 2),
    }

    # Save to file
    output_file = os.path.join('results', 'metrics', 'evaluation_results.json')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"Results saved to: {output_file}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not save results to file: {e}")

    return metrics


def _print_report(
    tests, truths, normalized_texts,
    wer_before, wer_after, wer_imp,
    cer_before, cer_after, cer_imp,
    avg_overlap, avg_mos, exact_match,
    per_case_results, total_time_ms,
):
    n = len(tests)
    print('=' * 62)
    print('       TEXT NORMALIZATION ACCURACY REPORT — v2')
    print('=' * 62)
    print(f"  Total Test Cases        : {n}")
    print(f"  Exact Matches           : {exact_match}/{n}")
    print(f"  Avg Processing Time     : {total_time_ms/n:.1f}ms / text")
    print()
    print(f"  WER  (Before / After)   : {wer_before*100:.1f}%  →  {wer_after*100:.1f}%")
    print(f"  WER  Improvement        : +{wer_imp:.1f}% relative")
    print()
    print(f"  CER  (Before / After)   : {cer_before*100:.1f}%  →  {cer_after*100:.1f}%")
    print(f"  CER  Improvement        : +{cer_imp:.1f}% relative")
    print()
    print(f"  Pronunciation Accuracy  : {avg_overlap*100:.1f}% (token overlap proxy)")
    print(f"  Simulated MOS           : {avg_mos:.2f} / 5.00")
    print()
    print('-' * 62)
    print('  Per-Case Breakdown:')
    print('-' * 62)
    for i, r in enumerate(per_case_results):
        status = '✓' if r['match'] else '✗'
        print(f"\n  [{i+1}] {status}  Input   : {r['raw']}")
        print(f"          Expected : {r['truth']}")
        print(f"          Got      : {r['norm']}")
        print(f"          Overlap={r['overlap']*100:.0f}%  MOS={r['mos']:.1f}  {r['time_ms']:.1f}ms")
    print('=' * 62)


if __name__ == '__main__':
    import sys
    test_f = sys.argv[1] if len(sys.argv) > 1 else 'test_cases_v2.txt'
    truth_f = sys.argv[2] if len(sys.argv) > 2 else 'ground_truth_v2.txt'
    
    # Handle older default fallback if specific v2 files don't exist
    if not os.path.exists(test_f) and test_f == 'test_cases_v2.txt':
        test_f = os.path.join('tests', 'cases', 'demo_v2_test_cases.txt')
    if not os.path.exists(truth_f) and truth_f == 'ground_truth_v2.txt':
        truth_f = os.path.join('tests', 'ground_truth', 'demo_v2_ground_truth.txt')
        
    evaluate(test_f, truth_f)
