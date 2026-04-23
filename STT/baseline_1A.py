"""
baseline_1A.py — Approach 1A: Raw Baseline (No Audio)
=====================================================
Since we don't have WAV files yet, this runs in TRANSCRIPT-ONLY mode.
It measures what the WER would be if your ASR model perfectly transcribed.
It also serves as the scaffold for when real audio is added.

The 'prediction' here = raw normalized text from the dataset (column 2 = digits/mixed)
The 'reference' = the properly normalized text (column 3)

This gives us a meaningful baseline: "How much does number normalization alone affect WER?"
"""

import csv
import sys
import json
import re
import unicodedata
import datetime
from pathlib import Path

# ── Add STT folder to path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

try:
    from jiwer import wer, cer
except ImportError:
    print("ERROR: jiwer not found. Run: pip install jiwer")
    sys.exit(1)

# ── Load data directly from vits2 metadata ────────────────────────
SRC = Path("../project/data/vits2_dataset/metadata.csv")

refs  = []  # Properly normalized transcripts (ground truth)
preds = []  # Raw transcripts (simulating what a bad ASR might output)

with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\r\n")
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        raw        = parts[1].strip()  # digits like ৪৫৮৯২
        normalized = parts[2].strip()  # words like পঁয়তাল্লিশ হাজার...
        refs.append(normalized)
        preds.append(raw)

# ── Clean text ────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।!?,;:\.\'\"\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

refs_c  = [clean(r) for r in refs]
preds_c = [clean(p) for p in preds]

# ── Compute metrics ───────────────────────────────────────────────
ref_joined  = " ".join(refs_c)
pred_joined = " ".join(preds_c)

word_err  = wer(ref_joined, pred_joined)
char_err  = cer(ref_joined, pred_joined)
sent_acc  = sum(r == p for r, p in zip(refs_c, preds_c)) / len(refs_c)

# English token WER
en_refs  = [" ".join(re.findall(r"[a-z]+", r)) for r in refs_c]
en_preds = [" ".join(re.findall(r"[a-z]+", p)) for p in preds_c]
pairs    = [(r, p) for r, p in zip(en_refs, en_preds) if r]
en_wer   = wer(" ".join(p[0] for p in pairs), " ".join(p[1] for p in pairs)) if pairs else 0.0

# Token accuracy
total, correct = 0, 0
for r, p in zip(refs_c, preds_c):
    r_toks = r.split(); p_toks = p.split()
    total   += len(r_toks)
    correct += sum(a == b for a, b in zip(r_toks, p_toks[:len(r_toks)]))
tok_acc = correct / total if total else 0.0

metrics = {
    "wer":          round(word_err, 4),
    "cer":          round(char_err, 4),
    "sentence_acc": round(sent_acc, 4),
    "english_wer":  round(en_wer, 4),
    "token_acc":    round(tok_acc, 4),
}

# ── Log to results.jsonl ──────────────────────────────────────────
log_dir = Path("experiments/banglish_stt_v1")
log_dir.mkdir(parents=True, exist_ok=True)

entry = {
    "ts":       datetime.datetime.now().isoformat(),
    "step":     "Step1_AudioPreprocessing",
    "approach": "1A_RawBaseline",
    "metrics":  metrics,
    "config":   {"mode": "transcript_only", "samples": len(refs)},
}
with open(log_dir / "results.jsonl", "a") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── Print results ─────────────────────────────────────────────────
print("\n" + "="*60)
print("APPROACH 1A — Raw Baseline (Transcript-Only Mode)")
print("="*60)
print(f"  Samples         : {len(refs)}")
print(f"  WER             : {metrics['wer']:.4f}  ({metrics['wer']*100:.1f}%)")
print(f"  CER             : {metrics['cer']:.4f}  ({metrics['cer']*100:.1f}%)")
print(f"  Sentence Acc    : {metrics['sentence_acc']:.4f}  ({metrics['sentence_acc']*100:.1f}%)")
print(f"  English WER     : {metrics['english_wer']:.4f}  ({metrics['english_wer']*100:.1f}%)")
print(f"  Token Accuracy  : {metrics['token_acc']:.4f}  ({metrics['token_acc']*100:.1f}%)")
print("="*60)
print("\n[✓] Results logged to experiments/banglish_stt_v1/results.jsonl")
print("\nInterpretation:")
print("  This is the FLOOR — raw digits vs. normalized text.")
print("  Every later approach should produce a LOWER WER than this.")
