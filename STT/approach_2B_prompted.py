"""
approach_2B_prompted.py — Step 2B: Whisper + Domain Prompt (large-v3)
=====================================================================
Uses initial_prompt to guide Whisper's spelling of domain terms.
"""

import csv
import sys
import json
import re
import unicodedata
import datetime
from pathlib import Path

# Load STT tools
sys.path.insert(0, str(Path(__file__).parent))
try:
    import whisper
    import numpy as np
    from jiwer import wer, cer
except ImportError:
    print("ERROR: Missing dependencies. Run: venv/bin/pip install openai-whisper jiwer")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────
DOMAIN_PROMPT = (
    "এটি একটি কল সেন্টার কথোপকথন। এখানে order, refund, bKash, "
    "payment, OTP, delivery, tracking, parcel, warranty, replacement, "
    "cancel, support, customer — এই ধরনের শব্দ থাকতে পারে।"
)

# ── Load and Group Metadata ──────────────────────────────────────
METADATA_PATH = Path("data/metadata.csv")
scenarios = {}

with open(METADATA_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        scen = row["scenario"]
        if scen not in scenarios:
            scenarios[scen] = {"audio": row["audio_path"], "turns": []}
        scenarios[scen]["turns"].append(row["transcript"])

# ── Helper: Clean Text ──────────────────────────────────────────
def clean(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।!?,;:\.\'\"\-—]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Run Whisper Inference ────────────────────────────────────────
print(f"Loading Whisper 'large-v3'...")
model = whisper.load_model("large-v3")

all_refs = []
all_preds = []
results = []

print(f"\nProcessing {len(scenarios)} scenarios with Domain Prompting...")

for name, data in scenarios.items():
    audio_path = Path(data["audio"])
    if not audio_path.exists():
        continue
    
    # reference: join all turns
    ref_text = " ".join(data["turns"])
    ref_clean = clean(ref_text)
    
    # transcribe with prompt
    print(f"  Transcribing {name} with prompt...")
    result = model.transcribe(str(audio_path), language="bn", initial_prompt=DOMAIN_PROMPT)
    pred_text = result.get("text", "")
    pred_clean = clean(pred_text)
    
    all_refs.append(ref_clean)
    all_preds.append(pred_clean)
    
    s_wer = wer(ref_clean, pred_clean) if ref_clean else 1.0
    results.append({
        "scenario": name,
        "wer": round(s_wer, 4),
        "ref": ref_text[:100] + "...",
        "pred": pred_text[:100] + "..."
    })

# ── Global Metrics ──────────────────────────────────────────────
full_ref = " ".join(all_refs)
full_pred = " ".join(all_preds)

global_wer = wer(full_ref, full_pred) if full_ref else 1.0
global_cer = cer(full_ref, full_pred) if full_ref else 1.0

metrics = {
    "wer": round(global_wer, 4),
    "cer": round(global_cer, 4),
    "scenario_count": len(results)
}

# ── Log to experiments ──────────────────────────────────────────
log_dir = Path("experiments/banglish_stt_v1")
log_dir.mkdir(parents=True, exist_ok=True)

log_entry = {
    "ts": datetime.datetime.now().isoformat(),
    "step": "Step2_ASRModel",
    "approach": "2B_WhisperLargeV3_Prompted",
    "metrics": metrics,
    "config": {"model": "large-v3", "prompt": DOMAIN_PROMPT},
    "scenarios": results
}

with open(log_dir / "results.jsonl", "a") as f:
    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ── Report ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("WHISPER + DOMAIN PROMPT RESULTS (2B)")
print("="*60)
print(f"  Global WER      : {metrics['wer']:.4f}  ({metrics['wer']*100:.1f}%)")
print(f"  Global CER      : {metrics['cer']:.4f}  ({metrics['cer']*100:.1f}%)")
print("-"*60)
for r in results:
    print(f"  - {r['scenario']:20s} : WER={r['wer']:.4f}")
print("="*60)
