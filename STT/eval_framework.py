import json, time, datetime, csv, re, unicodedata
from pathlib import Path
from jiwer import wer, cer
import numpy as np

class STTEvaluator:
    # Bengali-script spellings frequently used for English terms in code-mixed speech.
    # We normalize these to English for EN-WER so the metric captures intended term accuracy.
    CODEMIX_EN_EQUIV = {
        "অর্ডার": "order",
        "অডার": "order",
        "ওডার": "order",
        "পেমেন্ট": "payment",
        "রিফান্ড": "refund",
        "রিফান্ডের": "refund",
        "ডেলিভারি": "delivery",
        "ট্র্যাকিং": "tracking",
        "আইডি": "id",
        "টিকেট": "ticket",
        "সাপোর্ট": "support",
        "অ্যাপ": "app",
        "ওয়েবসাইট": "website",
        "ওয়েবসাইট": "website",
        "ইমেইল": "email",
        "এসএমএস": "sms",
        "ওটিপি": "otp",
        "বিকাশ": "bkash",
        "নগদ": "nagad",
    }

    def __init__(self, experiment_name: str):
        self.name    = experiment_name
        self.log_dir = Path(f"experiments/{experiment_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self._load_existing_results()

    def _load_existing_results(self):
        log_file = self.log_dir / "results.jsonl"
        if log_file.exists():
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        self.results.append(json.loads(line))
                    except Exception:
                        continue

    # ── Core metrics ─────────────────────────────────────────────
    def compute_metrics(self, refs: list[str],
                        preds: list[str]) -> dict:
        if not refs:
            return {"wer": 1.0, "cer": 1.0, "sentence_acc": 0.0, "english_wer": 0.0, "token_acc": 0.0}

        refs_c  = [self._clean(r) for r in refs]
        preds_c = [self._clean(p) for p in preds]

        # Avoid joining empty lists which breaks wer
        ref_joined = " ".join(refs_c) if any(refs_c) else " "
        hyp_joined = " ".join(preds_c) if any(preds_c) else " "

        word_err  = wer(ref_joined, hyp_joined)
        char_err  = cer(ref_joined, hyp_joined)
        sent_acc  = sum(r == p for r, p in
                        zip(refs_c, preds_c)) / len(refs_c)
        en_wer    = self._english_token_wer(refs_c, preds_c)
        tok_acc   = self._token_accuracy(refs_c, preds_c)

        return {
            "wer":           round(word_err, 4),
            "cer":           round(char_err, 4),
            "sentence_acc":  round(sent_acc, 4),
            "english_wer":   round(en_wer,  4),
            "token_acc":     round(tok_acc,  4),
        }

    def _clean(self, text: str) -> str:
        if not text: return ""
        text = text.lower().strip()
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[।!?,;:\.\'\"\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _english_token_wer(self, refs: list[str],
                            preds: list[str]) -> float:
        en_refs  = [self._extract_english_tokens(r) for r in refs]
        en_preds = [self._extract_english_tokens(p) for p in preds]
        pairs = [(r, p) for r, p in zip(en_refs, en_preds) if r]
        if not pairs: return 0.0
        r_str = " ".join(p[0] for p in pairs)
        p_str = " ".join(p[1] for p in pairs)
        return wer(r_str, p_str) if r_str.strip() else 0.0

    def _extract_english_tokens(self, text: str) -> str:
        if not text:
            return ""
        t = unicodedata.normalize("NFC", text.lower())
        # Normalize common Bengali-script forms of English terms before extraction.
        for bn_token, en_token in self.CODEMIX_EN_EQUIV.items():
            t = re.sub(rf"(?<!\w){re.escape(bn_token)}(?!\w)", en_token, t)
        return " ".join(re.findall(r"[a-z]+", t))

    def _token_accuracy(self, refs: list[str],
                         preds: list[str]) -> float:
        total, correct = 0, 0
        for r, p in zip(refs, preds):
            r_toks, p_toks = r.split(), p.split()
            total   += len(r_toks)
            correct += sum(a == b for a, b in
                           zip(r_toks, p_toks[:len(r_toks)]))
        return correct / total if total else 0.0

    # ── Logging ──────────────────────────────────────────────────
    def log(self, step: str, approach: str,
            metrics: dict, config: dict = None):
        entry = {
            "ts":       datetime.datetime.now().isoformat(),
            "step":     step,
            "approach": approach,
            "metrics":  metrics,
            "config":   config or {},
        }
        self.results.append(entry)
        with open(self.log_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[{step}] {approach}: WER={metrics.get('wer','?'):.4f}  "
              f"CER={metrics.get('cer','?'):.4f}  "
              f"EN-WER={metrics.get('english_wer','?'):.4f}")

    def compare(self, step: str):
        rows = [r for r in self.results if r["step"] == step]
        ranked = sorted(rows, key=lambda x: x["metrics"].get("wer", 1))
        print(f"\n{'='*60}")
        print(f"RANKING — {step}")
        print(f"{'='*60}")
        for i, r in enumerate(ranked):
            print(f"  {i+1}. {r['approach']:40s} "
                  f"WER={r['metrics'].get('wer','?'):.4f}")
        return ranked[0] if ranked else None

# ── Build ground-truth test set from metadata ────────────────────
def build_test_set(csv_path: str) -> list[dict]:
    samples = []
    if not Path(csv_path).exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append({
                "id":       row.get("id", ""),
                "audio":    row.get("audio_path", ""),
                "ref":      row.get("transcript", ""),
                "scenario": row.get("scenario", "general"),
                "speaker":  row.get("speaker_role", ""),
            })
    return samples

if __name__ == "__main__":
    evaluator = STTEvaluator("banglish_stt_v1")
    # This will be used in the main runner script later
    print(f"STT Evaluator '{evaluator.name}' initialized.")
