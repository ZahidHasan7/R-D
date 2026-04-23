"""
build_ground_truth.py — Merge all per-scenario CSVs into a single metadata file
and link them to their corresponding .m4a audio files.
"""
import csv
from pathlib import Path

DATA_DIR = Path("data")

# Map CSV filename stem → audio filename (case-sensitive on disk)
AUDIO_MAP = {
    "delivery_delay":     "Delivery_delay.m4a",
    "delivery_delay_2":   "Delivery_delay_2.m4a",
    "payment_issue":      "Payment_Issue.m4a",
    "payment_issue_2":    "Payment_issue_2.m4a",
    "product_inquiry":    "Product_inquiry.m4a",
    "product_inquiry_2":  "Product_inquiry_2.m4a",
    "refund_request_2":   "Refund_Request_2.m4a",
    "wrong_item":         "Wrong_Item.m4a",
    "wrong_item_2":       "Wrong_item_2.m4a",
}

# Merge all per-scenario CSVs into one
all_rows = []
for stem, audio_file in AUDIO_MAP.items():
    csv_path = DATA_DIR / f"{stem}.csv"
    if not csv_path.exists():
        print(f"  [WARN] Missing CSV: {csv_path}")
        continue
    audio_path = DATA_DIR / audio_file
    with open(csv_path, encoding="utf-8") as f:
        # Read all lines, strip them, and skip truly empty ones
        lines = [l.strip() for l in f if l.strip()]
    
    # Use fixed fieldnames as all user files follow the same structure
    reader = csv.DictReader(lines, fieldnames=["id", "scenario", "speaker_role", "utterance"])
    rows = []
    for r in reader:
        # Skip the header row if it exists in the file
        if r.get("id") == "id" or not r.get("utterance"):
            continue
        rows.append(r)
    for row in rows:
        all_rows.append({
            "id":           row["id"].strip(),
            "audio_path":   str(audio_path),   # Full m4a path — shared per scenario
            "transcript":   row["utterance"].strip(),
            "scenario":     row.get("scenario", stem).strip(),
            "speaker_role": row.get("speaker_role", "").strip(),
        })
    print(f"  Loaded {len(rows):2d} rows from {stem}.csv → {audio_file}")

# Write master metadata
out_path = DATA_DIR / "metadata.csv"
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "audio_path", "transcript", "scenario", "speaker_role"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\n✓ Wrote {len(all_rows)} rows → {out_path}")
