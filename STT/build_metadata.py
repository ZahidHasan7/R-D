"""
build_metadata.py — Convert vits2_dataset/metadata.csv to STT metadata format
Run once: python3 build_metadata.py
"""
import csv, sys
from pathlib import Path

SRC  = Path("../project/data/vits2_dataset/metadata.csv")
DEST = Path("data/metadata.csv")

rows = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\r\n")
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        scenario = parts[0].strip()
        raw_transcript = parts[1].strip()   # Raw (contains digits)
        normalized_transcript = parts[2].strip() if len(parts) > 2 else raw_transcript
        rows.append({
            "id":           scenario,
            "audio_path":   "",   # No real audio yet — transcript-only mode
            "transcript":   normalized_transcript,
            "scenario":     scenario,
            "speaker_role": "mixed",
        })

with open(DEST, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "audio_path", "transcript", "scenario", "speaker_role"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {DEST}")
