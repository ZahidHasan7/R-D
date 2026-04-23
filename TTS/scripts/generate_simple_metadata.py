import os
import csv

# Paths (adjust if needed)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
INPUT_META = os.path.join(PROJECT_ROOT, 'metadata', 'metadata.csv')
PROCESSED_AUDIO_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'audiodata', 'processed'))
OUTPUT_META = os.path.join(PROJECT_ROOT, 'data', 'vits2_dataset', 'metadata.csv')

os.makedirs(os.path.dirname(OUTPUT_META), exist_ok=True)

available = {os.path.splitext(f)[0]: os.path.join(PROCESSED_AUDIO_DIR, f)
             for f in os.listdir(PROCESSED_AUDIO_DIR) if f.lower().endswith('.wav')}

written = 0
skipped = 0

with open(INPUT_META, encoding='utf-8') as inf, open(OUTPUT_META, 'w', encoding='utf-8', newline='') as outf:
    reader = csv.DictReader(inf, delimiter='|')
    writer = csv.writer(outf, delimiter='|')
    for row in reader:
        filename = row.get('filename','').strip()
        text = row.get('text','').strip()
        if not filename or not text:
            skipped += 1
            continue
        base = os.path.splitext(filename)[0]
        # prefer processed file matching base
        if base in available:
            # write base (no extension) | raw text | normalized_text (noop) | phonemes (empty)
            writer.writerow([base, text, text, ''])
            written += 1
        else:
            skipped += 1

print(f"Wrote {written} metadata entries to {OUTPUT_META}; skipped {skipped} entries (missing audio)")