import os
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
META_IN = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'vits2_dataset', 'metadata.csv')
OUT_WAV_DIR = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'vits2_dataset', 'wavs')
META_OUT = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'vits2_dataset', 'metadata_final.csv')
PROCESSED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'audiodata', 'processed'))

os.makedirs(OUT_WAV_DIR, exist_ok=True)

written = 0
with open(META_IN, encoding='utf-8') as inf, open(META_OUT, 'w', encoding='utf-8', newline='') as outf:
    reader = csv.reader(inf, delimiter='|')
    writer = csv.writer(outf, delimiter='|')
    for row in reader:
        if not row:
            continue
        file_id = row[0].strip()
        text = row[1].strip() if len(row) > 1 else ''
        # find processed file matching base
        candidates = [f for f in os.listdir(PROCESSED_DIR) if os.path.splitext(f)[0].lower() == file_id.lower()]
        if not candidates:
            # try case-insensitive substring match
            candidates = [f for f in os.listdir(PROCESSED_DIR) if file_id.lower() in os.path.splitext(f)[0].lower()]
        if not candidates:
            continue
        src = os.path.join(PROCESSED_DIR, candidates[0])
        dst = os.path.join(OUT_WAV_DIR, os.path.basename(src))
        # create symlink if not exists
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
        writer.writerow([os.path.basename(src), text])
        written += 1

print(f"Assembled {written} entries into {OUT_WAV_DIR} and wrote metadata to {META_OUT}")