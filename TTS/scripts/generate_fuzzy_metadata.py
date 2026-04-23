import os
import csv
import re
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
INPUT_META = os.path.join(PROJECT_ROOT, 'metadata', 'metadata.csv')
PROCESSED_AUDIO_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'audiodata', 'processed'))
OUTPUT_META = os.path.join(PROJECT_ROOT, 'data', 'vits2_dataset', 'metadata.csv')

os.makedirs(os.path.dirname(OUTPUT_META), exist_ok=True)

def normalize_name(name):
    # lower, remove extension, strip common suffixes like _1,_2,-1 etc, remove non-alnum
    base = os.path.splitext(os.path.basename(name))[0].lower()
    base = re.sub(r"[_\- ]+(copy|backup)$", '', base)
    base = re.sub(r"[_\- ]+v?\d+$", '', base)
    base = re.sub(r"[_\- ]+\d+$", '', base)
    base = re.sub(r"[^a-z0-9]+", '', base)
    return base

# gather processed files
processed = {}
for f in os.listdir(PROCESSED_AUDIO_DIR):
    if not f.lower().endswith('.wav'):
        continue
    key = normalize_name(f)
    if key in processed:
        # Keep first, but record collisions
        processed[key + '_dup' + str(len(processed))] = os.path.join(PROCESSED_AUDIO_DIR, f)
    else:
        processed[key] = os.path.join(PROCESSED_AUDIO_DIR, f)

# Prepare for fuzzy match: also build prefix index
prefix_index = defaultdict(list)
for k in list(processed.keys()):
    for i in range(3, min(len(k), 40)):
        prefix_index[k[:i]].append(k)

written = 0
skipped = 0
matches = []

with open(INPUT_META, encoding='utf-8') as inf, open(OUTPUT_META, 'w', encoding='utf-8', newline='') as outf:
    reader = csv.DictReader(inf, delimiter='|')
    writer = csv.writer(outf, delimiter='|')
    for row in reader:
        filename = row.get('filename','').strip()
        text = row.get('text','').strip()
        if not filename or not text:
            skipped += 1
            continue
        norm = normalize_name(filename)
        chosen = None
        # exact key
        if norm in processed:
            chosen = processed[norm]
        else:
            # try prefix matching
            for L in range(min(len(norm),40), 2, -1):
                prefix = norm[:L]
                cand_keys = prefix_index.get(prefix, [])
                if cand_keys:
                    chosen = processed[cand_keys[0]]
                    break
            # try substring match
            if not chosen:
                for k in processed:
                    if norm in k or k in norm:
                        chosen = processed[k]
                        break
        if chosen:
            base = os.path.splitext(os.path.basename(chosen))[0]
            writer.writerow([base, text, text, ''])
            written += 1
            matches.append((filename, base))
        else:
            skipped += 1

print(f"Wrote {written} entries to {OUTPUT_META}; skipped {skipped} entries.")
if matches:
    print('Sample matches:')
    for a,b in matches[:10]:
        print(f'  {a} -> {b}')
else:
    print('No matches found')
