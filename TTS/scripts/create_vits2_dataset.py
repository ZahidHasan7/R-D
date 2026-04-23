import os
import csv
import argparse
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import local modules for normalization and graphing
try:
    from src.normalizer import TextNormalizer
    from src.g2p_engine import G2PEngine
except ImportError:
    print("Warning: normalizer.py or g2p_engine.py not found. Please ensure they are in the PYTHONPATH.")
    TextNormalizer = None
    G2PEngine = None

def create_ljspeech_dataset(metadata_csv_path, output_meta_path):
    """
    Reads the dataset metadata.csv, normalizes the text, extracts phonemes,
    and outputs vits2_metadata.csv in LJSpeech format:
    file_id|transcription|normalized_transcription|phonemes
    """
    if not TextNormalizer or not G2PEngine:
        print("Cannot process without normalizer.py and g2p_engine.py.")
        return

    print("Initializing Normalizer & G2P models...")
    normalizer = TextNormalizer()
    g2p = G2PEngine()
    
    print(f"Reading source metadata from: {metadata_csv_path}")
    print(f"Creating LJSpeech formatted metadata at: {output_meta_path}")
    
    rows = []
    
    with open(metadata_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            # The input has headers: filename|text
            file_name = row.get('filename', '')
            file_id = file_name.replace('.wav', '')
            raw_text = row.get('text', '')
            
            if not file_id or not raw_text:
                continue
                
            try:
                # 1. Normalize
                norm_text = normalizer.normalize(raw_text)
                
                # 2. Extract Phonemes
                # Ensure normalizer output is a string before passing to g2p
                norm_text_str = str(norm_text)
                phonemes = g2p.process_text(norm_text_str)
                
                rows.append({
                    "file_id": file_name, # Preserving '.wav' in VITS config might be easier for data loader
                    "raw_text": raw_text,
                    "norm_text": norm_text_str,
                    "phonemes": phonemes
                })
            except Exception as e:
                print(f"Skip {file_id}: Error processing text. {e}")

    # If an audio_dir was provided via a module-level variable, we can optionally warn
    # if referenced files are missing. This variable will be set by the CLI wrapper.
    audio_dir = globals().get("_AUDIO_DIR", None)
    if audio_dir:
        missing = []
        for r in rows:
            wav_name = r["file_id"]
            wav_path = os.path.join(audio_dir, wav_name)
            if not os.path.exists(wav_path):
                missing.append(wav_name)
        if missing:
            print(f"Warning: {len(missing)} referenced audio files not found in {audio_dir}. Example missing: {missing[:5]}")

    # Write Metadata
    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
    with open(output_meta_path, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        for r in rows:
            # LJSpeech commonly wants the pure base filename as ID:
            base_id = r["file_id"].replace(".wav", "")
            writer.writerow([base_id, r["raw_text"], r["norm_text"], r["phonemes"]])
            
    print(f"Dataset generation complete. Successfully processed {len(rows)} audio mappings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create VITS2 Dataset Metadata.")
    default_metadata = os.path.join("data", "datasets", "metadata.csv")
    default_output = os.path.join("data", "datasets", "vits2_dataset", "metadata.csv")
    parser.add_argument("--metadata_csv", type=str, default=default_metadata, help="Input metadata.csv path.")
    parser.add_argument("--output_meta", type=str, default=default_output, help="Output metadata.csv path.")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory that contains .wav audio files referenced by metadata.")
    args = parser.parse_args()

    # Expose audio_dir to the function via a module-level variable so existing call sites
    # or imports don't need to change function signature.
    if args.audio_dir:
        globals()["_AUDIO_DIR"] = args.audio_dir

    create_ljspeech_dataset(args.metadata_csv, args.output_meta)
