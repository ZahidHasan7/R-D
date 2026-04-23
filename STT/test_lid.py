import sys
import os
from pathlib import Path

# Add project root to sys.path to allow absolute imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import fasttext
except ImportError:
    print("❌ Error: fasttext not found. Please run: source STT/venv/bin/activate && pip install fasttext-wheel")
    sys.exit(1)

from TTS.src.language_detector import LanguageDetector

# Constants
MODEL_PATH = SCRIPT_DIR / "models" / "lid.176.bin"

def main():
    # Initialize components
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model and lid.176.bin not found at {MODEL_PATH}")
        return

    ft_model = fasttext.load_model(str(MODEL_PATH))
    detector = LanguageDetector()

    # Input handling
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("▶ No input provided. Using example: 'হ্যালো স্যার আমি কীভাবে help করতে পারি? আপু আমার order টা confirm করেন'")
        text = "হ্যালো স্যার আমি কীভাবে help করতে পারি? আপু আমার order টা confirm করেন"

    # Token-level logic
    tokens = text.split()
    
    print(f"\n{'='*75}")
    print(f"{'Token':<18} | {'4A: Rule-Based (Unicode)':<25} | {'4B: fastText (Neural)':<20}")
    print(f"{'='*75}")

    for token in tokens:
        # 4A - Rule-Based (from LanguageDetector)
        label_4a = detector.detect_token(token)
        
        # 4B - fastText
        # Note: we predict on the raw token, but often punctuation can confuse LID
        clean_token = token.strip("।,?!:;\"'") or token
        pred = ft_model.predict(clean_token, k=1)
        lang_code = pred[0][0].replace("__label__", "")
        
        # Map some common codes for readability
        label_map = {"bn": "bangla", "en": "english", "hi": "hindi", "ar": "arabic"}
        label_4b = label_map.get(lang_code, lang_code)

        # Highlight differences
        marker = " "
        if label_4a != label_4b and label_4a in ["bangla", "english"] and label_4b in ["bangla", "english"]:
            marker = "*"

        print(f"{token:<18} | {label_4a:<25} | {label_4b:<20} {marker}")

    print(f"{'='*75}")
    print("* indicates a disagreement between approaches.")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    main()
