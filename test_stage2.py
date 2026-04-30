import sys
import os

# Add TTS root to sys.path
sys.path.append(os.path.join(os.getcwd(), 'TTS'))

from src.normalizer import TextNormalizer

def test_normalization():
    normalizer = TextNormalizer()
    
    test_cases = [
        ("আমি ভাত খাই।", "আমি ভাত খাই."),
        ("তিনি আসবেন এবং আমরা খাব।", "তিনি আসবেন, এবং আমরা খাব."),
        ("৫ কেজি চাল।", "পাঁচ কেজি চাল."),
        ("ডাক্তার রহমান আসবেন কিন্তু দেরি হতে পারে।", "ডাক্তার রহমান আসবেন, কিন্তু দেরি হতে পারে।"), # ডাক্তার রহমান might be abbreviated or not depending on NER
    ]
    
    print("Testing Stage 2: Punctuation Mapping & Conjunction Breaks\n")
    print("-" * 50)
    for original, expected in test_cases:
        normalized = normalizer.normalize(original)
        print(f"Original:   {original}")
        print(f"Normalized: {normalized}")
        
        # Check mapping of । to .
        if '।' in original and '.' in normalized:
            print("[✓] Punctuation Mapping (। -> .) matched.")
        
        # Check conjunction breaks
        for conj in ['এবং', 'কিন্তু', 'তবে', 'যদি', 'কারণ', 'তাই', 'অথবা', 'বরং']:
            if conj in original and f", {conj}" in normalized:
                print(f"[✓] Phrase break found before conjunction '{conj}'.")
        
        print("-" * 50)

if __name__ == "__main__":
    test_normalization()
