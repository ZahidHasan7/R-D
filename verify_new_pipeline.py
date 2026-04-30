import sys
import os

# Add TTS root to sys.path
sys.path.append(os.path.abspath('TTS'))

from src.normalizer import TextNormalizer

def test_pipeline():
    normalizer = TextNormalizer(use_ml=True)
    
    test_cases = [
        ("আমার মোবাইল নম্বর ০১৭১১-২৩৪৫৬৭।", "আমার মোবাইল নম্বর শূন্য এক সাত এক এক দুই তিন চার পাঁচ ছয় সাত."),
        ("আমি ১৯৭১ সালে জন্মগ্রহণ করি।", "আমি উনিশশো একাত্তর সালে জন্মগ্রহণ করি."),
        ("মোট বিল ১২৫০০ টাকা এবং ডেলিভারি ১৫০ টাকা।", "মোট বিল বারো হাজার পাঁচশো টাকা, এবং ডেলিভারি একশো পঞ্চাশ টাকা."),
        ("ট্রেন ছাড়বে রাত ১০:৩০ এ।", "ট্রেন ছাড়বে রাত সাড়ে দশটা এ."),
        ("আজ ১৫ আগস্ট ২০২৪।", "আজ পনেরো আগস্ট দুই হাজার চব্বিশ."),
        ("NID: ৯৮৭৬৫৪৩২১০", "NID: নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য"),
        ("৩.৫ কিমি", "তিন দশমিক পাঁচ কিমি"),
        ("রাত ১০:৩০", "রাত সাড়ে দশটা"),
        ("সকাল ৯:০০", "সকাল নয়টা"),
        ("৩য় শ্রেণি", "তৃতীয় শ্রেণি")
    ]
    
    print("Verifying Upgraded TTS Preprocessing Pipeline\n" + "="*50)
    
    all_passed = True
    for raw, expected in test_cases:
        out = normalizer.normalize(raw)
        status = "[✓] PASS" if out == expected else "[✗] FAIL"
        if out != expected:
            all_passed = False
        print(f"RAW: {raw}")
        print(f"OUT: {out}")
        print(f"EXP: {expected}")
        print(f"STATUS: {status}")
        print("-" * 50)
    
    if all_passed:
        print("\n[SUCCESS] All high-fidelity rules were correctly implemented!")
    else:
        print("\n[WARNING] Some cases failed. Check normalization logic.")

if __name__ == "__main__":
    test_pipeline()
