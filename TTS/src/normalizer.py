import re
import unicodedata
from src.rules.numbers import normalize_numbers
from src.rules.months import normalize_months
from src.rules.currency import normalize_currency
from src.rules.time import normalize_time
from src.rules.ordinals import normalize_ordinals
from src.ml_translator import MLTranslator
from src.ner_handler import NERHandler
from src.services.decision_engine import DecisionEngine
from src.language_detector import LanguageDetector

BANGLA_CONJUNCTIONS = [
    'এবং', 'কিন্তু', 'তবে', 'যদি', 'কারণ', 'তাই', 'অথবা', 
    'বরং', 'যদিও', 'তবুও', 'অতএব', 'ফলে', 'তখন'
]

class TextNormalizer:
    def __init__(self, use_ml=False):
        self.use_ml = use_ml
        self.ml_translator = MLTranslator()
        self.ner = NERHandler(literal_mode=True)
        self.de = DecisionEngine()
        self.ld = LanguageDetector()
        
        # Rule set excludes normalize_numbers because we call it explicitly to avoid double expansion
        self.rules = [
            normalize_months,
            normalize_ordinals,
            normalize_currency,
            normalize_time,
            self.ner.handle,
        ]

    def _normalize_tokens(self, text):
        """Processes text through the DecisionEngine for abbreviations and units."""
        tokens = text.split()
        out_tokens = []
        for i, token in enumerate(tokens):
            stripped = token.rstrip(',!?।')
            punct = token[len(stripped):]
            next_token = tokens[i+1] if i+1 < len(tokens) else ''
            
            lang = self.ld.detect_token(stripped)
            processed = self.de.process(stripped, next_token=next_token, lang=lang)
            out_tokens.append(processed + punct)
            
        return ' '.join(out_tokens)

    def unicode_normalize(self, text):
        return unicodedata.normalize('NFC', text)

    def normalize_punctuation_for_vits(self, text: str) -> str:
        replacements = {
            '।': '.',
            '॥': '...',
            '—': ',',
            '–': ',',
            '\u200c': '',
            '\u200d': '',
            '৷': '.',
        }
        for src, tgt in replacements.items():
            text = text.replace(src, tgt)
        return text

    def insert_phrase_breaks(self, text: str) -> str:
        for conj in BANGLA_CONJUNCTIONS:
            pattern = rf'(?<![,.!?;।]) {conj}'
            text = re.sub(pattern, f', {conj}', text)
        return text

    def normalize(self, text):
        if not text:
            return ""

        text = self.unicode_normalize(text)
        
        # Abbreviations and Units (Hybrid Step)
        text = self._normalize_tokens(text)

        # Rule-based logic
        for rule in self.rules:
            text = rule(text)
            
        # Number expansion (Call once here)
        text = normalize_numbers(text)

        # If using ML, run translator then do a FINAL cleanup ONLY if needed
        if self.use_ml:
            text = self.ml_translator.translate(text)
            # Cleanup double spaces from ML
            text = re.sub(r'\s+', ' ', text).strip()
            
        # Punctuation Normalization
        text = self.normalize_punctuation_for_vits(text) 
        
        # Phrase Break Insertion
        text = self.insert_phrase_breaks(text)
        
        # Final Cleanup
        text = re.sub(r'\s+', ' ', text).strip()
            
        return text

if __name__ == "__main__":
    normalizer = TextNormalizer()
    samples = {
        "আমার নম্বর ০১৭১১-২৩৪৫৬৭।": "আমার নম্বর শূন্য এক সাত এক এক দুই তিন চার পাঁচ ছয় সাত.",
        "আমি ১৯৭১ সালে জন্মগ্রহণ করি।": "আমি উনিশশো একাত্তর সালে জন্মগ্রহণ করি.",
    }
    for raw, expected in samples.items():
        out = normalizer.normalize(raw)
        print(f"RAW: {raw}\nOUT: {out}\nEXPECTED: {expected}\n---")