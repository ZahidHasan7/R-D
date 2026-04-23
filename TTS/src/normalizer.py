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

class TextNormalizer:
    def __init__(self, use_ml=False):
        self.use_ml = use_ml
        self.ml_translator = MLTranslator()
        self.ner = NERHandler(literal_mode=True)
        self.de = DecisionEngine()
        self.ld = LanguageDetector()
        
        # Order matters for rule-based stage:
        self.rules = [
            normalize_months,
            normalize_ordinals,
            normalize_currency,
            normalize_time,
            normalize_numbers,
            self.ner.handle,  # Handle names/brands from dict
        ]

    def _normalize_tokens(self, text):
        """Processes text through the DecisionEngine for abbreviations and units."""
        tokens = text.split()
        out_tokens = []
        for i, token in enumerate(tokens):
            # Strip trailing punctuation but keep dots for abbreviations
            stripped = token.rstrip(',!?।')
            punct = token[len(stripped):]
            next_token = tokens[i+1] if i+1 < len(tokens) else ''
            
            lang = self.ld.detect_token(stripped)
            processed = self.de.process(stripped, next_token=next_token, lang=lang)
            out_tokens.append(processed + punct)
            
        return ' '.join(out_tokens)

    def unicode_normalize(self, text):
        """Normalizes Bangla characters to NFC for consistency."""
        return unicodedata.normalize('NFC', text)

    def normalize(self, text):
        """Main normalization method with rule-based and optional ML stages."""
        if not text:
            return ""

        # 1. Unicode Normalization
        text = self.unicode_normalize(text)

        # 2. Hybrid token step (Abbreviations, Units, Mixed)
        text = self._normalize_tokens(text)

        # 3. Rule-based expansions (Dates, Currency, Numbers)
        for rule in self.rules:
            text = rule(text)
        
        # 4. Cleanup multi-spaces before ML
        text = re.sub(r'\s+', ' ', text).strip()

        # 5. Optional: Call ML translator for remaining context
        if self.use_ml:
            text = self.ml_translator.translate(text)
            
            # 6. Post-ML Cleanup Pass
            # ML models (like BanglaT5) often revert expansions (e.g., 'ডাক্তার' -> 'ড.')
            text = self._normalize_tokens(text)
            text = normalize_numbers(text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text

if __name__ == "__main__":
    normalizer = TextNormalizer()
    samples = [
        "5kg চাল",
        "12 Aug 2024",
        "Dr. Rahman",
        "১০:৩০ AM এ google এ meeting করবেন।"
    ]
    for sample in samples:
        print(f"Original: {sample} -> Normalized: {normalizer.normalize(sample)}")