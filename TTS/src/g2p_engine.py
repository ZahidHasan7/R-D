"""
g2p_engine.py — Hybrid Grapheme-to-Phoneme Engine for Bangla TTS

Pipeline:
  Layer 1: Dictionary lookup        (data/pronunciation_dict.json)
  Layer 2: Rule-based G2P           (Bangla phoneme mapping rules)
  Layer 3: Neural G2P stub          (Future: DeepPhonemizer)
  Layer 4: Context resolver stub    (Future: homograph disambiguation)
"""

import re
import json
import os
import unicodedata
from typing import Optional


# ---------------------------------------------------------------------------
# Layer 3: Neural G2P stub (future-ready interface)
# ---------------------------------------------------------------------------

class NeuralG2P:
    """
    Real implementation of a neural G2P model using transformers.
    Uses 'bangla-ipa/bangla-ipa' or similar sequence-to-sequence model.
    """

    def __init__(self, model_name: str = "teamapocalypseml/regben2ipa-byt5small"):
        self._model_name = model_name
        self._available = False
        self._tokenizer = None
        self._model = None
        self._unknown_log: list[str] = []
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"Loading Neural G2P model '{model_name}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._available = True
            print("Neural G2P loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load Neural G2P model: {e}")

    def predict(self, word: str) -> Optional[str]:
        """
        Attempts neural phoneme prediction.
        """
        if not self._available or not self._model or not self._tokenizer:
            self._unknown_log.append(word)
            return None
        
        try:
            inputs = self._tokenizer(word, return_tensors="pt")
            outputs = self._model.generate(**inputs, max_length=50)
            result = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return result.strip()
        except Exception as e:
            print(f"Neural G2P inference error for word '{word}': {e}")
            return None

    @property
    def unknown_words(self) -> list[str]:
        return list(set(self._unknown_log))


# ---------------------------------------------------------------------------
# Layer 4: Context resolver stub (future-ready interface)
# ---------------------------------------------------------------------------

class ContextResolver:
    """
    Placeholder for homograph disambiguation.
    e.g. "মেলা" can mean 'fair/festival' or 'a lot' depending on context.
    Future: implement with a language model or POS tagger.
    """

    def resolve(self, word: str, context: str = '') -> Optional[str]:
        """
        Returns a pronunciation hint string if the word is a known homograph.
        Returns None if no special handling is needed.
        """
        # Reserved for future implementation
        return None


# ---------------------------------------------------------------------------
# Layer 2: Rule-based Bangla G2P
# ---------------------------------------------------------------------------

# Bangla vowel letters → IPA-lite phoneme labels (for internal representation)
VOWEL_MAP = {
    'অ': 'ɔ', 'আ': 'a', 'ই': 'i', 'ঈ': 'i', 'উ': 'u', 'ঊ': 'u',
    'ঋ': 'ri', 'এ': 'e', 'ঐ': 'oi', 'ও': 'o', 'ঔ': 'ou',
}

# Vowel diacritics (matras) → phoneme labels
MATRA_MAP = {
    'া': 'a', 'ি': 'i', 'ী': 'i', 'ু': 'u', 'ূ': 'u',
    'ৃ': 'ri', 'ে': 'e', 'ৈ': 'oi', 'ো': 'o', 'ৌ': 'ou',
}

# Consonant → phoneme label
CONSONANT_MAP = {
    'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
    'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'ny',
    'ট': 'T', 'ঠ': 'Th', 'ড': 'D', 'ঢ': 'Dh', 'ণ': 'N',
    'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
    'প': 'p', 'ফ': 'ph', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
    'য': 'j', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'Sh',
    'স': 's', 'হ': 'h', 'ড়': 'r', 'ঢ়': 'rh', 'য়': 'y',
    'ৎ': 't',
}

# Hasanta (virama) — suppresses inherent vowel
HASANTA = '\u09CD'
# Anusvara (nasalization dot)
ANUSVARA = '\u0982'
# Visarga
VISARGA = '\u0983'


def _rule_based_g2p(word: str) -> str:
    """
    Converts a single Bangla word to a phoneme string using rule-based mapping.
    Returns a space-separated phoneme sequence.
    """
    phonemes = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]

        # Anusvara (nasalization) → append nasal marker to previous phoneme
        if ch == ANUSVARA:
            if phonemes:
                phonemes[-1] += '~'
            i += 1
            continue

        # Visarga → append 'h'
        if ch == VISARGA:
            phonemes.append('h')
            i += 1
            continue

        # Hasanta → next consonant has no inherent vowel (skip implicit 'ɔ')
        if ch == HASANTA:
            i += 1
            continue

        # Vowel letter (standalone)
        if ch in VOWEL_MAP:
            phonemes.append(VOWEL_MAP[ch])
            i += 1
            continue

        # Matra (vowel diacritic attached to consonant)
        if ch in MATRA_MAP:
            phonemes.append(MATRA_MAP[ch])
            i += 1
            continue

        # Consonant
        if ch in CONSONANT_MAP:
            phonemes.append(CONSONANT_MAP[ch])
            # Add inherent vowel 'ɔ' unless followed by hasanta or another matra
            next_ch = chars[i + 1] if i + 1 < len(chars) else ''
            if next_ch not in (HASANTA,) and next_ch not in MATRA_MAP:
                phonemes.append('ɔ')
            i += 1
            continue

        # Unknown character — pass through as-is
        phonemes.append(ch)
        i += 1

    return ' '.join(phonemes)


# ---------------------------------------------------------------------------
# Layer 1: Dictionary lookup
# ---------------------------------------------------------------------------

_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dictionaries', 'pronunciation_dict.json')


def _load_dict() -> dict:
    """Loads all sub-dictionaries from pronunciation_dict.json into one flat map."""
    flat: dict[str, str] = {}
    try:
        with open(_DICT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for section in data.values():
            if isinstance(section, dict) and not section.get('_comment'):
                flat.update(section)
    except FileNotFoundError:
        pass
    return flat


# ---------------------------------------------------------------------------
# Main G2P Engine
# ---------------------------------------------------------------------------

class G2PEngine:
    """
    Hybrid Grapheme-to-Phoneme engine for Bangla.

    Usage:
        g2p = G2PEngine()
        phonemes = g2p.grapheme_to_phoneme('ঢাকা')
        translated = g2p.translate_english_word('meeting')
    """

    def __init__(self):
        self._dict = _load_dict()
        self._neural = NeuralG2P()
        self._resolver = ContextResolver()

    def grapheme_to_phoneme(self, word: str, context: str = '') -> str:
        """
        Main entry point. Converts a word to phonemes through the 4-layer pipeline.

        Args:
            word:    The word to convert (Bangla or transliterable token)
            context: Surrounding sentence text for context resolution (future use)

        Returns:
            Phoneme string, or the original word if conversion is not possible.
        """
        if not word:
            return ''

        word_lower = word.lower()

        # Layer 4: Context resolution (homograph disambiguation)
        resolved = self._resolver.resolve(word, context)
        if resolved:
            return resolved

        # Layer 1: Dictionary lookup (case-insensitive)
        if word_lower in self._dict:
            return self._dict[word_lower]
        if word in self._dict:
            return self._dict[word]

        # Layer 2: Rule-based G2P (only for Bangla script)
        if self._is_bangla(word):
            return _rule_based_g2p(word)

        # Layer 3: Neural G2P fallback (best for OOV/Complex words)
        if self._neural._available:
            neural_result = self._neural.predict(word)
            if neural_result:
                return neural_result

        # Fallback: return original word
        return word

    def translate_english_word(self, word: str) -> str:
        """
        Looks up an English word in the dictionary to return its Bangla equivalent.
        Returns the original word if not found.
        """
        return self._dict.get(word.lower(), word)

    def process_text(self, text: str) -> str:
        """
        Processes a full text string token by token.
        Bangla tokens go through G2P; English tokens through dictionary lookup.
        Returns text with phoneme representations injected where available.
        """
        tokens = text.split()
        processed = []
        for token in tokens:
            # Strip trailing punctuation for lookup, re-attach after
            stripped = token.rstrip('.,!?;:।')
            punct = token[len(stripped):]
            result = self.grapheme_to_phoneme(stripped, context=text)
            processed.append(result + punct)
        return ' '.join(processed)

    @staticmethod
    def _is_bangla(text: str) -> bool:
        """Returns True if the text contains primarily Bangla Unicode characters."""
        bangla_chars = sum(1 for ch in text if '\u0980' <= ch <= '\u09FF')
        return bangla_chars > len(text) * 0.3

    @property
    def unknown_words(self) -> list[str]:
        """Returns words that fell through to neural G2P (no dict/rule match)."""
        return self._neural.unknown_words
