"""
language_detector.py — Token-level language detector for code-mixed Bangla-English text.

Detects language at the word level using Unicode ranges, routes tokens
appropriately into the Bangla or English processing pipeline.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

# Unicode range for Bangla script: U+0980–U+09FF
_BANGLA_RE = re.compile(r'[\u0980-\u09FF]')
# ASCII letters (English / romanized)
_ENGLISH_RE = re.compile(r'[a-zA-Z]')
# Pure digit tokens (English or Bangla digits)
_DIGIT_RE = re.compile(r'^[\d০-৯.,]+$')

Lang = Literal['bangla', 'english', 'neutral', 'mixed']


@dataclass
class TokenInfo:
    """Represents a single detected token with its language label."""
    token: str
    lang: Lang
    index: int  # position in the original token list


@dataclass
class RoutingResult:
    """Result of routing a full text through the language detector."""
    original_text: str
    tokens: list[TokenInfo] = field(default_factory=list)
    bangla_tokens: list[str] = field(default_factory=list)
    english_tokens: list[str] = field(default_factory=list)
    neutral_tokens: list[str] = field(default_factory=list)
    is_mixed: bool = False

    @property
    def language_summary(self) -> str:
        """Returns a human-readable summary of detected languages."""
        parts = []
        if self.bangla_tokens:
            parts.append(f"Bangla({len(self.bangla_tokens)})")
        if self.english_tokens:
            parts.append(f"English({len(self.english_tokens)})")
        if self.neutral_tokens:
            parts.append(f"Neutral({len(self.neutral_tokens)})")
        label = '+'.join(parts) if parts else 'Unknown'
        return f"{'Mixed: ' if self.is_mixed else ''}{label}"


class LanguageDetector:
    """
    Token-level language detector for Bangla-English code-mixed text.

    Usage:
        detector = LanguageDetector()
        result = detector.route("আজকে meeting আছে")
        # result.bangla_tokens  → ['আজকে', 'আছে']
        # result.english_tokens → ['meeting']
        # result.is_mixed       → True
    """

    def detect_token(self, token: str) -> Lang:
        """
        Detects the language of a single token.

        Returns:
            'bangla'  — token contains Bangla Unicode script
            'english' — token contains only ASCII letters
            'neutral' — digits, punctuation, or symbols
            'mixed'   — token contains both Bangla and English characters
        """
        clean = token.strip()
        if not clean:
            return 'neutral'

        # Pure digit or punctuation
        if _DIGIT_RE.match(clean):
            return 'neutral'

        has_bangla = bool(_BANGLA_RE.search(clean))
        has_english = bool(_ENGLISH_RE.search(clean))

        if has_bangla and has_english:
            return 'mixed'
        if has_bangla:
            return 'bangla'
        if has_english:
            return 'english'
        return 'neutral'

    def detect_tokens(self, text: str) -> list[TokenInfo]:
        """
        Tokenizes text and detects language for each token.

        Args:
            text: Input string (may be code-mixed)

        Returns:
            List of TokenInfo objects with token, lang, and position.
        """
        # Split on whitespace while preserving punctuation attached to words
        raw_tokens = text.split()
        result = []
        for idx, token in enumerate(raw_tokens):
            lang = self.detect_token(token)
            result.append(TokenInfo(token=token, lang=lang, index=idx))
        return result

    def route(self, text: str) -> RoutingResult:
        """
        Analyzes the full text and routes tokens by language.

        Args:
            text: Input string

        Returns:
            RoutingResult with categorized token lists and mix flag.

        Example:
            route("আজকে meeting আছে")
            → RoutingResult(
                bangla_tokens=['আজকে', 'আছে'],
                english_tokens=['meeting'],
                is_mixed=True
              )
        """
        token_infos = self.detect_tokens(text)
        routing = RoutingResult(original_text=text, tokens=token_infos)

        for ti in token_infos:
            if ti.lang == 'bangla':
                routing.bangla_tokens.append(ti.token)
            elif ti.lang == 'english':
                routing.english_tokens.append(ti.token)
            elif ti.lang == 'mixed':
                # Mixed tokens go to both lists for dual processing
                routing.bangla_tokens.append(ti.token)
                routing.english_tokens.append(ti.token)
                routing.is_mixed = True
            else:
                routing.neutral_tokens.append(ti.token)

        if routing.bangla_tokens and routing.english_tokens:
            routing.is_mixed = True

        return routing

    def reconstruct_with_translations(
        self,
        text: str,
        english_to_bangla: dict[str, str]
    ) -> str:
        """
        Rebuilds the text, replacing detected English tokens with their
        Bangla equivalents from the provided dictionary.

        Args:
            text:                Input code-mixed string
            english_to_bangla:  Mapping of English words → Bangla equivalents

        Returns:
            Reconstructed text with English tokens replaced where mappings exist.

        Example:
            reconstruct_with_translations(
                "আজকে meeting আছে",
                {"meeting": "মিটিং"}
            )
            → "আজকে মিটিং আছে"
        """
        tokens = text.split()
        out = []
        for token in tokens:
            stripped = token.rstrip('.,!?;:।')
            punct = token[len(stripped):]
            lang = self.detect_token(stripped)
            if lang == 'english':
                replacement = english_to_bangla.get(stripped.lower(), stripped)
                out.append(replacement + punct)
            else:
                out.append(token)
        return ' '.join(out)


if __name__ == '__main__':
    detector = LanguageDetector()
    samples = [
        "আজকে meeting আছে",
        "Dr. Rahman ৫টা ক্লাস নেবেন",
        "গাড়িটি ৬০ km/h গতিতে চলছে",
        "Pure English sentence here",
        "সম্পূর্ণ বাংলা বাক্য",
    ]
    for s in samples:
        res = detector.route(s)
        print(f"\nInput:   {s}")
        print(f"Summary: {res.language_summary}")
        print(f"Bangla:  {res.bangla_tokens}")
        print(f"English: {res.english_tokens}")
