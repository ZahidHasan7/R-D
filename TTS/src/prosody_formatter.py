"""
prosody_formatter.py — Prosody-aware text formatter for Bangla TTS.

Inserts pause markers, detects sentence type (question/statement/exclamation),
and adds emphasis markers to guide TTS prosody synthesis.

Output format uses SSML-lite markers compatible with most TTS backends.
"""

import re
from enum import Enum, auto
from dataclasses import dataclass


class SentenceType(Enum):
    STATEMENT = auto()
    QUESTION = auto()
    EXCLAMATION = auto()
    MIXED = auto()


@dataclass
class ProsodyAnnotation:
    """Holds the fully annotated text and its detected metadata."""
    raw_text: str
    formatted_text: str
    sentence_type: SentenceType
    pause_count: int
    has_emphasis: bool


# Bangla question words that signal interrogative sentences
_BANGLA_QUESTION_WORDS = re.compile(
    r'\b(কি|কী|কেন|কোথায়|কখন|কীভাবে|কে|কার|কাকে|কতটা|কতক্ষণ|কোন)\b'
)

# Punctuation → pause marker mapping
_PAUSE_MAP = [
    (re.compile(r'([।])'),  r'\1 <pause:long> '),
    (re.compile(r'([.])(?=\s|$)'), r'\1 <pause:long> '),
    (re.compile(r'([,،])'),  r'\1 <pause:short> '),
    (re.compile(r'([;])'),   r'\1 <pause:medium> '),
    (re.compile(r'([?])'),   r'\1 <pause:long> '),
    (re.compile(r'([!])'),   r'\1 <pause:long> '),
    (re.compile(r'(-{1,2})'), r' <pause:short> '),  # dashes as a short break
]

# Sentence-final markers
_QUESTION_MARKERS = re.compile(r'[?？]|কি\s*[?।]|কী\s*[?।]')
_EXCLAIM_MARKERS  = re.compile(r'[!！]')


class ProsodyFormatter:
    """
    Inserts prosody markers into Bangla text for TTS engines.

    Usage:
        pf = ProsodyFormatter()
        ann = pf.annotate("আপনাকে ধন্যবাদ। আমি কীভাবে সাহায্য করতে পারি?")
        print(ann.formatted_text)
        # → "আপনাকে ধন্যবাদ। <pause:long> আমি কীভাবে সাহায্য করতে পারি? <pause:long>"
        print(ann.sentence_type)
        # → SentenceType.QUESTION
    """

    def __init__(
        self,
        long_pause_marker: str = '<pause:long>',
        short_pause_marker: str = '<pause:short>',
        medium_pause_marker: str = '<pause:medium>',
        emphasis_open: str = '<emphasis>',
        emphasis_close: str = '</emphasis>',
    ):
        self.long_pause   = long_pause_marker
        self.short_pause  = short_pause_marker
        self.medium_pause = medium_pause_marker
        self.emphasis_open  = emphasis_open
        self.emphasis_close = emphasis_close

    def detect_sentence_type(self, text: str) -> SentenceType:
        """
        Detects whether the sentence is a question, exclamation, or statement.

        Handles:
            - Explicit '?' character
            - Bangla question words (কি, কেন, কোথায়, ...)
            - '!' for exclamations
        """
        is_question  = bool(_QUESTION_MARKERS.search(text)) or bool(_BANGLA_QUESTION_WORDS.search(text))
        is_exclaim   = bool(_EXCLAIM_MARKERS.search(text))

        if is_question and is_exclaim:
            return SentenceType.MIXED
        if is_question:
            return SentenceType.QUESTION
        if is_exclaim:
            return SentenceType.EXCLAMATION
        return SentenceType.STATEMENT

    def insert_pauses(self, text: str) -> str:
        """
        Inserts pause markers based on punctuation.

        Pause durations:
            ,  → <pause:short>    (~150ms)
            ;  → <pause:medium>   (~300ms)
            .  → <pause:long>     (~500ms)
            ।  → <pause:long>     (~500ms)
            ?! → <pause:long>     (~500ms)
            -- → <pause:short>    (dash separator)
        """
        for pattern, replacement in _PAUSE_MAP:
            text = pattern.sub(replacement, text)
        # Collapse multiple consecutive pause markers
        text = re.sub(r'(<pause:\w+>\s*){2,}', r'\1', text)
        return text

    def mark_emphasis(self, text: str) -> str:
        """
        Wraps words preceded by emphasis indicators (!, very, অনেক, খুব) with
        emphasis markers for TTS engines that support it.

        Examples:
            "খুব ভালো" → "খুব <emphasis>ভালো</emphasis>"
            "অনেক বড়"  → "অনেক <emphasis>বড়</emphasis>"
        """
        emphasis_triggers = re.compile(
            r'\b(খুব|অনেক|বেশ|সত্যিই|really|very|so|extremely)\s+(\S+)',
            re.IGNORECASE
        )
        def _add_emphasis(match):
            trigger = match.group(1)
            target  = match.group(2)
            return f'{trigger} {self.emphasis_open}{target}{self.emphasis_close}'

        return emphasis_triggers.sub(_add_emphasis, text)

    def format(self, text: str, add_emphasis: bool = True) -> str:
        """
        Applies full prosody formatting to a text string.

        Args:
            text:          Input text
            add_emphasis:  Whether to insert emphasis markers (default True)

        Returns:
            Formatted text string with prosody markers inserted.
        """
        text = self.insert_pauses(text)
        if add_emphasis:
            text = self.mark_emphasis(text)
        # Normalize whitespace
        text = re.sub(r'  +', ' ', text).strip()
        return text

    def annotate(self, text: str, add_emphasis: bool = True) -> ProsodyAnnotation:
        """
        Full annotation: formats the text and returns metadata.

        Returns:
            ProsodyAnnotation with formatted_text, sentence_type, pause_count,
            and has_emphasis flag.
        """
        sentence_type = self.detect_sentence_type(text)
        formatted     = self.format(text, add_emphasis=add_emphasis)
        pause_count   = len(re.findall(r'<pause:\w+>', formatted))
        has_emphasis  = self.emphasis_open in formatted

        return ProsodyAnnotation(
            raw_text=text,
            formatted_text=formatted,
            sentence_type=sentence_type,
            pause_count=pause_count,
            has_emphasis=has_emphasis,
        )

    def strip_markers(self, text: str) -> str:
        """
        Removes all prosody markers from text (for clean TTS backends that
        don't support SSML-like syntax).
        """
        text = re.sub(r'<pause:\w+>', '', text)
        text = re.sub(r'</?emphasis>', '', text)
        return re.sub(r'  +', ' ', text).strip()


if __name__ == '__main__':
    pf = ProsodyFormatter()
    samples = [
        "আপনাকে ধন্যবাদ। আমি কীভাবে সাহায্য করতে পারি?",
        "সে খুব দ্রুত দৌড়াচ্ছে!",
        "গাড়িটি ঢাকা থেকে চট্টগ্রাম গেছে।",
        "আজকে meeting আছে, তুমি কি আসবে?",
    ]
    for s in samples:
        ann = pf.annotate(s)
        print(f"Input:   {ann.raw_text}")
        print(f"Type:    {ann.sentence_type.name}")
        print(f"Pauses:  {ann.pause_count}")
        print(f"Output:  {ann.formatted_text}")
        print()
