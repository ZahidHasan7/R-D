"""
ner_handler.py — Named Entity Recognition handler for Bangla TTS.
"""

import re
import json
import os
from typing import Optional

_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dictionaries', 'pronunciation_dict.json')

_SECTION_TYPE_MAP = {
    'names': 'name',
    'loanwords': 'exception',
    'brands': 'brand',
    'exceptions': 'exception',
}

def _load_entity_db() -> dict[str, dict]:
    db: dict[str, dict] = {}
    try:
        with open(_DICT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for section_name, entries in data.items():
            if section_name.startswith('_') or not isinstance(entries, dict):
                continue
            etype = _SECTION_TYPE_MAP.get(section_name, 'exception')
            for word, pronunciation in entries.items():
                db[word.lower()] = {'pronunciation': pronunciation, 'type': etype}
    except FileNotFoundError:
        pass
    return db

class NERHandler:
    def __init__(self, literal_mode=True):
        self._db = _load_entity_db()
        self._overrides: dict[str, str] = {}
        self.literal_mode = literal_mode

    def lookup(self, word: str) -> Optional[str]:
        key = word.lower()
        pronunciation = None
        if key in self._overrides:
            pronunciation = self._overrides[key]
        elif key in self._db:
            pronunciation = self._db[key]['pronunciation']
        
        if pronunciation and self.literal_mode:
            # Strip phonetic-only characters for clean text output
            return pronunciation.replace('-', '').replace('্', '')
        return pronunciation

    def handle(self, text: str) -> str:
        tokens = text.split()
        result = []
        for token in tokens:
            stripped = token.rstrip('.,!?;:।')
            punct = token[len(stripped):]
            pronunciation = self.lookup(stripped)
            if pronunciation:
                result.append(pronunciation + punct)
            else:
                result.append(token)
        return ' '.join(result)

if __name__ == '__main__':
    ner = NERHandler(literal_mode=True)
    print(ner.handle("ঢাকা এবং চট্টগ্রাম"))
