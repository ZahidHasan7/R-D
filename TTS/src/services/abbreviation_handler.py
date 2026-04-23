import os
import json
import re
import sys

# Add src to path so we can import utils safely
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.text_utils import spell_out
from src.utils.token_splitter import split_mixed_token

_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'dictionaries', 'abbreviation_dict.json')

class AbbreviationHandler:
    def __init__(self):
        self._dict = self._load_dict()

    def _load_dict(self) -> dict:
        try:
            with open(_DICT_PATH, 'r', encoding='utf-8') as f:
                # Store keys as lower for case-insensitive exact matching
                data = json.load(f)
                return {k.lower(): v for k, v in data.items()}
        except FileNotFoundError:
            return {}

    def process_token(self, token: str, next_token: str = '', lang: str = 'neutral') -> str:
        """
        Implementation of the 6-layer logic for abbreviation resolution.
        """
        if not token:
            return ""

        token_lower = token.lower()

        # Step 1: Dictionary lookup (highest priority)
        if token_lower in self._dict:
            return self._dict[token_lower]

        # Step 2: Context-aware rules (Dr. vs Drive)
        if token_lower == 'dr.' or token_lower == 'dr':
            # Simple heuristic: if next token is capitalized name, it's Doctor.
            # Very simplistic for now, assumes uppercase or Bangla name
            if next_token and (next_token[0].isupper() or '\u09A1' <= next_token[0] <= '\u09B9'):
                return "ডক্টর"
            return "ড্রাইভ"

        if token_lower == 'prof.' or token_lower == 'prof':
            return "প্রফেসর"

        # Step 4: Mixed token handling (e.g. AI-based)
        # Checking if it has hyphens or numbers mixed in
        if '-' in token or (re.search(r'\d', token) and re.search(r'[a-zA-Z]', token)):
            parts = split_mixed_token(token)
            if len(parts) > 1:
                # Process parts recursively
                out_parts = []
                for p in parts:
                    # Recursive call but don't infinitely loop
                    if re.match(r'^\d+$', p):
                        out_parts.append(p)
                    else:
                        out_parts.append(self.process_token(p, '', lang=lang))
                return ' '.join(out_parts)

        # Step 3 & 5: Uppercase abbreviation detection & Lowercase abbreviation support
        # Example: 'CPU' or 'cpu'. If it's all alphabetical and 2-5 chars, likely abbreviation
        # if not common word and language is English
        if re.match(r'^[a-zA-Z]{2,5}$', token):
            if token.isupper():
                return spell_out(token)
            elif token.islower() and token_lower not in ['is', 'in', 'at', 'on', 'of', 'to', 'the', 'and', 'for', 'a', 'an']:
                # Assume technical abbreviation like cpu or api
                return spell_out(token)

        # Step 6: Fallback spelling
        if re.match(r'^[A-Za-z]+$', token):
            return spell_out(token)
            
        # Bangla abbreviations like বি.এন.পি
        if '.' in token and re.match(r'^([\u0980-\u09FF]+\.)+[\u0980-\u09FF]*$', token):
            # Safeguard: ensure it's not a number
            if re.match(r'^[0-9০-৯.]+$', token):
                return token
            parsed = token.replace('.', '')
            return self._dict.get(parsed, parsed)
            
        return token

if __name__ == "__main__":
    handler = AbbreviationHandler()
    print(handler.process_token("API"))
    print(handler.process_token("cpu"))
    print(handler.process_token("Dr.", "Rahman"))
    print(handler.process_token("AI-based"))
    print(handler.process_token("বি.এন.পি"))
