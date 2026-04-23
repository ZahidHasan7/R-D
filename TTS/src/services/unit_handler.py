import os
import json
import re

_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'dictionaries', 'unit_dict.json')

class UnitHandler:
    def __init__(self):
        self._dict = self._load_dict()
        # Create a regex to match units and ignore case or spaces
        # Sort by length descending to match longest first (e.g. km/h before km)
        keys = sorted(self._dict.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in keys]
        self._unit_pattern = re.compile(f"({('|'.join(escaped_keys))})$", re.IGNORECASE)

    def _load_dict(self) -> dict:
        try:
            with open(_DICT_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def normalize_unit(self, token: str) -> str:
        """
        Normalizes a unit token (e.g., 'GB' -> 'গিগাবাইট', '10GB' -> '10 গিগাবাইট').
        """
        if not token:
            return token

        # If it perfectly matches a unit exactly
        token_lower = token.lower()
        if token_lower in self._dict:
            return self._dict[token_lower]

        # Extract numerical prefix + unit suffix using regex
        # Updated to include Bangla digits ০-৯
        match = re.match(r'^([0-9০-৯.]+)(.+)$', token)
        if match:
            num_part = match.group(1)
            unit_part = match.group(2).lower()
            if unit_part in self._dict:
                return f"{num_part} {self._dict[unit_part]}"

        # Maybe there's a space or it's just the symbol at the end
        match_suffix = self._unit_pattern.search(token_lower)
        if match_suffix:
            unit_sym = match_suffix.group(1)
            prefix = token[:match_suffix.start()].strip()
            if not prefix:
                return self._dict[unit_sym]
            return f"{prefix} {self._dict[unit_sym]}"

        return token

if __name__ == "__main__":
    handler = UnitHandler()
    print(handler.normalize_unit("GB"))
    print(handler.normalize_unit("10GB"))
    print(handler.normalize_unit("80km/h"))
