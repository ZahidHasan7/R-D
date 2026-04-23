import re

ABBREV_MAP = {
    'Dr.': 'ডাক্তার',
    'Mr.': 'মিস্টার',
    'Mrs.': 'মিসেস',
    'Ms.': 'মিস',
    'Prof.': 'প্রফেসর',
    'Eng.': 'ইঞ্জিনিয়ার',
    'ড.': 'ডাক্তার',
}

def normalize_abbreviations(text):
    """
    Expands abbreviations to Bangla.
    """
    for abbr, expansion in ABBREV_MAP.items():
        # Handle the case where it might be followed by a name or space
        # Use regex with word-boundary, supporting both ASCII and Bangla context
        pattern = rf'(?<![a-zA-Z\u0980-\u09FF]){re.escape(abbr)}(?![a-zA-Z\u0980-\u09FF])'
        text = re.sub(pattern, expansion, text)
        
    return text