import re
from .numbers import normalize_number_to_word

# Maps currency symbols to their Bangla spoken form
CURRENCY_MAP = {
    '৳': 'টাকা',
    'Tk': 'টাকা',
    'BDT': 'টাকা',
    '$': 'ডলার',
    'USD': 'ডলার',
    '€': 'ইউরো',
    'EUR': 'ইউরো',
    '£': 'পাউন্ড',
    'GBP': 'পাউন্ড',
    '¥': 'ইয়েন',
    'JPY': 'ইয়েন',
    'INR': 'রুপি',
    '₹': 'রুপি',
}

def normalize_currency(text: str) -> str:
    """
    Normalizes currency expressions to Bangla spoken form.

    Examples:
        ৳100   → একশ টাকা
        $50    → পঞ্চাশ ডলার
        €20.5  → বিশ দশমিক পাঁচ ইউরো
        100 Tk → একশ টাকা
    """
    # Pattern 1: Symbol BEFORE the number (e.g. ৳100, $50.5)
    for symbol, bangla_unit in CURRENCY_MAP.items():
        pattern = rf'{re.escape(symbol)}\s*([0-9০-৯]+(?:[.,][0-9০-৯]+)?)'
        def _replace_prefix(match, unit=bangla_unit):
            raw_num_raw = match.group(1)
            # Ensure we handle dots correctly by not stripping them
            # Convert to ASCII and strip thousands commas but keep decimal dots
            bangla_to_ascii = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
            raw_num = raw_num_raw.translate(bangla_to_ascii).replace(',', '')
            
            num_word = normalize_number_to_word(raw_num)
            return f'{num_word} {unit}'
        text = re.sub(pattern, _replace_prefix, text)

    # Pattern 2: Symbol/word AFTER the number (e.g. 100 Tk, 50 BDT)
    for symbol, bangla_unit in CURRENCY_MAP.items():
        if symbol in ('৳', '$', '€', '£', '¥', '₹'):
            continue  # already handled as prefix above
        pattern = rf'([0-9০-৯]+(?:[.,][0-9০-৯]+)?)\s*{re.escape(symbol)}\b'
        def _replace_suffix(match, unit=bangla_unit):
            raw_num_raw = match.group(1)
            bangla_to_ascii = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
            raw_num = raw_num_raw.translate(bangla_to_ascii).replace(',', '')
            
            num_word = normalize_number_to_word(raw_num)
            return f'{num_word} {unit}'
        text = re.sub(pattern, _replace_suffix, text, flags=re.IGNORECASE)

    return text
