import re

# Bangla digits mapping
BANGLA_DIGITS = {
    '0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪',
    '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'
}

# Full Bangla ones (0-19)
_ONES = [
    'শূন্য', 'এক', 'দুই', 'তিন', 'চার', 'পাঁচ', 'ছয়', 'সাত', 'আট', 'নয়',
    'দশ', 'এগারো', 'বারো', 'তেরো', 'চৌদ্দ', 'পনেরো', 'ষোলো', 'সতেরো', 'আঠারো', 'উনিশ',
]

# Bangla tens
_TENS = [
    '', 'দশ', 'বিশ', 'ত্রিশ', 'চল্লিশ', 'পঞ্চাশ', 'ষাট', 'সত্তর', 'আশি', 'নব্বই',
]

# Legacy lookup table kept for backward compat with existing normalize_numbers()
NUMBER_WORDS = {
    '0': 'শূন্য', '1': 'এক', '2': 'দুই', '3': 'তিন', '4': 'চার',
    '5': 'পাঁচ', '6': 'ছয়', '7': 'সাত', '8': 'আট', '9': 'নয়',
    '10': 'দশ', '11': 'এগারো', '12': 'বারো', '13': 'তেরো', '14': 'চৌদ্দ', '15': 'পনেরো',
    '16': 'ষোলো', '17': 'সতেরো', '18': 'আঠারো', '19': 'উনিশ',
    '20': 'বিশ', '21': 'একুশ', '22': 'বাইশ', '23': 'তেইশ', '24': 'চব্বিশ',
    '25': 'পঁচিশ', '26': 'ছাব্বিশ', '27': 'সাতাশ', '28': 'আটাশ', '29': 'উনত্রিশ',
    '30': 'ত্রিশ', '31': 'একত্রিশ', '32': 'বত্রিশ', '33': 'তেত্রিশ', '34': 'চৌত্রিশ',
    '35': 'পঁয়ত্রিশ', '36': 'ছত্রিশ', '37': 'সাঁইত্রিশ', '38': 'আটত্রিশ', '39': 'ঊনচল্লিশ',
    '40': 'চল্লিশ', '41': 'একচল্লিশ', '42': 'বিয়াল্লিশ', '43': 'তেতাল্লিশ', '44': 'চৌয়াল্লিশ',
    '45': 'পঁয়তাল্লিশ', '46': 'ছেচল্লিশ', '47': 'সাতচল্লিশ', '48': 'আটচল্লিশ', '49': 'ঊনপঞ্চাশ',
    '50': 'পঞ্চাশ', '51': 'একান্ন', '52': 'বায়ান্ন', '53': 'তিপ্পান্ন', '54': 'চৌয়ান্ন',
    '55': 'পঞ্চান্ন', '56': 'ছাপ্পান্ন', '57': 'সাতান্ন', '58': 'আটান্ন', '59': 'ঊনষাট',
    '60': 'ষাট', '61': 'একষট্টি', '62': 'বাষট্টি', '63': 'তেষট্টি', '64': 'চৌষট্টি',
    '65': 'পঁয়ষট্টি', '66': 'ছেষট্টি', '67': 'সাতষট্টি', '68': 'আটষট্টি', '69': 'ঊনসত্তর',
    '70': 'সত্তর', '71': 'একাত্তর', '72': 'বাহাত্তর', '73': 'তিয়াত্তর', '74': 'চৌয়াত্তর',
    '75': 'পঁচাত্তর', '76': 'ছিয়াত্তর', '77': 'সাতাত্তর', '78': 'আটাত্তর', '79': 'ঊনআশি',
    '80': 'আশি', '81': 'একাশি', '82': 'বিরাশি', '83': 'তিরাশি', '84': 'চৌরাশি',
    '85': 'পঁচাশী', '86': 'ছিয়াশি', '87': 'সাতাশি', '88': 'আটাশি', '89': 'ঊননব্বই',
    '90': 'নব্বই', '91': 'একানব্বই', '92': 'বিরানব্বই', '93': 'তিরানব্বই', '94': 'চৌরানব্বই',
    '95': 'পঁচানব্বই', '96': 'ছিয়ানব্বই', '97': 'সাতানব্বই', '98': 'আটানব্বই', '99': 'নিরানব্বই',
    '100': 'একশ', '200': 'দুশো', '300': 'তিনশো', '400': 'চারশো', '500': 'পাঁচশো',
    '600': 'ছয়শো', '700': 'সাতশো', '800': 'আটশো', '900': 'নয়শো',
    '1000': 'এক হাজার',
}

# Digit mapping
BN_DIGITS = "০১২৩৪৫৬৭৮৯"
EN_DIGITS = "0123456789"
BN_TO_EN_MAP = str.maketrans(BN_DIGITS, EN_DIGITS)

def bangla_to_ascii_digits(text: str) -> str:
    """Converts Bangla digits (০-৯) to ASCII digits (0-9)."""
    return text.translate(BN_TO_EN_MAP)


def normalize_number_to_word(num_str: str) -> str:
    """
    Converts a numeral string (integer or decimal) to Bangla words.
    Handles English digits (0-9) and Bangla digits (০-৯).
    Supports numbers up to 9,99,99,999 (ten crore).

    Examples:
        '5'      → 'পাঁচ'
        '100'    → 'একশ'
        '1500'   → 'পনেরোশো'
        '3.5'    → 'তিন দশমিক পাঁচ'
        '21'     → 'একুশ'
    """
    # Normalize Bangla digits to ASCII first, and remove commas
    bangla_to_ascii = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
    num_str = num_str.translate(bangla_to_ascii).replace(',', '').strip()

    # Handle decimal
    if '.' in num_str:
        parts = num_str.split('.', 1)
        left_word = normalize_number_to_word(parts[0])
        # For decimal part, expand digit by digit (e.g., .75 -> সাত পাঁচ)
        right_digits = parts[1]
        right_words = [NUMBER_WORDS.get(d, d) for d in right_digits]
        return f'{left_word} দশমিক {" ".join(right_words)}'

    try:
        n = int(num_str)
    except ValueError:
        return num_str  # return as-is if not parseable

    return _int_to_bangla(n)


def _int_to_bangla(n: int) -> str:
    """Recursive integer-to-Bangla-words converter."""
    if n < 0:
        return 'মাইনাস ' + _int_to_bangla(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        # Priority: use the lookup table for composite numbers (21, 22, etc.)
        if str(n) in NUMBER_WORDS:
            return NUMBER_WORDS[str(n)]
        
        tens = n // 10
        ones = n % 10
        return _TENS[tens] + ('' if ones == 0 else ' ' + _ONES[ones])
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        h_word = _ONES[hundreds] + 'শো' if hundreds > 1 else 'একশ'
        return h_word + ('' if remainder == 0 else ' ' + _int_to_bangla(remainder))
    if n < 100000:  # up to 99,999 (ninetynine thousand)
        thousands = n // 1000
        remainder = n % 1000
        t_word = _int_to_bangla(thousands) + ' হাজার'
        return t_word + ('' if remainder == 0 else ' ' + _int_to_bangla(remainder))
    if n < 10000000:  # up to 99,99,999 (lakh system)
        lakhs = n // 100000
        remainder = n % 100000
        l_word = _int_to_bangla(lakhs) + ' লাখ'
        return l_word + ('' if remainder == 0 else ' ' + _int_to_bangla(remainder))
    # Crore
    crores = n // 10000000
    remainder = n % 10000000
    c_word = _int_to_bangla(crores) + ' কোটি'
    return c_word + ('' if remainder == 0 else ' ' + _int_to_bangla(remainder))

def to_bangla_digits(text):
    for eng, bg in BANGLA_DIGITS.items():
        text = text.replace(eng, bg)
    return text

def normalize_numbers(text):
    """
    Finds all number-like strings (ASCII/Bangla digits, commas, dots) 
    and converts them to Bangla words.
    """
    # Regex to find: 
    # 1. Decimals/Integers with commas: 12,345.67 or ১২,৩৪৫.৬৭
    # 2. Simple integers: 100 or ১০০
    # Updated pattern to better handle dots as decimals
    pattern = r'[0-9০-৯]+(?:,[0-9০-৯]+)*(?:\.[0-9০-৯]+)?'

    def _replace_match(match):
        num_str = match.group(0)
        # Ensure we don't normalize just a dot if it somehow got matched
        if num_str == '.':
            return '.'
        return normalize_number_to_word(num_str)

    # Use word boundary lookahead/lookbehind.
    # We allow Bangla characters to be attached (suffixes like 'টা', 'টি') 
    # but avoid matching inside English words.
    safe_pattern = rf'(?<![a-zA-Z]){pattern}(?![a-zA-Z])'
    
    return re.sub(safe_pattern, _replace_match, text)