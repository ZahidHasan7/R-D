import re

# Bangla digits mapping
BANGLA_DIGITS = {
    '0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪',
    '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'
}

_ONES = [
    'শূন্য', 'এক', 'দুই', 'তিন', 'চার', 'পাঁচ', 'ছয়', 'সাত', 'আট', 'নয়',
    'দশ', 'এগারো', 'বারো', 'তেরো', 'চৌদ্দ', 'পনেরো', 'ষোলো', 'সতেরো', 'আঠারো', 'উনিশ',
]

_TENS = [
    '', 'দশ', 'বিশ', 'ত্রিশ', 'চল্লিশ', 'পঞ্চাশ', 'ষাট', 'সত্তর', 'আশি', 'নব্বই',
]

# Cardinal mapping for standard numbers (0-99)
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
}

SPECIAL_ORDINALS = {
    '১ম': 'প্রথম', '২য়': 'দ্বিতীয়', '৩য়': 'তৃতীয়', '৪র্থ': 'চতুর্থ', '৫ম': 'পঞ্চম',
    '৬ষ্ঠ': 'ষষ্ঠ', '৭ম': 'সপ্তম', '৮ম': 'অষ্টম', '৯ম': 'নবম', '১০ম': 'দশম'
}

BN_TO_EN_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def bangla_to_ascii_digits(text: str) -> str:
    return text.translate(BN_TO_EN_MAP)

def _int_to_bangla(n: int) -> str:
    if n < 0: return 'মাইনাস ' + _int_to_bangla(-n)
    if n < 20: return _ONES[n]
    if n < 100:
        if str(n) in NUMBER_WORDS: return NUMBER_WORDS[str(n)]
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ('' if ones == 0 else ' ' + _ONES[ones])
    if n < 1000:
        hundreds, remainder = divmod(n, 100)
        h_word = _ONES[hundreds] + 'শো' if hundreds > 1 else 'একশো'
        return h_word + ('' if remainder == 0 else ' ' + _int_to_bangla(remainder))
    if n < 100000:
        thousands, remainder = divmod(n, 1000)
        t_word = _int_to_bangla(thousands) + ' হাজার'
        return t_word + ('' if remainder == 0 else ' ' + _int_to_bangla(remainder))
    if n < 10000000:
        lakhs, remainder = divmod(n, 100000)
        l_word = _int_to_bangla(lakhs) + ' লাখ'
        return l_word + ('' if remainder == 0 else ' ' + _int_to_bangla(remainder))
    crores, remainder = divmod(n, 10000000)
    c_word = _int_to_bangla(crores) + ' কোটি'
    return c_word + ('' if remainder == 0 else ' ' + _int_to_bangla(remainder))

def normalize_number_to_word(num_str: str) -> str:
    clean = bangla_to_ascii_digits(num_str).replace(',', '')
    if '.' in clean:
        parts = clean.split('.')
        whole = _int_to_bangla(int(parts[0]))
        dec = expand_digit_by_digit(parts[1])
        return f"{whole} দশমিক {dec}"
    return _int_to_bangla(int(clean))

def expand_digit_by_digit(num_str: str) -> str:
    ascii_str = bangla_to_ascii_digits(num_str)
    digits = [d for d in ascii_str if d.isdigit()]
    return ' '.join(_ONES[int(d)] for d in digits)

def expand_year(num_str: str) -> str:
    ascii_str = bangla_to_ascii_digits(num_str)
    n = int(ascii_str)
    if 1000 <= n <= 1999:
        hundreds, remainder = divmod(n, 100)
        # Use space between tens and "শো" for clearer model pronunciation
        # e.g. 1971 → "উনিশ শো একাত্তর" not "উনিশশো একাত্তর"
        h_part = _ONES[hundreds] + ' শো'
        r_part = NUMBER_WORDS.get(str(remainder), _int_to_bangla(remainder)) if remainder > 0 else ''
        return (h_part + ' ' + r_part).strip()
    return _int_to_bangla(n)

def normalize_numbers(text: str) -> str:
    # Pattern includes optional ordinal suffixes
    num_pattern = r'[0-9০-৯]+(?:[-,.][0-9০-৯]+)*(?:ম|য়|র্থ|ষ্ঠ|তম)?'
    
    def _replace(match):
        raw_full = match.group(0)
        # Find just the number part
        match_num = re.search(r'[0-9০-৯]+(?:[-,.][0-9০-৯]+)*', raw_full)
        if not match_num: return raw_full
        raw = match_num.group(0)
        suffix_part = raw_full[len(raw):]
        
        clean = bangla_to_ascii_digits(raw).replace(',', '')
        
        # Context window
        start = match.start()
        prefix = text[max(0, start-30):start].lower()
        suffix_context = text[match.end():match.end()+15]
        
        # 1A & 1B: Phone / ID / OTP / Code (Digit by digit)
        phone_signals = ["ফোন", "মোবাইল", "নম্বর", "নাম্বার", "হেল্পলাইন", "nid", "পাসপোর্ট", "ভোটার আইডি", "আইডি", "otp", "পিন", "pin", "কোড", "id #", "id:", "#"]
        if any(sig in prefix for sig in phone_signals):
            return expand_digit_by_digit(raw)
        
        if len(clean.replace('-', '').replace('.', '')) >= 10:
            return expand_digit_by_digit(raw)

        # 1F: Ordinals
        if suffix_part or any(suffix_context.startswith(s) for s in ["ম", "য়", "র্থ", "ষ্ঠ", "তম"]):
            s = suffix_part or next((s for s in ["ম", "য়", "র্থ", "ষ্ঠ", "তম"] if suffix_context.startswith(s)), "")
            if raw + s in SPECIAL_ORDINALS:
                return SPECIAL_ORDINALS[raw + s]
            try:
                return _int_to_bangla(int(clean)) + "তম"
            except: pass

        # 1E: Years
        year_signals = ["সালে", "সনে", "সালের", "খ্রিস্টাব্দ", "বঙ্গাব্দ", "ইং"]
        if any(sig in suffix_context for sig in year_signals):
            try: return expand_year(clean)
            except: pass
            
        if len(clean) == 4 and '.' not in clean and '-' not in clean:
            try:
                n_val = int(clean)
                if 1000 <= n_val <= 2099:
                    # Standalone 4-digit numbers in this range are years
                    first_suffix_char = suffix_context.strip()[:1]
                    if not first_suffix_char or not first_suffix_char.isalpha() or any(sig.startswith(first_suffix_char) for sig in year_signals):
                        return expand_year(clean)
            except: pass

        # 1G: Decimal
        if '.' in raw:
            parts = clean.split('.')
            try:
                whole = _int_to_bangla(int(parts[0]))
                dec = expand_digit_by_digit(parts[1])
                return f"{whole} দশমিক {dec}"
            except: pass

        # 1H & 1I: Plain Cardinal
        try:
            return _int_to_bangla(int(clean))
        except:
            return raw_full

    return re.sub(num_pattern, _replace, text)