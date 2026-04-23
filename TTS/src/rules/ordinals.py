import re

# Explicit ordinals for the most common values
ORDINAL_MAP = {
    '1st': 'প্রথম', '2nd': 'দ্বিতীয়', '3rd': 'তৃতীয়', '4th': 'চতুর্থ',
    '5th': 'পঞ্চম', '6th': 'ষষ্ঠ', '7th': 'সপ্তম', '8th': 'অষ্টম',
    '9th': 'নবম', '10th': 'দশম', '11th': 'একাদশ', '12th': 'দ্বাদশ',
    '13th': 'ত্রয়োদশ', '14th': 'চতুর্দশ', '15th': 'পঞ্চদশ',
    '16th': 'ষোড়শ', '17th': 'সপ্তদশ', '18th': 'অষ্টাদশ',
    '19th': 'ঊনবিংশ', '20th': 'বিংশ', '21st': 'একুশতম',
    '22nd': 'বাইশতম', '23rd': 'তেইশতম', '24th': 'চব্বিশতম',
    '25th': 'পঁচিশতম', '30th': 'ত্রিশতম', '31st': 'একত্রিশতম',
}

# Special Bangla date-day forms (for "Xth <Month>" patterns)
DATE_ORDINAL_MAP = {
    '1':  'পহেলা',   '2':  'দোসরা',   '3':  'তেসরা',   '4':  'চৌঠা',
    '5':  'পাঁচই',  '6':  'ছয়ই',    '7':  'সাতই',    '8':  'আটই',
    '9':  'নয়ই',   '10': 'দশই',     '11': 'এগারোই',  '12': 'বারোই',
    '13': 'তেরোই',  '14': 'চোদ্দই',  '15': 'পনেরোই',  '16': 'ষোলোই',
    '17': 'সতেরোই', '18': 'আঠারোই',  '19': 'উনিশই',   '20': 'বিশই',
    '21': 'একুশে',  '22': 'বাইশে',   '23': 'তেইশে',   '24': 'চব্বিশে',
    '25': 'পঁচিশে', '26': 'ছাব্বিশে','27': 'সাতাশে',  '28': 'আঠাশে',
    '29': 'উনত্রিশে','30': 'ত্রিশে',  '31': 'একত্রিশে',
}

MONTH_MAP_EN_BN = {
    'january': 'জানুয়ারি', 'february': 'ফেব্রুয়ারি', 'march': 'মার্চ',
    'april': 'এপ্রিল', 'may': 'মে', 'june': 'জুন', 'july': 'জুলাই',
    'august': 'আগস্ট', 'september': 'সেপ্টেম্বর', 'october': 'অক্টোবর',
    'november': 'নভেম্বর', 'december': 'ডিসেম্বর',
    'jan': 'জানুয়ারি', 'feb': 'ফেব্রুয়ারি', 'mar': 'মার্চ',
    'apr': 'এপ্রিল', 'jun': 'জুন', 'jul': 'জুলাই',
    'aug': 'আগস্ট', 'sep': 'সেপ্টেম্বর', 'oct': 'অক্টোবর',
    'nov': 'নভেম্বর', 'dec': 'ডিসেম্বর',
}

def normalize_ordinals(text: str) -> str:
    """
    Normalizes English ordinal numbers and date expressions to Bangla.

    Examples:
        1st        → প্রথম
        21st       → একুশতম
        21st February → একুশে ফেব্রুয়ারি
        5th Aug    → পাঁচই আগস্ট
    """
    # Step 1: Handle "Xst/Xnd/Xrd/Xth <Month>" date patterns first (special case)
    # Updated to support Bangla digits and optional spaces
    month_pattern = '|'.join(re.escape(m) for m in MONTH_MAP_EN_BN.keys())
    date_with_month = rf'\b([0-9০-৯]{{1,2}})\s*(?:st|nd|rd|th)\s+({month_pattern})\b'

    def _replace_date(match):
        day_raw = match.group(1).strip()
        bangla_to_ascii = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
        day = day_raw.translate(bangla_to_ascii)
        
        month_raw = match.group(2).lower()
        day_word = DATE_ORDINAL_MAP.get(day, day_raw)
        month_word = MONTH_MAP_EN_BN.get(month_raw, month_raw)
        return f'{day_word} {month_word}'

    text = re.sub(date_with_month, _replace_date, text, flags=re.IGNORECASE)

    # Step 2: Handle standalone ordinals (e.g. "1st", "21st", "3rd")
    def _replace_ordinal(match):
        full = match.group(0)   # e.g. "21st" or "21 st"
        num_raw  = match.group(1).strip()
        suffix = match.group(2).lower()
        
        bangla_to_ascii = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
        num = num_raw.translate(bangla_to_ascii)
        
        key = f'{num}{suffix}'
        return ORDINAL_MAP.get(key, full)

    text = re.sub(r'\b([0-9০-৯]{1,2})\s*(st|nd|rd|th)\b', _replace_ordinal, text, flags=re.IGNORECASE)

    return text
