import re

MONTH_MAP = {
    'Jan': 'জানুয়ারি',
    'Feb': 'ফেব্রুয়ারি',
    'Mar': 'মার্চ',
    'Apr': 'এপ্রিল',
    'May': 'মে',
    'Jun': 'জুন',
    'Jul': 'জুলাই',
    'Aug': 'আগস্ট',
    'Sep': 'সেপ্টেম্বর',
    'Oct': 'অক্টোবর',
    'Nov': 'নভেম্বর',
    'Dec': 'ডিসেম্বর',
    # Support full names too
    'January': 'জানুয়ারি',
    'February': 'ফেব্রুয়ারি',
    'March': 'মার্চ',
    'April': 'এপ্রিল',
    'June': 'জুন',
    'July': 'জুলাই',
    'August': 'আগস্ট',
    'September': 'সেপ্টেম্বর',
    'October': 'অক্টোবর',
    'November': 'নভেম্বর',
    'December': 'ডিসেম্বর',
}

def normalize_months(text):
    """
    Normalizes month names (English) to Bangla.
    """
    for mon, expansion in MONTH_MAP.items():
        pattern = rf'\b{mon}\.?'
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
    return text