import re

UNIT_MAP = {
    'kg': 'কেজি',
    'g': 'গ্রাম',
    'km': 'কিলোমিটার',
    'm': 'মিটার',
    'cm': 'সেন্টিমিটার',
    'mm': 'মিলিমিটার',
    'mg': 'মিলিগ্রাম',
    'ml': 'মিলিলিটার',
    'h': 'ঘণ্টা',
    'hr': 'ঘণ্টা',
    'min': 'মিনিট',
    's': 'সেকেন্ড',
}

COMPOUND_UNIT_MAP = {
    r'([0-9০-৯]+)\s*km/h': r'\1 কিলোমিটার প্রতি ঘণ্টা',
    r'([0-9০-৯]+)\s*কিমি/ঘণ্টা': r'\1 কিলোমিটার প্রতি ঘণ্টা',
}

def normalize_units(text):
    """
    Expands units to Bangla equivalents.
    Compound units (km/h) are handled BEFORE individual units
    to prevent partial replacement.
    """
    # Step 1: Handle compound units first
    for pattern, replacement in COMPOUND_UNIT_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Step 2: Handle individual units
    for unit, expansion in UNIT_MAP.items():
        pattern = rf'([0-9০-৯]+)\s*({unit})\b'
        text = re.sub(pattern, rf'\1 {expansion}', text, flags=re.IGNORECASE)

    return text