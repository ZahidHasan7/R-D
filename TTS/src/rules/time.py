import re
from .numbers import normalize_number_to_word

# Bangla time-of-day prefixes mapped to hour ranges (24h)
def _get_time_of_day(hour_24: int) -> str:
    """Returns the Bangla time-of-day word for a given 24h hour."""
    if hour_24 == 0:
        return 'মধ্যরাত'
    elif 1 <= hour_24 <= 4:
        return 'রাত'
    elif 5 <= hour_24 <= 6:
        return 'ভোর'
    elif 7 <= hour_24 <= 11:
        return 'সকাল'
    elif hour_24 == 12:
        return 'দুপুর'
    elif 13 <= hour_24 <= 15:
        return 'দুপুর'
    elif 16 <= hour_24 <= 17:
        return 'বিকাল'
    elif 18 <= hour_24 <= 19:
        return 'সন্ধ্যা'
    else:  # 20-23
        return 'রাত'

# Clock-hour spoken forms (1-12 in Bangla)
HOUR_WORDS = {
    1: 'একটা', 2: 'দুটো', 3: 'তিনটা', 4: 'চারটা',
    5: 'পাঁচটা', 6: 'ছয়টা', 7: 'সাতটা', 8: 'আটটা',
    9: 'নয়টা', 10: 'দশটা', 11: 'এগারোটা', 12: 'বারোটা',
}

# Common minute spoken forms
MINUTE_WORDS = {
    0: '',         # on the hour — omit
    5: 'পাঁচ',    10: 'দশ',     15: 'পনেরো',
    20: 'বিশ',    25: 'পঁচিশ',  30: 'ত্রিশ',
    35: 'পঁয়ত্রিশ', 40: 'চল্লিশ', 45: 'পঁয়তাল্লিশ',
    50: 'পঞ্চাশ', 55: 'পঞ্চান্ন',
}

def _minute_to_word(minute: int) -> str:
    """Convert minute integer to Bangla word."""
    if minute == 0:
        return ''
    if minute in MINUTE_WORDS:
        return MINUTE_WORDS[minute]
    # Fallback: use generic number word
    return normalize_number_to_word(str(minute))

def _format_time(hour_24: int, minute: int) -> str:
    """Formats a time into Bangla spoken string."""
    time_of_day = _get_time_of_day(hour_24)
    clock_hour = hour_24 % 12 or 12  # convert 0/24h to 12h clock
    hour_word = HOUR_WORDS.get(clock_hour, normalize_number_to_word(str(clock_hour)) + 'টা')
    minute_word = _minute_to_word(minute)

    if minute_word:
        return f'{time_of_day} {hour_word} {minute_word}'
    else:
        return f'{time_of_day} {hour_word}'

def normalize_time(text: str) -> str:
    """
    Normalizes time expressions to Bangla spoken form.

    Examples:
        10:30 AM  → সকাল দশটা ত্রিশ
        21:45     → রাত নয়টা পঁয়তাল্লিশ
        00:00     → মধ্যরাত বারোটা
        12:00 PM  → দুপুর বারোটা
        9:05 AM   → সকাল নয়টা পাঁচ
    """
    # Pattern: HH:MM (optional AM/PM). Supports ASCII and Bangla digits/indicators.
    pattern = r'\b([01০১]?[0-9০-৯]|2[0-3০-৩]):([0-5০-৫][0-9০-৯])(?:\s*(AM|PM|am|pm|এএম|পিএম))?\b'

    def _replace_time(match):
        hour_str = match.group(1)
        min_str  = match.group(2)
        ampm     = match.group(3)

        # Convert everything to ASCII for integer parsing
        from .numbers import bangla_to_ascii_digits
        hour = int(bangla_to_ascii_digits(hour_str))
        minute = int(bangla_to_ascii_digits(min_str))

        # Convert to 24h if any AM/PM indicator is given
        if ampm:
            ampm_lower = ampm.lower()
            # Handle both English and Bangla indicators
            is_pm = ampm_lower in ('pm', 'পিএম')
            is_am = ampm_lower in ('am', 'এএম')

            if is_pm and hour != 12:
                hour += 12
            elif is_am and hour == 12:
                hour = 0  # midnight edge case

        return _format_time(hour, minute)

    return re.sub(pattern, _replace_time, text)
