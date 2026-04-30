import re
from .numbers import bangla_to_ascii_digits, _ONES, NUMBER_WORDS

# 1C — Time spoken conversion rules
PERIODS = ['রাত', 'সকাল', 'দুপুর', 'বিকেল', 'সন্ধ্যা', 'ভোর', 'মধ্যরাত']

def _get_time_of_day(hour_24: int) -> str:
    """Returns the Bangla period of day word for a given 24h hour."""
    if 0 <= hour_24 <= 4:
        return 'রাত'
    elif 5 <= hour_24 <= 11:
        return 'সকাল'
    elif hour_24 == 12:
        return 'দুপুর'
    elif 13 <= hour_24 <= 16:
        return 'বিকেল'
    elif 17 <= hour_24 <= 19:
        return 'সন্ধ্যা'
    else:  # 20-23
        return 'রাত'

HOUR_WORDS = {
    1: 'একটা', 2: 'দুটো', 3: 'তিনটা', 4: 'চারটা',
    5: 'পাঁচটা', 6: 'ছয়টা', 7: 'সাতটা', 8: 'আটটা',
    9: 'নয়টা', 10: 'দশটা', 11: 'এগারোটা', 12: 'বারোটা',
}

def _format_time(hour_24: int, minute: int, include_period=True) -> str:
    period = _get_time_of_day(hour_24)
    hour_12 = hour_24 % 12 or 12
    hour_word = HOUR_WORDS.get(hour_12, _ONES[hour_12] + 'টা')
    
    parts = []
    if include_period:
        parts.append(period)
        
    if minute == 30:
        parts.append(f'সাড়ে {hour_word}')
    elif minute == 0:
        parts.append(hour_word)
    elif minute == 15:
        parts.append(f'{hour_word} পনেরো মিনিট')
    elif minute == 45:
        parts.append(f'{hour_word} পঁয়তাল্লিশ মিনিট')
    else:
        min_word = NUMBER_WORDS.get(str(minute), str(minute))
        parts.append(f'{hour_word} {min_word} মিনিট')
        
    return ' '.join(parts).strip()

def normalize_time(text: str) -> str:
    pattern = r'([01০১]?[0-9০-৯]|2[0-3০-৩]):([0-5০-৫][0-9০-৯])'

    def _replace(match):
        start = match.start()
        # Look back for period words (e.g. রাত ১০:৩০)
        lookback = text[max(0, start-15):start]
        has_period = any(p in lookback for p in PERIODS)
        
        hour = int(bangla_to_ascii_digits(match.group(1)))
        minute = int(bangla_to_ascii_digits(match.group(2)))
        
        return _format_time(hour, minute, include_period=not has_period)

    return re.sub(pattern, _replace, text)
