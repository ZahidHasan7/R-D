# English letter pronunciation map in Bangla
_ENG_LETTER_MAP = {
    'A': 'এ', 'B': 'বি', 'C': 'সি', 'D': 'ডি', 'E': 'ই', 'F': 'এফ',
    'G': 'জি', 'H': 'এইচ', 'I': 'আই', 'J': 'জে', 'K': 'কে', 'L': 'এল',
    'M': 'এম', 'N': 'এন', 'O': 'ও', 'P': 'পি', 'Q': 'কিউ', 'R': 'আর',
    'S': 'এস', 'T': 'টি', 'U': 'ইউ', 'V': 'ভি', 'W': 'ডব্লিউ', 'X': 'এক্স',
    'Y': 'ওয়াই', 'Z': 'জেড'
}

def spell_out(token: str) -> str:
    """
    Reads out an English abbreviation letter by letter in Bangla.
    Example: cpu -> C P U -> সি পি ইউ
    """
    clean_token = token.upper().replace('.', '')
    pronounced = []
    
    for char in clean_token:
        # If the character is in the map, use it. Otherwise, pass it through.
        if char in _ENG_LETTER_MAP:
            pronounced.append(_ENG_LETTER_MAP[char])
        else:
            pronounced.append(char)
            
    return ' '.join(pronounced)

if __name__ == "__main__":
    print(spell_out("cpu"))
    print(spell_out("API"))
    print(spell_out("WHO"))
