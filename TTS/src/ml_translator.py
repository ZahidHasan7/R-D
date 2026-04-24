import re
import urllib.request
import urllib.parse
import json

class MLTranslator:
    def __init__(self):
        # Repurposed MLTranslator to Transliteration for code-mixing
        print("Initialized Phonetic Transliteration API for cross-coded English words.")

    def _transliterate_word(self, word):
        url = "https://inputtools.google.com/request?text=" + urllib.parse.quote(word) + "&itc=bn-t-i0-und&num=1"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                res = json.loads(response.read().decode())
                return res[1][0][1][0]
        except:
            return word

    def translate(self, text):
        """
        Transliterates English tokens to Bengali phonetics for cross-code pronunciation.
        """
        if not re.search(r'[a-zA-Z]', text):
            return text

        def _replace_en_chunk(match):
            return self._transliterate_word(match.group(0))

        # Replace all English words dynamically
        return re.sub(r'[a-zA-Z]+', _replace_en_chunk, text)
