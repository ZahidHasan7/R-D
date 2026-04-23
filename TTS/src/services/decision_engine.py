import os
import sys
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.classifier import Classifier
from src.services.unit_handler import UnitHandler
from src.services.abbreviation_handler import AbbreviationHandler
from src.utils.token_splitter import split_mixed_token
from src.utils.text_utils import spell_out

class DecisionEngine:
    """
    Routes tokens based on Classifier output.
    """
    def __init__(self):
        self.classifier = Classifier()
        self.unit_handler = UnitHandler()
        self.abbrev_handler = AbbreviationHandler()
        
    def process(self, token: str, next_token: str = '', lang: str = 'neutral') -> str:
        if not token:
            return ""
            
        label = self.classifier.classify_token(token)
        
        if label == "UNIT":
            return self.unit_handler.normalize_unit(token)
            
        elif label == "MIXED":
            parts = split_mixed_token(token)
            processed_parts = []
            for p in parts:
                # Recurse for each part
                processed_parts.append(self.process(p, next_token, lang))
            return " ".join(processed_parts)
            
        elif label == "ABBREVIATION":
            out = self.abbrev_handler.process_token(token, next_token, lang)
            # Fallback strategy if abbreviation handler returns strictly English letters
            if re.match(r'^[A-Za-z\s]+$', out):
                return spell_out(out.replace(' ', ''))
            return out
            
        else:
            # NORMAL - fallback to default behavior (e.g. check if there's any abbreviation fallback needed)
            if re.match(r'^[A-Za-z]+$', token) and len(token) <= 4 and token.isupper():
                return spell_out(token)
            return token

if __name__ == "__main__":
    import re
    de = DecisionEngine()
    print(de.process("GPU"))
    print(de.process("10GB"))
    print(de.process("AI-based"))
