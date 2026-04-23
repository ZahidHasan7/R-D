import sys
import os

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalizer import TextNormalizer
from src.prosody_formatter import ProsodyFormatter

text = '"Sure… I’d like that"'
print("RAW INPUT:", text)

norm = TextNormalizer(use_ml=True)
normalized = norm.normalize(text)
print("NORMALIZED:", normalized)

prosody = ProsodyFormatter()
formatted = prosody.format(normalized)
print("PROSODY FORMATTED:", formatted)
