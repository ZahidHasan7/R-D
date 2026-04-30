import re
from src.language_detector import LanguageDetector

def segment_text(text: str, detector: LanguageDetector):
    """
    Groups tokens by language to create stable synthesis chunks.
    Returns: List of tuples (lang, text_chunk)
    """
    if not text:
        return []
        
    # Initial tokenization based on whitespace
    tokens = text.split()
    if not tokens:
        return []
        
    segments = []
    current_lang = None
    current_chunk = []
    
    for token in tokens:
        # Detect token language (ignoring trailing punctuation for detection)
        clean_token = token.rstrip('.,!?;:।')
        lang = detector.detect_token(clean_token)
        
        # 'neutral' tokens (like digits or punctuation) stick to the current language context
        if lang == 'neutral' and current_lang is not None:
            lang = current_lang
        elif lang == 'neutral':
            lang = 'bangla' # Default for neutral start
            
        # Treat 'mixed' as 'bangla' for segmentation (bangla model handles mixed via transliteration)
        if lang == 'mixed':
            lang = 'bangla'
            
        if lang != current_lang:
            if current_chunk:
                segments.append((current_lang, ' '.join(current_chunk)))
            current_lang = lang
            current_chunk = [token]
        else:
            current_chunk.append(token)
            
    if current_chunk:
        segments.append((current_lang, ' '.join(current_chunk)))
        
    return segments
