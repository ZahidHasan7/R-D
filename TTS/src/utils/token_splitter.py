import re

def split_mixed_token(token: str) -> list[str]:
    """
    Splits complex mixed tokens into constituent parts.
    Examples:
      'modelটা' -> ['model', 'টা']
      'AI-based' -> ['AI', 'based']
      '10GB' -> ['10', 'GB']
      '80km/h' -> ['80', 'km/h']
    """
    if not token:
        return []
        
    # Split by explicit hyphens first
    if '-' in token:
        parts = []
        for p in token.split('-'):
            if p:
                parts.extend(split_mixed_token(p))
        return parts

    # Regex to split numbers and text "10GB" -> ["10", "GB"]
    # matches numbers (including decimals) and text separately
    # Updated to include Bangla digits ০-৯
    matches = re.finditer(r'([0-9০-৯.]+)|([a-zA-Z/]+)|([\u0980-\u09FF]+)', token)
    parts = [m.group(0) for m in matches if m.group(0)]
    
    # If no matches found through regex (maybe special chars), return original token
    if not parts:
        return [token]
        
    return parts

if __name__ == "__main__":
    # Test cases
    print(split_mixed_token('modelটা'))
    print(split_mixed_token('AI-based'))
    print(split_mixed_token('10GB'))
    print(split_mixed_token('80km/h'))
