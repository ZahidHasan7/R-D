import re

class Classifier:
    """
    Mock integration of the 'BanglishBERT classifier'.
    Since the actual ML model isn't active in this environment, this acts as a 
    heuristic-based stub simulating the behavior.
    """
    def __init__(self):
        pass
        
    def classify_token(self, token: str) -> str:
        """
        Classifies token into: ABBREVIATION, UNIT, MIXED, or NORMAL
        """
        if not token:
            return "NORMAL"
            
        # Detect MIXED
        if '-' in token or (re.search(r'\d', token) and re.search(r'[a-zA-Z]', token)):
            return "MIXED"
            
        # Detect UNIT
        if re.match(r'^\d*[A-Za-z/%]+$', token):
            if any(token.lower().endswith(u) for u in ['kg', 'gb', 'mb', 'ml', 'km', 'km/h', 'hz', '%']):
                return "UNIT"
                
        # Detect ABBREVIATION - ensure it's not a number before classifying
        if re.match(r'^[A-Z]{2,5}$', token) or token.lower() in ['cpu', 'api', 'nlp', 'ai', 'dr.', 'prof.']:
            return "ABBREVIATION"
            
        if '.' in token and re.match(r'^([\u0980-\u09FFa-zA-Z]+\.)+[\u0980-\u09FFa-zA-Z]*$', token):
            # Check if it starts with a currency symbol
            if token[0] in ['৳', '$', '€', '£', '¥', '₹', '৳']:
                return "NORMAL"
            # Ensure it's not purely numeric characters mixed with dots (handle both ASCII and Bangla)
            if re.match(r'^[0-9০-৯.]+$', token):
                return "NORMAL"
            # Check if the parts between dots are mostly digits
            parts = token.split('.')
            if all(re.match(r'^[0-9০-৯]+$', p) for p in parts if p):
                return "NORMAL"
            return "ABBREVIATION"

        return "NORMAL"

if __name__ == "__main__":
    c = Classifier()
    for t in ['10GB', 'cpu', 'Dr.', 'AI-based', 'hello']:
        print(f"{t}: {c.classify_token(t)}")
