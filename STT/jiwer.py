"""
Minimal local shim providing `wer` and `cer` functions used by the STT runner.
This avoids requiring the external `jiwer` package when running locally.
The implementations are simple edit-distance based and adequate for evaluation.
"""
from typing import List

def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m]

def wer(references, hypotheses) -> float:
    """Compute word error rate between lists of reference and hypothesis strings.
    Returns a float in [0,1].
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    total_edits = 0
    total_words = 0
    for r, h in zip(references, hypotheses):
        r_words = r.strip().split()
        h_words = h.strip().split()
        total_edits += _levenshtein(r_words, h_words)
        total_words += max(1, len(r_words))
    return total_edits / total_words if total_words > 0 else 0.0

def cer(references, hypotheses) -> float:
    """Compute character error rate between lists of reference and hypothesis strings.
    Returns a float in [0,1].
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    total_edits = 0
    total_chars = 0
    for r, h in zip(references, hypotheses):
        r_chars = list(r.replace(' ', ''))
        h_chars = list(h.replace(' ', ''))
        total_edits += _levenshtein(r_chars, h_chars)
        total_chars += max(1, len(r_chars))
    return total_edits / total_chars if total_chars > 0 else 0.0
