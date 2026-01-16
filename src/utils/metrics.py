from typing import List, Tuple, Dict
from .text import normalize_text

def accuracy(pairs: List[Tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    correct = 0
    for p, g in pairs:
        if normalize_text(p) == normalize_text(g):
            correct += 1
    return correct / len(pairs)

def bleu_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute BLEU-1/2/4 with NLTK + smoothing, averaged over samples."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except Exception:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

    smoothie = SmoothingFunction().method1

    def tok(s: str) -> List[str]:
        s = normalize_text(s)
        return s.split() if s else []

    n = len(preds)
    if n == 0:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

    b1 = b2 = b4 = 0.0
    for p, r in zip(preds, refs):
        p_tok = tok(p)
        r_tok = tok(r)
        if not r_tok:
            continue
        b1 += sentence_bleu([r_tok], p_tok, weights=(1,0,0,0), smoothing_function=smoothie)
        b2 += sentence_bleu([r_tok], p_tok, weights=(0.5,0.5,0,0), smoothing_function=smoothie)
        b4 += sentence_bleu([r_tok], p_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)

    return {"bleu1": b1/n, "bleu2": b2/n, "bleu4": b4/n}

def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, m+1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[m]

def rouge_l(preds: List[str], refs: List[str]) -> float:
    """ROUGE-L F1 averaged (token-based)."""
    def tok(s: str) -> List[str]:
        s = normalize_text(s)
        return s.split() if s else []

    scores = []
    for p, r in zip(preds, refs):
        p_t = tok(p); r_t = tok(r)
        if not p_t or not r_t:
            scores.append(0.0)
            continue
        lcs = _lcs_len(p_t, r_t)
        prec = lcs / len(p_t)
        rec = lcs / len(r_t)
        scores.append((2*prec*rec)/(prec+rec) if (prec+rec) else 0.0)
    return sum(scores)/len(scores) if scores else 0.0
