import re
from typing import List, Dict

YES_SET = {"y", "yes", "true", "1"}
NO_SET  = {"n", "no", "false", "0"}

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.strip("\"'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-/\.]", "", s)
    return s.strip()

def normalize_yesno(ans: str) -> str:
    a = normalize_text(ans)
    if a in YES_SET:
        return "yes"
    if a in NO_SET:
        return "no"
    return a

def simple_tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []

def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counter.items():
        if c >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab

def encode(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens][:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids
