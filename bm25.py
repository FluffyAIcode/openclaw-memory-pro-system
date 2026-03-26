"""
BM25 sparse retrieval — zero-dependency implementation.

Used as the sparse arm of hybrid search (BM25 + dense vector).
Supports Chinese text via character n-gram tokenization as fallback.

BM25 parameters follow Elasticsearch defaults: k1=1.2, b=0.75
"""

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_STOPWORDS_EN = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "about", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "and", "but",
    "or", "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
})

_STOPWORDS_ZH = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它",
})

_CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')


def tokenize(text: str) -> List[str]:
    """Simple tokenizer supporting mixed CJK + Latin text.

    - CJK characters: bigram tokenization (overlapping 2-char windows)
    - Latin words: whitespace + punctuation split, lowercased
    """
    text = text.lower()
    text = re.sub(r'[^\w\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', ' ', text)

    tokens = []
    cjk_buffer = []

    for char in text:
        if _CJK_RANGE.match(char):
            cjk_buffer.append(char)
        else:
            if cjk_buffer:
                tokens.extend(_cjk_bigrams(cjk_buffer))
                cjk_buffer = []
            if char.strip():
                pass  # handled below

    if cjk_buffer:
        tokens.extend(_cjk_bigrams(cjk_buffer))

    words = re.findall(r'[a-z0-9]+', text)
    for w in words:
        if w not in _STOPWORDS_EN and len(w) > 1:
            tokens.append(w)

    return tokens


def _cjk_bigrams(chars: List[str]) -> List[str]:
    """Generate bigrams from CJK character sequence, filtering stopwords."""
    result = []
    for c in chars:
        if c not in _STOPWORDS_ZH:
            result.append(c)
    bigrams = []
    if len(result) == 1:
        bigrams.append(result[0])
    else:
        for i in range(len(result) - 1):
            bigrams.append(result[i] + result[i + 1])
        if result:
            bigrams.append(result[-1])
    return bigrams


class BM25Index:
    """In-memory BM25 index for a list of documents.

    Documents are represented as (doc_id, text) pairs.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_tokens: List[List[str]] = []
        self._doc_ids: List[int] = []
        self._doc_lens: List[int] = []
        self._avg_dl: float = 0.0
        self._df: Dict[str, int] = {}
        self._N: int = 0
        self._built = False

    def build(self, documents: List[str]) -> None:
        """Build BM25 index from a list of document texts."""
        self._doc_tokens = []
        self._doc_ids = list(range(len(documents)))
        self._df = {}

        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            self._doc_tokens.append(tokens)

            seen = set(tokens)
            for t in seen:
                self._df[t] = self._df.get(t, 0) + 1

        self._doc_lens = [len(t) for t in self._doc_tokens]
        self._N = len(documents)
        self._avg_dl = sum(self._doc_lens) / max(self._N, 1)
        self._built = True

    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, int]]:
        """Search the index, returning (score, doc_index) pairs sorted by score desc."""
        if not self._built or self._N == 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = [0.0] * self._N

        for qt in q_tokens:
            df = self._df.get(qt, 0)
            if df == 0:
                continue
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

            for i, doc_tokens in enumerate(self._doc_tokens):
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                scores[i] += idf * numerator / denominator

        scored = [(s, i) for i, s in enumerate(scores) if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


def reciprocal_rank_fusion(
    *rankings: List[Tuple[float, int]],
    k: int = 60,
) -> List[Tuple[float, int]]:
    """Reciprocal Rank Fusion (RRF) for merging multiple ranked lists.

    Each ranking is a list of (score, doc_index) pairs.
    Returns fused (rrf_score, doc_index) sorted by rrf_score desc.

    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each ranking i
    """
    fused: Dict[int, float] = {}

    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    result = [(score, idx) for idx, score in fused.items()]
    result.sort(key=lambda x: x[0], reverse=True)
    return result
