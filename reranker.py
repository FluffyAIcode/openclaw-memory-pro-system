"""
Reranker — Cross-encoder reranking for multi-stage retrieval.

Stage 1 (bi-encoder): fast cosine similarity retrieval → top-K candidates
Stage 2 (cross-encoder): precise relevance scoring → reranked results

Uses sentence-transformers CrossEncoder with ms-marco-MiniLM-L-6-v2 (22M params).
Falls back to pass-through if cross-encoder is unavailable.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_cross_encoder = None
_cross_encoder_loaded = False


def _get_cross_encoder():
    global _cross_encoder, _cross_encoder_loaded
    if _cross_encoder_loaded:
        return _cross_encoder
    _cross_encoder_loaded = True
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        _cross_encoder = model
        logger.info("CrossEncoder loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        logger.warning("CrossEncoder unavailable (%s), reranking disabled", e)
        _cross_encoder = None
    return _cross_encoder


def rerank(query: str, candidates: List[dict],
           content_key: str = "content",
           top_k: int = 0,
           score_key: str = "rerank_score") -> List[dict]:
    """Rerank candidates using cross-encoder.

    Args:
        query: The search query
        candidates: List of dicts with at least a content_key field
        content_key: Key to extract text from each candidate
        top_k: Max results to return (0 = return all, reranked)
        score_key: Key to store the cross-encoder score in each result

    Returns:
        Reranked list of candidates with score_key added.
        Falls back to original order if cross-encoder is unavailable.
    """
    if not candidates:
        return candidates

    ce = _get_cross_encoder()
    if ce is None:
        return candidates[:top_k] if top_k > 0 else candidates

    pairs = []
    for c in candidates:
        text = c.get(content_key, "")
        if not text:
            text = str(c)
        pairs.append((query, text[:512]))

    try:
        scores = ce.predict(pairs)
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, candidate in scored:
            candidate = dict(candidate)
            candidate[score_key] = round(float(score), 4)
            results.append(candidate)

        if top_k > 0:
            results = results[:top_k]

        return results
    except Exception as e:
        logger.warning("Cross-encoder rerank failed: %s", e)
        return candidates[:top_k] if top_k > 0 else candidates


def rerank_pairs(query: str, texts: List[str]) -> List[Tuple[float, int]]:
    """Score query-text pairs and return sorted (score, original_index)."""
    ce = _get_cross_encoder()
    if ce is None:
        return [(0.0, i) for i in range(len(texts))]

    pairs = [(query, t[:512]) for t in texts]
    try:
        scores = ce.predict(pairs)
        indexed = [(float(s), i) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: x[0], reverse=True)
        return indexed
    except Exception as e:
        logger.warning("Cross-encoder scoring failed: %s", e)
        return [(0.0, i) for i in range(len(texts))]
