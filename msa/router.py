"""
SparseRouter — MSA's sparse routing mechanism.

Performs batch cosine similarity between a query embedding and all
document routing keys, returning the top-k most relevant documents.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import load_config
from .memory_bank import MemoryBank

logger = logging.getLogger(__name__)


@dataclass
class ScoredDocument:
    doc_id: str
    score: float
    rank: int


class SparseRouter:
    """
    Batch cosine similarity over all routing keys.
    Routing keys are already L2-normalized, so dot product = cosine sim.
    """

    def __init__(self, memory_bank: MemoryBank, config=None):
        self.memory_bank = memory_bank
        self.config = config or load_config()

    def route(self, query_embedding: np.ndarray,
              top_k: Optional[int] = None,
              threshold: Optional[float] = None) -> List[ScoredDocument]:
        top_k = top_k or self.config.top_k
        threshold = threshold if threshold is not None else self.config.similarity_threshold

        routing_keys = self.memory_bank.get_all_routing_keys()
        if routing_keys is None or len(routing_keys) == 0:
            return []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # batch cosine similarity: (N, D) @ (D,) -> (N,)
        scores = routing_keys @ query_embedding

        doc_ids = self.memory_bank.get_doc_ids()
        actual_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-actual_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score < threshold:
                continue
            results.append(ScoredDocument(
                doc_id=doc_ids[idx],
                score=score,
                rank=rank + 1,
            ))

        logger.debug("Routed query to %d/%d documents (top_k=%d, threshold=%.2f)",
                      len(results), len(doc_ids), top_k, threshold)
        return results

    def route_multi(self, query_embeddings: np.ndarray,
                    top_k: Optional[int] = None) -> List[List[ScoredDocument]]:
        """Route multiple queries at once (batch)."""
        return [self.route(q, top_k=top_k) for q in query_embeddings]
