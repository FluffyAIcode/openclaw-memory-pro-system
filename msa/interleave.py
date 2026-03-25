"""
MemoryInterleave — MSA's multi-hop retrieval-generation loop.

Each round: route → expand context → generate intermediate answer.
If the intermediate answer is insufficient, reformulate the query
and run another round. Accumulates context across rounds for
cross-document reasoning.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

try:
    import llm_client as _llm_client
except ImportError:
    _llm_client = None

from .config import load_config
from .encoder import ChunkEncoder
from .memory_bank import MemoryBank
from .router import ScoredDocument, SparseRouter

logger = logging.getLogger(__name__)

INSUFFICIENT_MARKERS = [
    "i don't know",
    "i'm not sure",
    "no information",
    "cannot determine",
    "insufficient",
    "not enough context",
    "unable to answer",
]


@dataclass
class InterleaveRound:
    round_num: int
    query: str
    docs_retrieved: List[ScoredDocument]
    context_size: int
    intermediate_answer: str


@dataclass
class InterleaveResult:
    final_answer: str
    rounds: List[InterleaveRound]
    total_docs_used: int
    doc_ids_used: List[str]


def _default_generate_fn(query: str, context: str) -> str:
    """Default LLM generation via configured provider (OpenRouter / xAI)."""
    if _llm_client and _llm_client.is_available():
        result = _llm_client.generate(
            prompt=(
                f"基于以下文档上下文回答问题。如果信息不足以完整回答，"
                f"请明确指出哪些信息缺失。\n\n"
                f"文档上下文：\n{context[:6000]}\n\n"
                f"问题：{query}"
            ),
            system=(
                "你是知识检索助手。基于给定的文档上下文回答问题。"
                "如果上下文不足，说 '信息不足' 并指出缺失的部分。"
            ),
            max_tokens=1024,
        )
        if result:
            return result
    return context


class MemoryInterleave:
    """
    Multi-hop retrieval-generation loop inspired by MSA's Memory Interleave.

    Uses xAI Grok API for intermediate generation by default.
    Falls back to context passthrough if LLM is unavailable.
    """

    def __init__(self, encoder: ChunkEncoder, memory_bank: MemoryBank,
                 router: SparseRouter, config=None,
                 generate_fn: Optional[Callable[[str, str], str]] = None):
        self.encoder = encoder
        self.memory_bank = memory_bank
        self.router = router
        self.config = config or load_config()
        self.generate_fn = generate_fn or _default_generate_fn

    def run(self, query: str,
            max_rounds: Optional[int] = None,
            top_k: Optional[int] = None) -> InterleaveResult:
        max_rounds = max_rounds or self.config.max_interleave_rounds
        top_k = top_k or self.config.top_k

        context_doc_ids: Set[str] = set()
        all_context_chunks: Dict[str, List[str]] = {}
        rounds: List[InterleaveRound] = []
        current_query = query

        for round_num in range(1, max_rounds + 1):
            query_embedding = self.encoder.encode_query(current_query)
            scored_docs = self.router.route(query_embedding, top_k=top_k)

            new_docs = [sd for sd in scored_docs if sd.doc_id not in context_doc_ids]
            for sd in new_docs:
                context_doc_ids.add(sd.doc_id)
                chunks, _ = self.memory_bank.load_document_content(sd.doc_id)
                all_context_chunks[sd.doc_id] = chunks

            assembled = self._assemble_context(all_context_chunks)

            if self.generate_fn is not None:
                intermediate = self.generate_fn(current_query, assembled)
            else:
                intermediate = assembled

            round_info = InterleaveRound(
                round_num=round_num,
                query=current_query,
                docs_retrieved=scored_docs,
                context_size=len(assembled),
                intermediate_answer=intermediate[:500],
            )
            rounds.append(round_info)

            logger.info("Interleave round %d: %d docs retrieved, %d total context docs",
                        round_num, len(scored_docs), len(context_doc_ids))

            if self._is_sufficient(query, intermediate) or not new_docs:
                break

            if round_num < max_rounds:
                current_query = self._reformulate(query, intermediate)

        return InterleaveResult(
            final_answer=intermediate,
            rounds=rounds,
            total_docs_used=len(context_doc_ids),
            doc_ids_used=list(context_doc_ids),
        )

    def _assemble_context(self, doc_chunks: Dict[str, List[str]]) -> str:
        parts = []
        for doc_id, chunks in doc_chunks.items():
            meta = self.memory_bank.get_document_meta(doc_id)
            title = meta.metadata.get("title", doc_id) if meta else doc_id
            parts.append(f"=== Document: {title} ===")
            parts.append("\n".join(chunks))
            parts.append("")
        return "\n".join(parts)

    def _is_sufficient(self, query: str, answer: str) -> bool:
        """LLM-powered sufficiency check with keyword-based fallback."""
        if not answer or len(answer.strip()) < 20:
            return False
        try:
            if _llm_client and _llm_client.is_available():
                result = _llm_client.generate(
                    prompt=(
                        f"Question: {query}\n"
                        f"Current answer: {answer[:1000]}\n\n"
                        f"Does this answer adequately address the question? "
                        f"Reply only YES or NO."
                    ),
                    system="You are an answer quality judge. Reply only YES or NO.",
                    max_tokens=5,
                )
                if result:
                    return "YES" in result.strip().upper()
        except Exception as e:
            logger.debug("LLM sufficiency check failed, using fallback: %s", e)
        answer_lower = answer.lower()
        return not any(marker in answer_lower for marker in INSUFFICIENT_MARKERS)

    def _reformulate(self, original_query: str, intermediate: str) -> str:
        """LLM-powered query reformulation with string-concat fallback."""
        try:
            if _llm_client and _llm_client.is_available():
                result = _llm_client.generate(
                    prompt=(
                        f"Original question: {original_query}\n"
                        f"Information gathered so far: {intermediate[:800]}\n\n"
                        f"Generate a follow-up query to fill in the missing information. "
                        f"Output only the query, nothing else."
                    ),
                    system="You are a query refinement assistant. Output only the refined query.",
                    max_tokens=100,
                )
                if result and len(result.strip()) > 5:
                    return result.strip()
        except Exception as e:
            logger.debug("LLM reformulation failed, using fallback: %s", e)
        snippet = intermediate[:200].strip()
        return f"{original_query} (considering: {snippet})"
