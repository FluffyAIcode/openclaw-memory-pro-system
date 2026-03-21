"""
MSASystem — Top-level orchestrator for the MSA memory pipeline.

Three-stage pipeline:
  Stage 1 (Offline): encode document → store routing key + content
  Stage 2 (Online):  route query → select top-k documents → load content
  Stage 3 (Generate): assemble sparse context → generate answer
  + Memory Interleave for multi-hop reasoning
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config
from .encoder import ChunkEncoder, EncodedDocument
from .interleave import InterleaveResult, MemoryInterleave
from .memory_bank import MemoryBank
from .router import ScoredDocument, SparseRouter

logger = logging.getLogger(__name__)


class MSASystem:
    """
    Memory Sparse Attention system.

    Adapts MSA's three-stage inference pipeline as an external memory layer:
    document-level routing via chunk-mean pooled representations, sparse Top-k
    selection, and Memory Interleave for multi-hop reasoning.
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self.encoder = ChunkEncoder(config=self.config)
        self.memory_bank = MemoryBank(config=self.config)
        self.router = SparseRouter(self.memory_bank, config=self.config)
        self.interleave = MemoryInterleave(
            self.encoder, self.memory_bank, self.router, config=self.config
        )
        self._initialized = False
        self._ingest_count = 0

    def initialize(self):
        if self._initialized:
            return
        self.config.ensure_dirs()
        self._load_state()
        self._initialized = True
        logger.info("MSA Memory System v1.0 initialized (%d documents)",
                     self.memory_bank.document_count())

    def _load_state(self):
        state_path = self.config.state_path
        if state_path and state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                self._ingest_count = state.get("ingest_count", 0)

    def _save_state(self):
        state_path = self.config.state_path
        if state_path:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "ingest_count": self._ingest_count,
                "document_count": self.memory_bank.document_count(),
                "last_updated": datetime.now().isoformat(),
            }
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

    def ingest(self, text: str, doc_id: Optional[str] = None,
               metadata: Optional[dict] = None) -> EncodedDocument:
        """Stage 1: Encode document and store in memory bank."""
        if not self._initialized:
            self.initialize()

        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        encoded = self.encoder.encode_document(doc_id, text, metadata)

        self.memory_bank.add_document(
            doc_id=encoded.doc_id,
            routing_key=encoded.routing_key,
            chunks=encoded.chunks,
            chunk_embeddings=encoded.chunk_embeddings,
            metadata=encoded.metadata,
        )
        self.memory_bank.flush()

        self._ingest_count += 1
        self._save_state()

        logger.info("Ingested document '%s' (%d chunks)", doc_id, len(encoded.chunks))
        return encoded

    def batch_ingest(self, documents: List[Dict]) -> List[EncodedDocument]:
        """Batch version of ingest. Each dict needs 'text' and optional 'doc_id', 'metadata'."""
        results = []
        for doc in documents:
            encoded = self.ingest(
                text=doc["text"],
                doc_id=doc.get("doc_id"),
                metadata=doc.get("metadata"),
            )
            results.append(encoded)
        return results

    def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """Stage 2+3: Route query → select documents → assemble context."""
        if not self._initialized:
            self.initialize()

        query_embedding = self.encoder.encode_query(question)
        scored_docs = self.router.route(query_embedding, top_k=top_k)

        context_parts = []
        for sd in scored_docs:
            chunks, _ = self.memory_bank.load_document_content(sd.doc_id)
            meta = self.memory_bank.get_document_meta(sd.doc_id)
            title = meta.metadata.get("title", sd.doc_id) if meta else sd.doc_id
            context_parts.append({
                "doc_id": sd.doc_id,
                "title": title,
                "score": sd.score,
                "rank": sd.rank,
                "chunks": chunks,
                "chunk_count": len(chunks),
            })

        return {
            "question": question,
            "results": context_parts,
            "total_results": len(context_parts),
            "total_documents": self.memory_bank.document_count(),
        }

    def interleave_query(self, question: str,
                         max_rounds: Optional[int] = None) -> InterleaveResult:
        """Multi-hop query via Memory Interleave."""
        if not self._initialized:
            self.initialize()

        return self.interleave.run(question, max_rounds=max_rounds)

    def remove(self, doc_id: str) -> bool:
        if not self._initialized:
            self.initialize()

        removed = self.memory_bank.remove_document(doc_id)
        if removed:
            self.memory_bank.flush()
            self._save_state()
            logger.info("Removed document '%s'", doc_id)
        return removed

    def status(self) -> dict:
        if not self._initialized:
            self.initialize()
        bank_stats = self.memory_bank.stats()
        return {
            "initialized": self._initialized,
            "ingest_count": self._ingest_count,
            "document_count": bank_stats["document_count"],
            "total_chunks": bank_stats["total_chunks"],
            "routing_matrix_shape": bank_stats["routing_matrix_shape"],
            "top_k": self.config.top_k,
            "chunk_size": self.config.chunk_size,
            "similarity_threshold": self.config.similarity_threshold,
            "max_interleave_rounds": self.config.max_interleave_rounds,
        }

    def report(self) -> dict:
        if not self._initialized:
            self.initialize()
        st = self.status()
        st["documents"] = []
        for doc_id in self.memory_bank.get_doc_ids():
            meta = self.memory_bank.get_document_meta(doc_id)
            if meta:
                st["documents"].append({
                    "doc_id": meta.doc_id,
                    "chunk_count": meta.chunk_count,
                    "ingested_at": meta.ingested_at,
                    "metadata": meta.metadata,
                })
        return st


msa_system = MSASystem()
