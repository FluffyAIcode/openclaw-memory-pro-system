"""
MSA <-> OpenClaw bridge.

Provides triple-write: documents are ingested into the MSA sparse routing
pipeline, indexed into Memora for snippet-level search, and appended to
OpenClaw's memory/ daily files for file-based access.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .system import msa_system
from .config import load_config

logger = logging.getLogger(__name__)


class MSABridge:

    def __init__(self, memory_dir: Path = None):
        workspace = Path(__file__).parent.parent
        self.memory_dir = memory_dir or (workspace / "memory")

    def ingest_and_save(self, content: str, source: str = "openclaw",
                        doc_id: Optional[str] = None,
                        metadata: Optional[dict] = None,
                        cross_index: bool = True,
                        write_daily: bool = True) -> dict:
        """
        Multi-write:
        1. MSA pipeline (encode → chunk-mean pool → store routing key + content)
        2. OpenClaw memory/ daily file (skipped when caller handles it)
        3. Memora vector store (snippet-level indexing for each chunk)
        """
        meta = metadata or {}
        meta["source"] = source

        encoded = msa_system.ingest(content, doc_id=doc_id, metadata=meta)

        if write_daily:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            today = self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
            preview = content[:500] + "..." if len(content) > 500 else content
            with open(today, "a", encoding="utf-8") as f:
                f.write(f"\n### {datetime.now().strftime('%H:%M:%S')} "
                        f"[MSA/{source}]\n{preview}\n")

        # Memora 交叉索引：将每个 chunk 写入 Memora 向量库以支持片段级搜索
        if cross_index:
            try:
                from memora.vectorstore import vector_store
                title = meta.get("title", encoded.doc_id)
                for i, chunk in enumerate(encoded.chunks):
                    vector_store.add(chunk, metadata={
                        "source": f"msa/{source}",
                        "msa_doc_id": encoded.doc_id,
                        "chunk_index": i,
                        "title": title,
                    })
                logger.info("Memora cross-index: %d chunks from %s",
                            len(encoded.chunks), encoded.doc_id)
            except Exception as e:
                logger.warning("Memora cross-index failed: %s", e)

        logger.info("Triple-write: MSA + Memora + OpenClaw memory/ (%s)", encoded.doc_id)
        return {
            "doc_id": encoded.doc_id,
            "chunks": len(encoded.chunks),
            "source": source,
        }

    def query_memory(self, question: str, top_k: Optional[int] = None) -> Dict:
        return msa_system.query(question, top_k=top_k)

    def query_with_memora(self, question: str, top_k: Optional[int] = None) -> Dict:
        """Merged query: MSA document-level routing + Memora snippet search."""
        msa_result = msa_system.query(question, top_k=top_k)

        try:
            from memora.vectorstore import vector_store
            memora_results = vector_store.search(question, limit=5)
            msa_result["memora_snippets"] = memora_results
        except Exception as e:
            logger.warning("Memora search in merged query failed: %s", e)
            msa_result["memora_snippets"] = []

        return msa_result

    def interleave_query(self, question: str,
                         max_rounds: Optional[int] = None) -> Dict:
        result = msa_system.interleave_query(question, max_rounds=max_rounds)
        return {
            "final_answer": result.final_answer,
            "rounds": len(result.rounds),
            "total_docs_used": result.total_docs_used,
            "doc_ids_used": result.doc_ids_used,
        }

    def remove(self, doc_id: str) -> bool:
        return msa_system.remove(doc_id)

    def status(self) -> dict:
        return msa_system.status()

    def report(self) -> dict:
        return msa_system.report()


bridge = MSABridge()
