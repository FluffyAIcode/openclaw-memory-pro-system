"""
MemoryBank — Tiered storage for MSA routing keys and document content.

Hot tier: All routing keys loaded in RAM as a numpy matrix for fast batch cosine sim.
Cold tier: Per-document chunk content + embeddings stored on disk as JSONL files.

Inspired by MSA's "GPU-resident routing keys, CPU-resident content KV".
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import load_config

logger = logging.getLogger(__name__)


@dataclass
class DocumentMeta:
    doc_id: str
    routing_key: np.ndarray
    chunk_count: int
    metadata: dict = field(default_factory=dict)
    ingested_at: str = ""


class MemoryBank:
    """
    Two-tier memory bank:
    - routing_index.jsonl  → doc_id + routing_key vectors (loaded into RAM)
    - content/{doc_id}.jsonl → chunks + chunk_embeddings (loaded on demand)
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self._documents: Dict[str, DocumentMeta] = {}
        self._routing_matrix: Optional[np.ndarray] = None
        self._doc_ids_order: List[str] = []
        self._dirty = False
        self._load_routing_index()

    def _load_routing_index(self):
        idx_path = self.config.routing_keys_path
        if idx_path is None or not idx_path.exists():
            return

        with open(idx_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                doc_id = entry["doc_id"]
                routing_key = np.array(entry["routing_key"], dtype=np.float32)
                meta = DocumentMeta(
                    doc_id=doc_id,
                    routing_key=routing_key,
                    chunk_count=entry.get("chunk_count", 0),
                    metadata=entry.get("metadata", {}),
                    ingested_at=entry.get("ingested_at", ""),
                )
                self._documents[doc_id] = meta
                self._doc_ids_order.append(doc_id)

        self._rebuild_matrix()
        logger.info("Loaded %d routing keys from index", len(self._documents))

    def _rebuild_matrix(self):
        if not self._doc_ids_order:
            self._routing_matrix = None
            return
        keys = [self._documents[did].routing_key for did in self._doc_ids_order]
        self._routing_matrix = np.stack(keys)

    def add_document(self, doc_id: str, routing_key: np.ndarray,
                     chunks: List[str], chunk_embeddings: np.ndarray,
                     metadata: Optional[dict] = None):
        meta = DocumentMeta(
            doc_id=doc_id,
            routing_key=routing_key,
            chunk_count=len(chunks),
            metadata=metadata or {},
            ingested_at=datetime.now().isoformat(),
        )

        if doc_id in self._documents:
            idx = self._doc_ids_order.index(doc_id)
            self._documents[doc_id] = meta
        else:
            self._documents[doc_id] = meta
            self._doc_ids_order.append(doc_id)

        self._rebuild_matrix()
        self._save_content(doc_id, chunks, chunk_embeddings, metadata or {})
        self._dirty = True

    def _save_content(self, doc_id: str, chunks: List[str],
                      chunk_embeddings: np.ndarray, metadata: dict):
        content_dir = self.config.content_store_path
        content_dir.mkdir(parents=True, exist_ok=True)
        content_file = content_dir / f"{doc_id}.jsonl"

        with open(content_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                entry = {
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": chunk_embeddings[i].tolist(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_document_content(self, doc_id: str) -> Tuple[List[str], np.ndarray]:
        content_file = self.config.content_store_path / f"{doc_id}.jsonl"
        if not content_file.exists():
            raise FileNotFoundError(f"Content not found for document: {doc_id}")

        chunks = []
        embeddings = []
        with open(content_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                chunks.append(entry["text"])
                embeddings.append(entry["embedding"])

        return chunks, np.array(embeddings, dtype=np.float32)

    def get_all_routing_keys(self) -> Optional[np.ndarray]:
        return self._routing_matrix

    def get_doc_ids(self) -> List[str]:
        return list(self._doc_ids_order)

    def get_document_meta(self, doc_id: str) -> Optional[DocumentMeta]:
        return self._documents.get(doc_id)

    def document_count(self) -> int:
        return len(self._documents)

    def flush(self):
        if not self._dirty:
            return

        idx_path = self.config.routing_keys_path
        idx_path.parent.mkdir(parents=True, exist_ok=True)

        with open(idx_path, "w", encoding="utf-8") as f:
            for doc_id in self._doc_ids_order:
                meta = self._documents[doc_id]
                entry = {
                    "doc_id": meta.doc_id,
                    "routing_key": meta.routing_key.tolist(),
                    "chunk_count": meta.chunk_count,
                    "metadata": meta.metadata,
                    "ingested_at": meta.ingested_at,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._dirty = False
        logger.info("Flushed routing index (%d documents)", len(self._documents))

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self._documents:
            return False

        del self._documents[doc_id]
        self._doc_ids_order.remove(doc_id)
        self._rebuild_matrix()

        content_file = self.config.content_store_path / f"{doc_id}.jsonl"
        if content_file.exists():
            content_file.unlink()

        self._dirty = True
        return True

    def stats(self) -> dict:
        total_chunks = sum(m.chunk_count for m in self._documents.values())
        return {
            "document_count": len(self._documents),
            "total_chunks": total_chunks,
            "routing_matrix_shape": (
                self._routing_matrix.shape if self._routing_matrix is not None else None
            ),
            "index_file": str(self.config.routing_keys_path),
            "content_dir": str(self.config.content_store_path),
        }
