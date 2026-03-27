import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import load_config
from .embedder import embedder

logger = logging.getLogger(__name__)
config = load_config()


class VectorStore:
    """Hybrid vector store: dense (cosine) + sparse (BM25) with RRF fusion.

    Stores entries as a JSON-lines file under config.vector_db_path.
    Search uses hybrid retrieval: bi-encoder cosine similarity fused with
    BM25 keyword scoring via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or config.vector_db_path
        self._index_file = self._db_path / "entries.jsonl"
        self._entries: Optional[List[dict]] = None
        self._bm25 = None
        self._bm25_dirty = True
        self._lock = threading.Lock()

    def _ensure_dir(self):
        self._db_path.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[dict]:
        if self._entries is not None:
            return self._entries
        self._entries = []
        if self._index_file.exists():
            with open(self._index_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning("跳过损坏的向量条目")
        logger.debug("加载了 %d 条向量条目", len(self._entries))
        self._bm25_dirty = True
        return self._entries

    def _ensure_bm25(self):
        """Build or rebuild the BM25 index when entries have changed."""
        if not self._bm25_dirty and self._bm25 is not None:
            return
        entries = self._load()
        if not entries:
            self._bm25 = None
            return
        try:
            from bm25 import BM25Index
            self._bm25 = BM25Index()
            docs = [e.get("content", "") for e in entries]
            self._bm25.build(docs)
            self._bm25_dirty = False
            logger.debug("BM25 index built: %d documents", len(docs))
        except Exception as e:
            logger.debug("BM25 index build failed: %s", e)
            self._bm25 = None

    def contains(self, content: str) -> bool:
        """Check if content (by first 80 chars) already exists in the store."""
        entries = self._load()
        prefix = content.strip()[:80]
        for e in entries:
            if e.get("content", "").strip()[:80] == prefix:
                return True
        return False

    def add(self, content: str, metadata: dict = None, dedup: bool = True):
        self._ensure_dir()
        with self._lock:
            entries = self._load()

            if dedup and self.contains(content):
                logger.debug("Skipped duplicate: %s", content[:60])
                return

            vector = embedder.embed_document(content)
            entry = {
                "content": content,
                "vector": vector,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
            entries.append(entry)
            self._bm25_dirty = True

            with open(self._index_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            logger.info("已存入向量库 (共 %d 条)", len(entries))

    def search(self, query: str, limit: int = 8,
               min_score: float = 0.0) -> List[dict]:
        """Hybrid search: dense cosine + BM25 sparse, fused via RRF."""
        entries = self._load()
        if not entries:
            return []

        dense_ranking = self._dense_search(query, entries, limit * 3)

        self._ensure_bm25()
        sparse_ranking = []
        if self._bm25 is not None:
            sparse_ranking = self._bm25.search(query, top_k=limit * 3)

        if sparse_ranking:
            try:
                from bm25 import reciprocal_rank_fusion
                fused = reciprocal_rank_fusion(dense_ranking, sparse_ranking, k=60)
            except ImportError:
                fused = dense_ranking
        else:
            fused = dense_ranking

        results = []
        for fused_score, idx in fused[:limit]:
            if idx >= len(entries):
                continue
            e = entries[idx]

            dense_score = 0.0
            for ds, di in dense_ranking:
                if di == idx:
                    dense_score = ds
                    break

            if min_score > 0 and dense_score < min_score and not any(si == idx for _, si in sparse_ranking[:limit]):
                continue

            final_score = dense_score if not sparse_ranking else fused_score
            results.append({
                "content": e["content"],
                "timestamp": e["timestamp"],
                "score": round(final_score, 4),
                "dense_score": round(dense_score, 4),
                "metadata": e.get("metadata", {}),
            })

        return results

    def search_dense_only(self, query: str, limit: int = 8,
                          min_score: float = 0.0) -> List[dict]:
        """Pure dense search (for benchmarking / comparison)."""
        entries = self._load()
        if not entries:
            return []

        ranking = self._dense_search(query, entries, limit * 2)
        results = []
        for score, idx in ranking[:limit]:
            if score < min_score:
                continue
            e = entries[idx]
            results.append({
                "content": e["content"],
                "timestamp": e["timestamp"],
                "score": round(score, 4),
                "metadata": e.get("metadata", {}),
            })
        return results

    def _dense_search(self, query: str, entries: list,
                      top_k: int) -> List[tuple]:
        """Bi-encoder cosine similarity search, returns (score, index) pairs."""
        query_vec = embedder.embed_query(query)
        scored = []
        for i, e in enumerate(entries):
            vec = e.get("vector")
            if vec and len(vec) == len(query_vec):
                score = self._cosine_sim(query_vec, vec)
            else:
                score = 0.0
            scored.append((score, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_sim(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def count(self) -> int:
        return len(self._load())


vector_store = VectorStore()
