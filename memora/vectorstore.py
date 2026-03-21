import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import load_config
from .embedder import embedder

logger = logging.getLogger(__name__)
config = load_config()


class VectorStore:
    """JSON-backed vector store with cosine similarity search.

    Stores entries as a JSON-lines file under config.vector_db_path.
    When a real embedding model is available, search uses cosine similarity;
    otherwise falls back to recency-based results.
    """

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or config.vector_db_path
        self._index_file = self._db_path / "entries.jsonl"
        self._entries: Optional[List[dict]] = None

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
        return self._entries

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

        with open(self._index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("已存入向量库 (共 %d 条)", len(entries))

    def search(self, query: str, limit: int = 8) -> List[dict]:
        entries = self._load()
        if not entries:
            return []

        query_vec = embedder.embed_query(query)

        scored = []
        for e in entries:
            vec = e.get("vector")
            if vec and len(vec) == len(query_vec):
                score = self._cosine_sim(query_vec, vec)
            else:
                score = 0.0
            scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, e in scored[:limit]:
            results.append({
                "content": e["content"],
                "timestamp": e["timestamp"],
                "score": round(score, 4),
                "metadata": e.get("metadata", {}),
            })
        return results

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
