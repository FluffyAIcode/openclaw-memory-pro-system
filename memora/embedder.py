"""
文本嵌入器 — 优先使用共享 embedder（Memory Server 注入），
其次 SentenceTransformer，最后回退到 MockEmbedder。
"""

import hashlib
import logging
import math
from typing import List

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()


class MockEmbedder:
    """基于哈希的伪向量，仅当 SentenceTransformer 不可用时使用。"""

    def __init__(self, dimension: int = None):
        self.dimension = dimension or config.embedding_dimension

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = []
        for i in range(0, min(len(digest), self.dimension * 2), 2):
            raw.append(int(digest[i: i + 2], 16) / 255.0)
        while len(raw) < self.dimension:
            raw.append(0.5)
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            raw = [x / norm for x in raw]
        return raw[:self.dimension]


class SentenceTransformerEmbedder:
    """真实嵌入模型，lazy-load 避免启动时阻塞。

    nomic-embed-text requires task prefixes for best quality:
      - "search_document: " for stored content
      - "search_query: "    for search queries
    """

    def __init__(self, model_name: str = None, dimension: int = None):
        self.model_name = model_name or config.embedding_model
        self.dimension = dimension or config.embedding_dimension
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info("Loaded SentenceTransformer: %s", self.model_name)

    def embed(self, text: str, prefix: str = "search_document") -> List[float]:
        self._load()
        prefixed = f"{prefix}: {text}"
        return self._model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text, prefix="search_query")

    def embed_document(self, text: str) -> List[float]:
        return self.embed(text, prefix="search_document")


def _create_embedder():
    try:
        import shared_embedder
        shared = shared_embedder.get()
        if shared is not None:
            logger.info("Using shared embedder from Memory Server")
            return shared
    except ImportError:
        pass

    try:
        emb = SentenceTransformerEmbedder()
        emb._load()
        return emb
    except Exception as e:
        logger.warning("SentenceTransformer unavailable (%s), using MockEmbedder", e)
        return MockEmbedder()


class _EmbedderProxy:
    """Lazy proxy — delays embedder creation until first use.
    Allows the Memory Server to inject a shared embedder before
    any embed() call happens.
    """
    _instance = None

    def _get(self):
        if self._instance is None:
            self._instance = _create_embedder()
        return self._instance

    def embed(self, text: str, prefix: str = "search_document") -> List[float]:
        inst = self._get()
        if hasattr(inst, "embed_document"):
            if prefix == "search_query":
                return inst.embed_query(text)
            return inst.embed_document(text)
        return inst.embed(text)

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text, prefix="search_query")

    def embed_document(self, text: str) -> List[float]:
        return self.embed(text, prefix="search_document")

    @property
    def dimension(self):
        return self._get().dimension


embedder = _EmbedderProxy()
