"""
ChunkEncoder — MSA's chunk-mean pooling implementation.

Splits documents into overlapping chunks, embeds each chunk,
then produces a single routing key via mean-pooling over chunk embeddings.
This is the core MSA concept: K_r = mean(embed(chunk_i) for i in chunks).
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import load_config

logger = logging.getLogger(__name__)


@dataclass
class EncodedDocument:
    doc_id: str
    routing_key: np.ndarray
    chunks: List[str]
    chunk_embeddings: np.ndarray
    metadata: dict = field(default_factory=dict)


class MockEmbedder:
    """Hash-based pseudo-embeddings for structural testing.
    Same text produces same vector; different text produces different vectors.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = []
        for i in range(0, min(len(digest), self.dimension * 2), 2):
            raw.append(int(digest[i: i + 2], 16) / 255.0)
        while len(raw) < self.dimension:
            raw.append(0.5)
        vec = np.array(raw[:self.dimension], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts], dtype=np.float32)


class SentenceTransformerEmbedder:
    """Real embedder using sentence-transformers. Lazy-loaded.

    nomic-embed-text requires task prefixes for best quality:
      - "search_document: " for stored content
      - "search_query: "    for search queries
    """

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info("Loaded SentenceTransformer: %s", self.model_name)

    def embed(self, text: str, prefix: str = "search_document") -> np.ndarray:
        self._load()
        prefixed = f"{prefix}: {text}"
        return self._model.encode(prefixed, normalize_embeddings=True).astype(np.float32)

    def embed_document(self, text: str) -> np.ndarray:
        return self.embed(text, prefix="search_document")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(text, prefix="search_query")

    def embed_batch(self, texts: List[str], prefix: str = "search_document") -> np.ndarray:
        self._load()
        prefixed = [f"{prefix}: {t}" for t in texts]
        return self._model.encode(prefixed, normalize_embeddings=True).astype(np.float32)


def _create_embedder(config):
    try:
        import shared_embedder
        shared = shared_embedder.get()
        if shared is not None:
            logger.info("Using shared embedder from Memory Server")
            return shared
    except ImportError:
        pass

    try:
        emb = SentenceTransformerEmbedder(config.embedding_model, config.embedding_dimension)
        emb._load()
        return emb
    except Exception as e:
        logger.warning("SentenceTransformer unavailable (%s), using MockEmbedder", e)
        return MockEmbedder(config.embedding_dimension)


class ChunkEncoder:
    """
    Splits text into overlapping chunks, embeds each, then mean-pools
    all chunk embeddings into a single routing key (K_r).
    """

    def __init__(self, embedder=None, config=None):
        self.config = config or load_config()
        self.embedder = embedder or _create_embedder(self.config)

    def split_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += chunk_size - overlap
        return chunks

    def encode_document(self, doc_id: str, text: str,
                        metadata: Optional[dict] = None) -> EncodedDocument:
        chunks = self.split_into_chunks(text)
        chunk_embeddings = self.embedder.embed_batch(chunks)

        # chunk-mean pooling: K_r = mean(chunk embeddings), then L2 normalize
        routing_key = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(routing_key)
        if norm > 0:
            routing_key = routing_key / norm

        return EncodedDocument(
            doc_id=doc_id,
            routing_key=routing_key,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            metadata=metadata or {},
        )

    def encode_query(self, query: str) -> np.ndarray:
        embed_q = self.embedder.embed_query if hasattr(self.embedder, 'embed_query') else self.embedder.embed
        return embed_q(query)
