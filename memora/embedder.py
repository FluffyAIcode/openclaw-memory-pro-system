"""
文本嵌入器 — 当前为 mock 实现，返回固定维度的哈希向量。
接入真实模型时替换 MockEmbedder 为 SentenceTransformerEmbedder。
"""

import hashlib
import logging
import math
from typing import List

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()


class MockEmbedder:
    """基于哈希的伪向量，仅供结构测试。
    相同文本会产生相同向量，不同文本向量不同，
    比固定 [0.1]*768 有意义得多。
    """

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


# TODO: 接入真实模型时取消注释并替换
# class SentenceTransformerEmbedder:
#     def __init__(self):
#         from sentence_transformers import SentenceTransformer
#         self.model = SentenceTransformer(config.embedding_model)
#
#     def embed(self, text: str) -> List[float]:
#         return self.model.encode(text).tolist()

embedder = MockEmbedder()
