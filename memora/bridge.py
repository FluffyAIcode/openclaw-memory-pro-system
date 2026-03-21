"""
Memora <-> OpenClaw 记忆系统桥接器
"""

import logging
from datetime import datetime
from pathlib import Path

from .collector import collector
from .vectorstore import vector_store
from .digest import digest_memories
from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()

_MSA_WORD_THRESHOLD = 100


class MemoraBridge:
    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or config.base_dir

    def save_to_both(self, content: str, source: str = "openclaw", importance: float = 0.7):
        """同时写入 Memora 和 OpenClaw 记忆系统，并按条件转发到 Chronos 和 MSA"""
        entry = collector.collect(content, source=source, importance=importance)

        vector_store.add(content, metadata={
            "source": source,
            "importance": importance,
            "timestamp": entry["timestamp"],
        })

        # Chronos 集成：高重要性记忆实时内化（直接 import，避免 subprocess 开销）
        if importance >= 0.75:
            try:
                from chronos.bridge import bridge as chronos_bridge
                chronos_bridge.learn_and_save(
                    content, source=source, importance=importance, write_daily=False)
            except Exception as e:
                logger.warning("Chronos learn failed: %s", e)

        # MSA 集成：长文本自动摄入 MSA 文档级记忆库
        word_count = len(content.split())
        if word_count >= _MSA_WORD_THRESHOLD:
            try:
                from msa.bridge import bridge as msa_bridge
                msa_bridge.ingest_and_save(
                    content, source=source, cross_index=False, write_daily=False)
                logger.info("Long text forwarded to MSA (%d words)", word_count)
            except Exception as e:
                logger.warning("MSA ingest failed: %s", e)

        logger.info("Memora save complete → vectorstore + Chronos + MSA")
        return entry

    def search_across(self, query: str, include_msa: bool = True):
        """跨系统搜索：Memora 向量检索 + MSA 文档级路由"""
        memora_results = vector_store.search(query, limit=8)

        if include_msa:
            try:
                from msa.bridge import bridge as msa_bridge
                msa_result = msa_bridge.query_memory(query, top_k=3)
                for doc in msa_result.get("results", []):
                    memora_results.append({
                        "content": "\n".join(doc["chunks"][:2]),
                        "timestamp": "",
                        "score": round(doc["score"], 4),
                        "metadata": {"source": "msa", "doc_id": doc["doc_id"],
                                     "title": doc["title"]},
                    })
            except Exception as e:
                logger.warning("MSA cross-search failed: %s", e)

        memora_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return memora_results

    def auto_digest(self):
        """自动执行记忆提炼（可被 OpenClaw Heartbeat 调用）"""
        logger.info("MemoraBridge: executing auto digest")
        digest_memories(days=config.digest_interval_days)
        logger.info("Auto digest complete")


bridge = MemoraBridge()
