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


class MemoraBridge:
    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or config.base_dir

    def save_to_both(self, content: str, source: str = "openclaw", importance: float = 0.7):
        """同时写入 Memora 和 OpenClaw 记忆系统"""
        entry = collector.collect(content, source=source, importance=importance)

        vector_store.add(content, metadata={
            "source": source,
            "importance": importance,
            "timestamp": entry["timestamp"],
        })

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        today_file = self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        with open(today_file, "a", encoding="utf-8") as f:
            f.write(f"\n### {datetime.now().strftime('%H:%M:%S')} [Memora]\n{content}\n")

        logger.info("双写成功 → OpenClaw + Memora")
        return entry

    def search_across(self, query: str):
        """跨系统搜索"""
        return vector_store.search(query, limit=8)

    def auto_digest(self):
        """自动执行记忆提炼（可被 OpenClaw Heartbeat 调用）"""
        logger.info("MemoraBridge: 执行自动记忆提炼")
        digest_memories(days=config.digest_interval_days)
        logger.info("自动提炼完成")


bridge = MemoraBridge()
