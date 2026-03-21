import json
import logging
import random
from pathlib import Path
from typing import List

from .encoder import EncodedMemory
from .config import load_config

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience-replay buffer with importance-based sampling
    and JSONL-backed persistence.
    """

    def __init__(self):
        cfg = load_config()
        self._max_size: int = cfg.max_buffer_size
        self._importance_threshold: float = cfg.importance_threshold
        self._buffer_path: Path = cfg.buffer_path
        self._buffer: List[EncodedMemory] = []
        self._load()

    # ---- persistence ----

    def _load(self):
        if not self._buffer_path or not self._buffer_path.exists():
            return
        try:
            with open(self._buffer_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._buffer.append(EncodedMemory.from_dict(json.loads(line)))
            logger.info("加载 %d 条记忆到缓冲区", len(self._buffer))
        except Exception as e:
            logger.warning("加载缓冲区失败: %s", e)
            self._buffer = []

    def _append_to_disk(self, memory: EncodedMemory):
        if not self._buffer_path:
            return
        self._buffer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._buffer_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(memory.to_dict(), ensure_ascii=False) + "\n")

    def _rewrite_disk(self):
        if not self._buffer_path:
            return
        self._buffer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._buffer_path, "w", encoding="utf-8") as f:
            for m in self._buffer:
                f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")

    # ---- core API ----

    def add(self, memory: EncodedMemory):
        keep = (memory.importance >= self._importance_threshold
                or random.random() < 0.3)
        if not keep:
            logger.debug("低重要性记忆被丢弃 (%.2f)", memory.importance)
            return

        self._buffer.append(memory)
        self._append_to_disk(memory)

        if len(self._buffer) > self._max_size:
            self._evict()

    def sample(self, batch_size: int = 32) -> List[EncodedMemory]:
        """Importance-weighted sampling for experience replay."""
        if not self._buffer:
            return []
        if len(self._buffer) <= batch_size:
            return list(self._buffer)

        weights = [m.importance for m in self._buffer]
        total = sum(weights)
        if total == 0:
            return random.sample(self._buffer, batch_size)

        chosen: set = set()
        while len(chosen) < batch_size:
            idx = random.choices(range(len(self._buffer)), weights=weights, k=1)[0]
            chosen.add(idx)
        return [self._buffer[i] for i in chosen]

    def get_important_memories(self, threshold: float = 0.8) -> List[EncodedMemory]:
        return [m for m in self._buffer if m.importance >= threshold]

    def size(self) -> int:
        return len(self._buffer)

    def stats(self) -> dict:
        if not self._buffer:
            return {"total": 0, "avg_importance": 0.0,
                    "max_importance": 0.0, "min_importance": 0.0}
        imps = [m.importance for m in self._buffer]
        return {
            "total": len(self._buffer),
            "avg_importance": round(sum(imps) / len(imps), 4),
            "max_importance": round(max(imps), 4),
            "min_importance": round(min(imps), 4),
        }

    # ---- internal ----

    def _evict(self):
        self._buffer.sort(key=lambda m: m.importance, reverse=True)
        self._buffer = self._buffer[: self._max_size]
        self._rewrite_disk()
        logger.info("缓冲区溢出，淘汰低重要性记忆，剩余 %d", len(self._buffer))


replay_buffer = ReplayBuffer()
