"""
Dynamic LoRA adapter manager.

Allocates virtual LoRA adapters based on memory importance and evicts
the least-important ones when capacity is reached.  When a real base
model is connected, these map to actual low-rank weight matrices;
until then they track importance metadata for the pipeline.
"""

import logging
from typing import Dict, List

from .encoder import EncodedMemory
from .config import load_config

logger = logging.getLogger(__name__)


class DynamicLoRA:

    def __init__(self):
        cfg = load_config()
        self._max: int = cfg.max_lora_adapters
        self._rank: int = cfg.lora_rank
        self._adapters: Dict[str, float] = {}
        self._total_allocated: int = 0

    def allocate(self, memory: EncodedMemory) -> str:
        adapter_id = memory.timestamp.isoformat()

        if len(self._adapters) >= self._max:
            lowest_key = min(self._adapters, key=self._adapters.get)
            if memory.importance <= self._adapters[lowest_key]:
                return ""
            del self._adapters[lowest_key]
            logger.debug("淘汰适配器 %s", lowest_key)

        self._adapters[adapter_id] = memory.importance
        self._total_allocated += 1
        logger.debug("分配 LoRA r=%d → %.3f", self._rank, memory.importance)
        return adapter_id

    def update(self, memories: List[EncodedMemory]):
        n = sum(1 for m in memories if self.allocate(m))
        logger.info("LoRA 更新 — 新 %d, 活跃 %d", n, len(self._adapters))

    def merge(self):
        count = len(self._adapters)
        self._adapters.clear()
        logger.info("LoRA 合并完成 — %d 适配器并入基础模型", count)

    @property
    def stats(self) -> dict:
        imps = list(self._adapters.values())
        return {
            "active_adapters": len(self._adapters),
            "max_adapters": self._max,
            "rank": self._rank,
            "total_allocated": self._total_allocated,
            "avg_importance": round(sum(imps) / len(imps), 4) if imps else 0.0,
        }


dynamic_lora = DynamicLoRA()
