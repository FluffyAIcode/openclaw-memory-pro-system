"""
ChronosSystem — top-level orchestrator (refactored).

New pipeline: raw text → Encode → Buffer → Training Export (→ Nebius)

The old pipeline (Encode → Buffer → EWC → LoRA → Consolidate) has been
simplified: EWC and LoRA simulation are removed. Chronos now focuses on
collecting high-quality training candidates and preparing datasets for
cloud-based fine-tuning (Nebius / Axolotl).
"""

import logging
from datetime import datetime

from .encoder import encoder, EncodedMemory
from .replay_buffer import replay_buffer
from .consolidator import consolidator

logger = logging.getLogger(__name__)


class ChronosSystem:
    """
    Chronos Training Pipeline.

    Collects memories into a replay buffer, and periodically exports
    training datasets for cloud fine-tuning. The consolidator handles
    personality profile generation from accumulated memories.
    """

    def __init__(self):
        self._initialized = False
        self._learn_count = 0

    def initialize(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("Chronos 训练管线 v2.0 已初始化")

    def learn(self, raw_text: str, importance: float = None) -> EncodedMemory:
        """Encode text and add to replay buffer for future training."""
        if not self._initialized:
            self.initialize()

        encoded = encoder.encode(raw_text, importance=importance)
        replay_buffer.add(encoded)

        if consolidator.should_consolidate():
            consolidator.consolidate()

        self._learn_count += 1
        logger.info("记忆已编码入缓冲区 (#%d) — %s",
                     self._learn_count, datetime.now().strftime("%H:%M:%S"))
        return encoded

    def consolidate(self, force: bool = False) -> dict:
        return consolidator.consolidate(force=force)

    def export_training_data(self) -> dict:
        """Export merged training dataset from digests + buffer."""
        from .distiller import distiller
        merged = distiller.prepare_merged()
        return {"dataset_path": str(merged), "status": "ready"}

    def report(self) -> dict:
        return consolidator.report()

    def status(self) -> dict:
        from .nebius_client import nebius_client
        return {
            "initialized": self._initialized,
            "learn_count": self._learn_count,
            "buffer_size": replay_buffer.size(),
            "nebius": nebius_client.status(),
        }


chronos = ChronosSystem()
