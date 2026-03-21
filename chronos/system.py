"""
ChronosSystem — top-level orchestrator.

Pipeline:  raw text → Encode → Buffer → EWC Learn → LoRA → Consolidate
"""

import logging
from datetime import datetime

from .encoder import encoder, EncodedMemory
from .replay_buffer import replay_buffer
from .ewc import ewc_engine
from .dynamic_lora import dynamic_lora
from .consolidator import consolidator

logger = logging.getLogger(__name__)


class ChronosSystem:
    """
    Chronos Continual-Learning Memory System.

    Core philosophy: 以算代存 — internalise memories into model parameters
    rather than relying solely on external storage / RAG retrieval.
    """

    def __init__(self):
        self._initialized = False
        self._learn_count = 0

    def initialize(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("Chronos 持续学习记忆系统 v1.0 已初始化")

    def learn(self, raw_text: str, importance: float = None) -> EncodedMemory:
        """Full learning pipeline: encode → buffer → EWC → LoRA → auto-consolidate."""
        if not self._initialized:
            self.initialize()

        encoded = encoder.encode(raw_text, importance=importance)
        replay_buffer.add(encoded)

        replay = replay_buffer.sample(batch_size=16)
        ewc_engine.learn([encoded], replay_memories=replay)
        dynamic_lora.allocate(encoded)

        if consolidator.should_consolidate():
            consolidator.consolidate()

        self._learn_count += 1
        logger.info("记忆已学习并内化 (#%d) — %s",
                     self._learn_count, datetime.now().strftime("%H:%M:%S"))
        return encoded

    def consolidate(self, force: bool = False) -> dict:
        return consolidator.consolidate(force=force)

    def report(self) -> dict:
        return consolidator.report()

    def status(self) -> dict:
        return {
            "initialized": self._initialized,
            "learn_count": self._learn_count,
            "buffer_size": replay_buffer.size(),
            "ewc_mode": ewc_engine.stats["mode"],
            "lora_adapters": dynamic_lora.stats["active_adapters"],
        }


chronos = ChronosSystem()
