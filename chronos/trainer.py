"""
Batch training loop for Chronos continual learning.
"""

import logging
from typing import List

from .system import chronos
from .encoder import EncodedMemory
from .consolidator import consolidator

logger = logging.getLogger(__name__)


class ChronosTrainer:

    def __init__(self):
        self._epoch = 0

    def train_step(self, raw_text: str) -> EncodedMemory:
        self._epoch += 1
        logger.info("训练步 %d", self._epoch)

        encoded = chronos.learn(raw_text)

        if self._epoch % 5 == 0:
            consolidator.consolidate(force=True)

        return encoded

    def train(self, texts: List[str], epochs: int = 1):
        total = len(texts) * epochs
        logger.info("开始训练 — %d 文本 × %d 周期 = %d 步", len(texts), epochs, total)

        for ep in range(epochs):
            logger.info("=== 周期 %d/%d ===", ep + 1, epochs)
            for text in texts:
                self.train_step(text)

        consolidator.consolidate(force=True)
        logger.info("训练完成 — 共 %d 步", total)


trainer = ChronosTrainer()
