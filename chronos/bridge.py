"""
Chronos <-> OpenClaw bridge (refactored).

Provides memory ingestion into the training pipeline buffer,
consolidation (personality profile), and training data export.
"""

import logging
from datetime import datetime
from pathlib import Path

from .system import chronos
from .consolidator import consolidator
from .config import load_config

logger = logging.getLogger(__name__)


class ChronosBridge:

    def __init__(self, memory_dir: Path = None):
        workspace = Path(__file__).parent.parent
        self.memory_dir = memory_dir or (workspace / "memory")

    def learn_and_save(self, content: str, source: str = "openclaw",
                       importance: float = 0.75, write_daily: bool = True):
        """Encode content into buffer for future training."""
        encoded = chronos.learn(content, importance=importance)

        if write_daily:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            today = self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
            with open(today, "a", encoding="utf-8") as f:
                f.write(f"\n### {datetime.now().strftime('%H:%M:%S')} "
                        f"[Chronos/{source}]\n{content}\n")

        logger.info("Chronos learn complete (write_daily=%s)", write_daily)
        return encoded

    def consolidate(self):
        return chronos.consolidate(force=True)

    def export_training_data(self):
        return chronos.export_training_data()

    def status(self) -> dict:
        return chronos.status()

    def report(self) -> dict:
        return chronos.report()


bridge = ChronosBridge()
