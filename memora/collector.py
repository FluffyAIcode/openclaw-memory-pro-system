import logging
import re
from datetime import datetime
from pathlib import Path

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize(text: str) -> str:
    return _CONTROL_CHARS.sub("", text)


def _truncate(text: str, limit: int = 60) -> str:
    return (text[:limit] + "...") if len(text) > limit else text


class MemoraCollector:
    def collect(self, content: str, source: str = "openclaw", importance: float = 0.7):
        """收集记忆并保存到 daily markdown"""
        content = _sanitize(content)
        source = _sanitize(source)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        daily_file = config.daily_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        daily_file.parent.mkdir(parents=True, exist_ok=True)

        with open(daily_file, "a", encoding="utf-8") as f:
            f.write(f"\n### {timestamp} [{source}]\n{content}\n")

        logger.info("记忆已收集: %s", _truncate(content))
        return {
            "content": content,
            "source": source,
            "importance": importance,
            "timestamp": timestamp,
        }


collector = MemoraCollector()


def collect(content: str, **kwargs):
    return collector.collect(content, **kwargs)
