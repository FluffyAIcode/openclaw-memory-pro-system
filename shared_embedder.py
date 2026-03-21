"""
Shared embedder singleton — allows the Memory Server to load the
SentenceTransformer model once and inject it into all subsystems.

In CLI mode, get() returns None and each subsystem creates its own.
In server mode, the server calls set() before any subsystem is imported.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_instance: Optional[Any] = None


def set(embedder) -> None:
    global _instance
    _instance = embedder
    logger.info("Shared embedder set: %s", type(embedder).__name__)


def get() -> Optional[Any]:
    return _instance
