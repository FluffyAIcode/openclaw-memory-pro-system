import logging

from .collector import collector
from .vectorstore import vector_store
from .digest import digest_memories
from .config import load_config

__version__ = "1.0.0"
__all__ = ["collector", "vector_store", "digest_memories", "load_config"]

logger = logging.getLogger(__name__)
logger.debug("Memora 记忆系统已加载 v%s", __version__)
