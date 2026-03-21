import logging

from .system import chronos
from .encoder import encoder
from .replay_buffer import replay_buffer
from .ewc import ewc_engine
from .dynamic_lora import dynamic_lora
from .consolidator import consolidator

__version__ = "1.0.0"
__all__ = [
    "chronos", "encoder", "replay_buffer",
    "ewc_engine", "dynamic_lora", "consolidator",
]

logger = logging.getLogger(__name__)
logger.debug("Chronos 持续学习记忆系统 v%s 已加载", __version__)
