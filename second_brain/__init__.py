"""Second Brain — intelligence layer: KG weaving, distillation, collision."""

__version__ = "2.0.0"

from .tracker import MemoryTracker
from .collision import CollisionEngine
from .bridge import SecondBrainBridge, bridge
from .digest import digest_memories

__all__ = [
    "MemoryTracker",
    "CollisionEngine",
    "SecondBrainBridge",
    "bridge",
    "digest_memories",
]
