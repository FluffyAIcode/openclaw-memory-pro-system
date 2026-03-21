"""Second Brain — long-term memory tracking and inspiration collision."""

__version__ = "1.0.0"

from .tracker import MemoryTracker
from .collision import CollisionEngine
from .bridge import SecondBrainBridge, bridge

__all__ = [
    "MemoryTracker",
    "CollisionEngine",
    "SecondBrainBridge",
    "bridge",
]
