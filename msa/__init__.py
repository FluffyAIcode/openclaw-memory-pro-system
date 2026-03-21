"""
MSA (Memory Sparse Attention) — OpenClaw's document-level memory system.

Implements MSA's three-stage inference pipeline:
  1. Offline Encoding: chunk-mean pooling for compact routing keys
  2. Online Routing: sparse Top-k document selection via cosine similarity
  3. Sparse Generation: assemble full document context for LLM

Plus Memory Interleave for multi-hop reasoning across documents.
"""

from .system import MSASystem, msa_system
from .encoder import ChunkEncoder, EncodedDocument
from .memory_bank import MemoryBank
from .router import SparseRouter, ScoredDocument
from .interleave import MemoryInterleave, InterleaveResult
from .config import MSAConfig, load_config

__version__ = "1.0.0"

__all__ = [
    "MSASystem",
    "msa_system",
    "ChunkEncoder",
    "EncodedDocument",
    "MemoryBank",
    "SparseRouter",
    "ScoredDocument",
    "MemoryInterleave",
    "InterleaveResult",
    "MSAConfig",
    "load_config",
]
