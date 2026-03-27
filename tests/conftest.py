"""Shared fixtures for all memory system tests."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

WORKSPACE = Path(__file__).parent.parent

# ── Ensure workspace is importable ──────────────────────────
sys.path.insert(0, str(WORKSPACE))


# ── Prevent real SentenceTransformer/CrossEncoder from loading ──
# This avoids HuggingFace network calls that hang in CI or sandboxed envs.
_fake_st_module = MagicMock()
_fake_model = MagicMock()
_fake_model.encode.side_effect = lambda text, **kw: (
    np.random.RandomState(abs(hash(str(text))) % 2**31).randn(768).astype(np.float32)
    if isinstance(text, str) else
    np.random.RandomState(42).randn(len(text) if hasattr(text, '__len__') else 1, 768).astype(np.float32)
)
_fake_st_module.SentenceTransformer.return_value = _fake_model
_fake_ce = MagicMock()
_fake_ce.predict.side_effect = lambda pairs: [0.5] * len(pairs)
_fake_st_module.CrossEncoder.return_value = _fake_ce

sys.modules["sentence_transformers"] = _fake_st_module


@pytest.fixture(autouse=True, scope="session")
def _reset_embedder_proxy():
    """Ensure the memora embedder proxy uses MockEmbedder instead of real ST."""
    try:
        from memora.embedder import embedder as emb_proxy
        if emb_proxy._instance is None:
            from memora.embedder import MockEmbedder
            emb_proxy._instance = MockEmbedder()
    except ImportError:
        pass
    yield


# ── Deterministic mock embedder used across all tests ───────
class DeterministicEmbedder:
    """Test embedder that produces stable, distinct, normalized vectors."""
    dimension = 768

    def embed(self, text: str, prefix: str = "search_document"):
        np.random.seed(hash(text) % 2**31)
        vec = np.random.randn(self.dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    def embed_document(self, text: str):
        return self.embed(text, prefix="search_document")

    def embed_query(self, text: str):
        return self.embed(text, prefix="search_query")

    def embed_batch(self, texts):
        return np.array([np.array(self.embed(t), dtype=np.float32) for t in texts])

    def embed_np(self, text: str):
        return np.array(self.embed(text), dtype=np.float32)


@pytest.fixture
def deterministic_embedder():
    return DeterministicEmbedder()


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Create a temp directory tree mimicking the memory layout."""
    dirs = ["daily", "raw", "long_term", "vector_db", "chronos/state",
            "msa/content"]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _block_real_llm_calls(request):
    """Prevent real LLM API calls in all tests by default.

    Tests that test llm_client itself can opt out with @pytest.mark.real_llm.
    Tests that need a working mock_llm can use the mock_llm fixture which
    patches is_available=True.
    """
    if "real_llm" in request.keywords:
        yield
        return
    with patch("llm_client.is_available", return_value=False), \
         patch("llm_client.generate", return_value=None):
        yield


@pytest.fixture
def mock_llm():
    """Patch llm_client.generate to return canned responses."""
    with patch("llm_client.is_available", return_value=True), \
         patch("llm_client.generate") as m:
        m.return_value = "LLM generated response for testing."
        yield m


@pytest.fixture
def mock_llm_unavailable():
    """Patch llm_client.is_available to return False."""
    with patch("llm_client.is_available", return_value=False), \
         patch("llm_client.generate", return_value=None):
        yield
