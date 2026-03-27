"""Tests for memora/* — config, embedder, vectorstore, digest, zfs_integration."""

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ═══════════════════════════════════════════════════════════
# memora/config.py
# ═══════════════════════════════════════════════════════════

class TestMemoraConfig:

    def test_load_from_yaml(self):
        import memora.config as mc
        mc._config_cache = None
        cfg = mc.load_config()
        assert cfg.embedding_model == "nomic-ai/nomic-embed-text-v1.5"
        assert cfg.embedding_dimension == 768
        assert cfg.daily_dir is not None
        assert cfg.raw_dir is not None
        assert cfg.long_term_dir is not None
        assert cfg.vector_db_path is not None
        mc._config_cache = None

    def test_cache_returns_same_instance(self):
        import memora.config as mc
        mc._config_cache = None
        c1 = mc.load_config()
        c2 = mc.load_config()
        assert c1 is c2
        mc._config_cache = None

    def test_importance_validator(self):
        from memora.config import MemoraConfig
        with pytest.raises(Exception):
            MemoraConfig(importance_threshold=1.5)

    def test_digest_days_validator(self):
        from memora.config import MemoraConfig
        with pytest.raises(Exception):
            MemoraConfig(digest_interval_days=0)

    def test_ensure_dirs(self, tmp_path):
        from memora.config import MemoraConfig
        cfg = MemoraConfig(base_dir=tmp_path / "ensure")
        cfg.ensure_dirs()
        assert (tmp_path / "ensure").exists()
        assert cfg.daily_dir.exists()

    def test_resolve_base_with_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEMORA_BASE_DIR", str(tmp_path / "envbase"))
        from memora.config import _resolve_base
        base = _resolve_base()
        assert base == (tmp_path / "envbase").resolve()

    def test_resolve_base_no_env(self, monkeypatch):
        monkeypatch.delenv("MEMORA_BASE_DIR", raising=False)
        from memora.config import _resolve_base, _PACKAGE_DIR
        base = _resolve_base()
        expected = (_PACKAGE_DIR.parent / "memory").resolve()
        assert base == expected

    def test_model_post_init_defaults(self, tmp_path):
        from memora.config import MemoraConfig
        cfg = MemoraConfig(base_dir=tmp_path)
        assert cfg.raw_dir == tmp_path / "raw"
        assert cfg.daily_dir == tmp_path / "daily"
        assert cfg.long_term_dir == tmp_path / "long_term"
        assert cfg.vector_db_path == tmp_path / "vector_db"

    def test_load_without_yaml(self, tmp_path, monkeypatch):
        import memora.config as mc
        mc._config_cache = None
        orig = mc._PACKAGE_DIR
        mc._PACKAGE_DIR = tmp_path  # no config.yaml here
        monkeypatch.setenv("MEMORA_BASE_DIR", str(tmp_path / "noml"))
        try:
            cfg = mc.load_config()
            assert cfg is not None
        finally:
            mc._PACKAGE_DIR = orig
            mc._config_cache = None


# ═══════════════════════════════════════════════════════════
# memora/embedder.py
# ═══════════════════════════════════════════════════════════

class TestMockEmbedder:

    def test_produces_correct_dimension(self):
        from memora.embedder import MockEmbedder
        emb = MockEmbedder(dimension=64)
        vec = emb.embed("hello")
        assert len(vec) == 64

    def test_normalized(self):
        from memora.embedder import MockEmbedder
        emb = MockEmbedder(dimension=128)
        vec = emb.embed("test")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self):
        from memora.embedder import MockEmbedder
        emb = MockEmbedder(dimension=64)
        assert emb.embed("same") == emb.embed("same")

    def test_different_inputs_different_outputs(self):
        from memora.embedder import MockEmbedder
        emb = MockEmbedder(dimension=64)
        assert emb.embed("a") != emb.embed("b")


class TestSentenceTransformerEmbedder:

    def test_lazy_load(self):
        from memora.embedder import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder(model_name="fake", dimension=768)
        assert emb._model is None

    def test_embed_calls_model(self):
        from memora.embedder import SentenceTransformerEmbedder
        import numpy as np
        emb = SentenceTransformerEmbedder(model_name="fake", dimension=768)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(768).astype(np.float32)
        emb._model = mock_model
        result = emb.embed("test")
        mock_model.encode.assert_called_once()
        assert len(result) == 768


class TestEmbedderProxy:

    def test_proxy_delegates_embed(self, deterministic_embedder):
        from memora.embedder import _EmbedderProxy
        proxy = _EmbedderProxy()
        proxy._instance = deterministic_embedder
        vec = proxy.embed("hello")
        assert len(vec) == 768

    def test_proxy_dimension(self, deterministic_embedder):
        from memora.embedder import _EmbedderProxy
        proxy = _EmbedderProxy()
        proxy._instance = deterministic_embedder
        assert proxy.dimension == 768

    def test_proxy_lazy_creates_on_first_use(self):
        from memora.embedder import _EmbedderProxy, MockEmbedder
        proxy = _EmbedderProxy()
        proxy._instance = None
        with patch("memora.embedder._create_embedder") as mc:
            mc.return_value = MockEmbedder(dimension=32)
            result = proxy.embed("test")
            mc.assert_called_once()
            assert len(result) == 32
            proxy._instance = None

    def test_shared_embedder_check(self, deterministic_embedder):
        import shared_embedder
        shared_embedder.set(deterministic_embedder)
        from memora.embedder import _create_embedder
        emb = _create_embedder()
        assert emb is deterministic_embedder
        shared_embedder._instance = None



# ═══════════════════════════════════════════════════════════
# memora/vectorstore.py
# ═══════════════════════════════════════════════════════════

class TestVectorStore:

    def test_add_and_search(self, tmp_path, deterministic_embedder):
        from memora.vectorstore import VectorStore
        with patch("memora.vectorstore.embedder", deterministic_embedder):
            vs = VectorStore(db_path=tmp_path / "vs")
            vs.add("quantum physics is amazing", metadata={"source": "test"})
            vs.add("python programming language", metadata={"source": "test"})
            results = vs.search("quantum", limit=2)
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]

    def test_search_empty(self, tmp_path, deterministic_embedder):
        from memora.vectorstore import VectorStore
        with patch("memora.vectorstore.embedder", deterministic_embedder):
            vs = VectorStore(db_path=tmp_path / "empty_vs")
            assert vs.search("anything") == []

    def test_count(self, tmp_path, deterministic_embedder):
        from memora.vectorstore import VectorStore
        with patch("memora.vectorstore.embedder", deterministic_embedder):
            vs = VectorStore(db_path=tmp_path / "cnt")
            assert vs.count() == 0
            vs.add("entry one")
            assert vs.count() == 1

    def test_persistence(self, tmp_path, deterministic_embedder):
        from memora.vectorstore import VectorStore
        with patch("memora.vectorstore.embedder", deterministic_embedder):
            vs1 = VectorStore(db_path=tmp_path / "persist")
            vs1.add("persisted entry")
            vs2 = VectorStore(db_path=tmp_path / "persist")
            assert vs2.count() == 1

    def test_corrupted_entry_skipped(self, tmp_path, deterministic_embedder):
        db_path = tmp_path / "corrupt"
        db_path.mkdir()
        (db_path / "entries.jsonl").write_text('{"bad json\nnot json\n')
        from memora.vectorstore import VectorStore
        with patch("memora.vectorstore.embedder", deterministic_embedder):
            vs = VectorStore(db_path=db_path)
            assert vs.count() == 0

    def test_cosine_sim_identical(self):
        from memora.vectorstore import VectorStore
        sim = VectorStore._cosine_sim([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_sim_orthogonal(self):
        from memora.vectorstore import VectorStore
        sim = VectorStore._cosine_sim([1, 0], [0, 1])
        assert abs(sim) < 1e-6

    def test_cosine_sim_zero_vector(self):
        from memora.vectorstore import VectorStore
        sim = VectorStore._cosine_sim([0, 0], [1, 0])
        assert sim == 0.0

    def test_mismatched_vector_dimensions(self, tmp_path, deterministic_embedder):
        from memora.vectorstore import VectorStore
        db_path = tmp_path / "mismatch"
        db_path.mkdir()
        entry = {"content": "x", "vector": [0.1, 0.2], "timestamp": "t", "metadata": {}}
        (db_path / "entries.jsonl").write_text(json.dumps(entry) + "\n")
        with patch("memora.vectorstore.embedder", deterministic_embedder):
            vs = VectorStore(db_path=db_path)
            results = vs.search("query")
            assert results[0]["score"] == 0.0


# ═══════════════════════════════════════════════════════════
# second_brain/digest.py (moved from memora/)
# ═══════════════════════════════════════════════════════════

class TestDigest:

    def _make_cfg(self, tmp_path):
        from second_brain.config import SecondBrainConfig
        cfg = SecondBrainConfig(base_dir=tmp_path)
        cfg.ensure_dirs()
        cfg.daily_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    def test_digest_empty_daily(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        import second_brain.digest as dmod
        orig = dmod.config
        dmod.config = cfg
        try:
            result = dmod.digest_memories(days=7)
            assert result is True
        finally:
            dmod.config = orig

    def test_digest_with_files_and_llm(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        today = datetime.now().strftime("%Y-%m-%d")
        (cfg.daily_dir / f"{today}.md").write_text("### test entry\ncontent here\n")
        import second_brain.digest as dmod
        orig = dmod.config
        dmod.config = cfg
        try:
            with patch("llm_client.is_available", return_value=True), \
                 patch("llm_client.generate", return_value="AI summary here"):
                result = dmod.digest_memories(days=7)
        finally:
            dmod.config = orig
        assert result is True
        digests = list(cfg.long_term_dir.glob("digest_*.md"))
        assert len(digests) == 1
        content = digests[0].read_text()
        assert "AI summary here" in content
        assert "AI 摘要" in content

    def test_digest_llm_unavailable(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        today = datetime.now().strftime("%Y-%m-%d")
        (cfg.daily_dir / f"{today}.md").write_text("entry\n")
        import second_brain.digest as dmod
        orig = dmod.config
        dmod.config = cfg
        try:
            with patch("llm_client.is_available", return_value=False):
                result = dmod.digest_memories(days=7)
        finally:
            dmod.config = orig
        assert result is True
        digests = list(cfg.long_term_dir.glob("digest_*.md"))
        content = digests[0].read_text()
        assert "AI 摘要" not in content

    def test_digest_skips_bad_filenames(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        (cfg.daily_dir / "bad-name.md").write_text("stuff\n")
        import second_brain.digest as dmod
        orig = dmod.config
        dmod.config = cfg
        try:
            result = dmod.digest_memories(days=7)
        finally:
            dmod.config = orig
        assert result is True

    def test_digest_nonexistent_daily_dir(self, tmp_path):
        import second_brain.digest as dmod
        orig = dmod.config
        mock_cfg = MagicMock()
        mock_cfg.daily_dir = tmp_path / "nonexistent_daily"
        mock_cfg.digest_interval_days = 7
        dmod.config = mock_cfg
        try:
            result = dmod.digest_memories(days=7)
        finally:
            dmod.config = orig
        assert result is False

    def test_digest_with_msa_context(self, tmp_path):
        """Digest incorporates MSA cross-day context when available."""
        cfg = self._make_cfg(tmp_path)
        today = datetime.now().strftime("%Y-%m-%d")
        (cfg.daily_dir / f"{today}.md").write_text("### entry\n日志内容\n")

        import second_brain.digest as dmod
        orig = dmod.config
        dmod.config = cfg

        try:
            with patch("second_brain.digest._collect_msa_context",
                       return_value="跨天分析：记忆系统从v0.0.1到v0.0.4经历了四次迭代"), \
                 patch("llm_client.is_available", return_value=False):
                result = dmod.digest_memories(days=7)
        finally:
            dmod.config = orig

        assert result is True
        digests = list(cfg.long_term_dir.glob("digest_*.md"))
        assert len(digests) == 1
        content = digests[0].read_text()
        assert "MSA 跨天上下文" in content
        assert "跨天分析" in content

    def test_digest_msa_context_empty(self, tmp_path):
        """Digest works fine when MSA has no documents."""
        cfg = self._make_cfg(tmp_path)
        today = datetime.now().strftime("%Y-%m-%d")
        (cfg.daily_dir / f"{today}.md").write_text("### entry\ncontent\n")

        import second_brain.digest as dmod
        orig = dmod.config
        dmod.config = cfg
        try:
            with patch("second_brain.digest._collect_msa_context", return_value=""):
                with patch("llm_client.is_available", return_value=False):
                    result = dmod.digest_memories(days=7)
        finally:
            dmod.config = orig

        assert result is True
        digests = list(cfg.long_term_dir.glob("digest_*.md"))
        content = digests[0].read_text()
        assert "MSA 跨天上下文" not in content


# ═══════════════════════════════════════════════════════════
# memora/zfs_integration.py
# ═══════════════════════════════════════════════════════════

class TestZFSIntegration:

    def test_check_zfs_not_available(self):
        from memora.zfs_integration import ZFSIntegration
        z = ZFSIntegration()
        with patch("memora.zfs_integration.subprocess.run",
                   side_effect=FileNotFoundError):
            assert z.check_zfs_available() is False

    def test_check_zfs_timeout(self):
        import subprocess
        from memora.zfs_integration import ZFSIntegration
        z = ZFSIntegration()
        with patch("memora.zfs_integration.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("zfs", 5)):
            assert z.check_zfs_available() is False

    def test_check_zfs_available_true(self):
        from memora.zfs_integration import ZFSIntegration
        z = ZFSIntegration()
        mock_result = MagicMock(returncode=0)
        with patch("memora.zfs_integration.subprocess.run", return_value=mock_result):
            assert z.check_zfs_available() is True

    def test_snapshot_disabled(self):
        from memora.zfs_integration import ZFSIntegration
        z = ZFSIntegration()
        z.enabled = False
        assert z.create_snapshot() is False

    def test_snapshot_no_dataset(self):
        from memora.zfs_integration import ZFSIntegration
        z = ZFSIntegration()
        z.enabled = True
        z.dataset = None
        assert z.create_snapshot() is False

    def test_snapshot_success(self):
        from memora.zfs_integration import ZFSIntegration
        z = ZFSIntegration()
        z.enabled = True
        z.dataset = "pool/memora"
        mock_result = MagicMock(returncode=0)
        with patch("memora.zfs_integration.subprocess.run", return_value=mock_result):
            assert z.create_snapshot("test_snap") is True

    def test_snapshot_failure(self):
        from memora.zfs_integration import ZFSIntegration
        z = ZFSIntegration()
        z.enabled = True
        z.dataset = "pool/memora"
        mock_result = MagicMock(returncode=1, stderr="error msg")
        with patch("memora.zfs_integration.subprocess.run", return_value=mock_result):
            assert z.create_snapshot() is False

    def test_snapshot_exception(self):
        from memora.zfs_integration import ZFSIntegration
        z = ZFSIntegration()
        z.enabled = True
        z.dataset = "pool/memora"
        with patch("memora.zfs_integration.subprocess.run",
                   side_effect=OSError("oops")):
            assert z.create_snapshot() is False


class TestVectorStoreLock:
    """Verify VectorStore uses a threading lock for concurrent writes."""

    def test_has_lock(self, tmp_path):
        import threading
        from memora.vectorstore import VectorStore
        vs = VectorStore(db_path=tmp_path)
        assert type(vs._lock) is type(threading.Lock())

    def test_concurrent_add_no_corruption(self, tmp_path, deterministic_embedder):
        import threading
        from memora.vectorstore import VectorStore
        from memora.embedder import embedder as emb_proxy
        orig = emb_proxy._instance
        emb_proxy._instance = deterministic_embedder
        try:
            vs = VectorStore(db_path=tmp_path)
            errors = []

            def add_entry(i):
                try:
                    vs.add(f"concurrent entry {i}", dedup=False)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=add_entry, args=(i,))
                       for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors
            assert vs.count() == 10
        finally:
            emb_proxy._instance = orig
