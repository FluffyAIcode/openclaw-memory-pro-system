"""Tests for memora/* — 100% coverage of config, embedder, collector,
vectorstore, digest, distiller, bridge, zfs_integration."""

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
# memora/collector.py
# ═══════════════════════════════════════════════════════════

class TestCollector:

    def _get_collector_mod(self):
        return sys.modules["memora.collector"]

    def test_collect_writes_daily_file(self, tmp_path):
        from memora.config import MemoraConfig
        cfg = MemoraConfig(base_dir=tmp_path)
        cfg.ensure_dirs()

        from memora.collector import MemoraCollector
        cmod = self._get_collector_mod()
        orig_cfg = cmod.config
        cmod.config = cfg
        try:
            c = MemoraCollector()
            entry = c.collect("test content", source="unit_test", importance=0.8)
        finally:
            cmod.config = orig_cfg

        assert entry["content"] == "test content"
        assert entry["source"] == "unit_test"
        assert entry["importance"] == 0.8
        assert "timestamp" in entry
        daily_files = list(cfg.daily_dir.glob("*.md"))
        assert len(daily_files) >= 1

    def test_sanitize_control_chars(self):
        from memora.collector import _sanitize
        assert _sanitize("hello\x00world") == "helloworld"
        assert _sanitize("normal") == "normal"

    def test_truncate(self):
        from memora.collector import _truncate
        assert _truncate("short", 60) == "short"
        assert _truncate("a" * 100, 60).endswith("...")

    def test_module_level_collect(self, tmp_path):
        from memora.config import MemoraConfig
        cfg = MemoraConfig(base_dir=tmp_path)
        cfg.ensure_dirs()
        cmod = self._get_collector_mod()
        orig_cfg = cmod.config
        cmod.config = cfg
        try:
            from memora.collector import collect
            entry = collect("test", source="x")
            assert entry["source"] == "x"
        finally:
            cmod.config = orig_cfg


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
# memora/bridge.py
# ═══════════════════════════════════════════════════════════

class TestMemoraBridge:

    def test_save_to_both(self, tmp_path, monkeypatch, deterministic_embedder):
        monkeypatch.setenv("MEMORA_BASE_DIR", str(tmp_path))
        import memora.config as mc
        mc._config_cache = None
        from memora.bridge import MemoraBridge
        with patch("memora.vectorstore.embedder", deterministic_embedder), \
             patch("memora.bridge.vector_store") as mvs, \
             patch("memora.bridge.collector") as mcoll:
            mcoll.collect.return_value = {"timestamp": "t", "content": "c",
                                          "source": "s", "importance": 0.7}
            b = MemoraBridge(memory_dir=tmp_path)
            entry = b.save_to_both("test", source="ut", importance=0.5)
            mcoll.collect.assert_called_once()
            mvs.add.assert_called_once()
        mc._config_cache = None

    def test_save_to_both_chronos_forwarding(self, tmp_path, monkeypatch):
        import memora.config as mc
        mc._config_cache = None
        from memora.bridge import MemoraBridge
        with patch("memora.bridge.collector") as mcoll, \
             patch("memora.bridge.vector_store"):
            mcoll.collect.return_value = {"timestamp": "t", "content": "c",
                                          "source": "s", "importance": 0.9}
            with patch("chronos.bridge.bridge") as chrono_mock:
                b = MemoraBridge(memory_dir=tmp_path)
                b.save_to_both("important", importance=0.9)
                chrono_mock.learn_and_save.assert_called_once()
        mc._config_cache = None

    def test_save_to_both_msa_forwarding(self, tmp_path, monkeypatch):
        import memora.config as mc
        mc._config_cache = None
        from memora.bridge import MemoraBridge
        long_text = " ".join(["word"] * 150)
        with patch("memora.bridge.collector") as mcoll, \
             patch("memora.bridge.vector_store"):
            mcoll.collect.return_value = {"timestamp": "t", "content": long_text,
                                          "source": "s", "importance": 0.5}
            with patch("msa.bridge.bridge") as msa_mock:
                b = MemoraBridge(memory_dir=tmp_path)
                b.save_to_both(long_text, importance=0.5)
                msa_mock.ingest_and_save.assert_called_once()
        mc._config_cache = None

    def test_save_chronos_exception(self, tmp_path):
        from memora.bridge import MemoraBridge
        with patch("memora.bridge.collector") as mcoll, \
             patch("memora.bridge.vector_store"):
            mcoll.collect.return_value = {"timestamp": "t", "content": "c",
                                          "source": "s", "importance": 0.9}
            with patch("chronos.bridge.bridge") as cm:
                cm.learn_and_save.side_effect = Exception("fail")
                b = MemoraBridge(memory_dir=tmp_path)
                entry = b.save_to_both("test", importance=0.9)
                assert entry is not None

    def test_search_across(self, tmp_path, deterministic_embedder):
        from memora.bridge import MemoraBridge
        with patch("memora.bridge.vector_store") as mvs:
            mvs.search.return_value = [{"content": "x", "score": 0.9, "timestamp": "t"}]
            b = MemoraBridge(memory_dir=tmp_path)
            results = b.search_across("query", include_msa=False)
            assert len(results) == 1

    def test_search_across_with_msa(self, tmp_path):
        from memora.bridge import MemoraBridge
        with patch("memora.bridge.vector_store") as mvs, \
             patch("msa.bridge.bridge") as msa_mock:
            mvs.search.return_value = []
            msa_mock.query_memory.return_value = {
                "results": [{"chunks": ["c1", "c2"], "score": 0.8,
                             "doc_id": "d1", "title": "T"}]
            }
            b = MemoraBridge(memory_dir=tmp_path)
            results = b.search_across("query", include_msa=True)
            assert len(results) == 1
            assert results[0]["metadata"]["source"] == "msa"

    def test_search_msa_exception(self, tmp_path):
        from memora.bridge import MemoraBridge
        with patch("memora.bridge.vector_store") as mvs, \
             patch("msa.bridge.bridge") as msa_mock:
            mvs.search.return_value = [{"content": "x", "score": 0.5}]
            msa_mock.query_memory.side_effect = Exception("msa fail")
            b = MemoraBridge(memory_dir=tmp_path)
            results = b.search_across("query", include_msa=True)
            assert len(results) == 1

    def test_auto_digest(self, tmp_path):
        from memora.bridge import MemoraBridge
        with patch("memora.bridge.digest_memories") as md:
            b = MemoraBridge(memory_dir=tmp_path)
            b.auto_digest()
            md.assert_called_once()


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
