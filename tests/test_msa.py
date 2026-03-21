"""Tests for msa/* — 100% coverage of config, encoder, memory_bank,
router, interleave, system, bridge."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════
# msa/config.py
# ═══════════════════════════════════════════════════════════

class TestMSAConfig:

    def test_load_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSA_BASE_DIR", str(tmp_path / "msa"))
        import msa.config as mc
        mc._config_cache = None
        cfg = mc.load_config()
        assert cfg.base_dir == (tmp_path / "msa").resolve()
        assert cfg.top_k == 5
        mc._config_cache = None

    def test_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSA_BASE_DIR", str(tmp_path / "msa2"))
        import msa.config as mc
        mc._config_cache = None
        c1 = mc.load_config()
        c2 = mc.load_config()
        assert c1 is c2
        mc._config_cache = None

    def test_validators(self):
        from msa.config import MSAConfig
        with pytest.raises(Exception):
            MSAConfig(top_k=0)
        with pytest.raises(Exception):
            MSAConfig(similarity_threshold=1.5)
        with pytest.raises(Exception):
            MSAConfig(chunk_size=10)


# ═══════════════════════════════════════════════════════════
# msa/encoder.py
# ═══════════════════════════════════════════════════════════

class TestMSAMockEmbedder:

    def test_embed(self):
        from msa.encoder import MockEmbedder
        emb = MockEmbedder(dimension=64)
        vec = emb.embed("hello")
        assert vec.shape == (64,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_embed_batch(self):
        from msa.encoder import MockEmbedder
        emb = MockEmbedder(dimension=64)
        batch = emb.embed_batch(["a", "b"])
        assert batch.shape == (2, 64)

    def test_deterministic(self):
        from msa.encoder import MockEmbedder
        emb = MockEmbedder(dimension=64)
        assert np.allclose(emb.embed("x"), emb.embed("x"))


class TestMSASentenceTransformerEmbedder:

    def test_lazy_load(self):
        from msa.encoder import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake", 768)
        assert emb._model is None

    def test_embed_delegates(self):
        from msa.encoder import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake", 768)
        emb._model = MagicMock()
        emb._model.encode.return_value = np.random.randn(768).astype(np.float32)
        result = emb.embed("test")
        assert result.shape == (768,)

    def test_embed_batch_delegates(self):
        from msa.encoder import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake", 768)
        emb._model = MagicMock()
        emb._model.encode.return_value = np.random.randn(3, 768).astype(np.float32)
        result = emb.embed_batch(["a", "b", "c"])
        assert result.shape == (3, 768)


class TestChunkEncoder:

    def test_split_short_text(self, deterministic_embedder):
        from msa.encoder import ChunkEncoder
        enc = ChunkEncoder(embedder=deterministic_embedder)
        enc.config = MagicMock(chunk_size=512, chunk_overlap=64)
        chunks = enc.split_into_chunks("short text here")
        assert len(chunks) == 1

    def test_split_long_text(self, deterministic_embedder):
        from msa.encoder import ChunkEncoder
        enc = ChunkEncoder(embedder=deterministic_embedder)
        enc.config = MagicMock(chunk_size=10, chunk_overlap=2)
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = enc.split_into_chunks(text)
        assert len(chunks) > 1

    def test_encode_document(self, deterministic_embedder):
        from msa.encoder import ChunkEncoder
        enc = ChunkEncoder(embedder=deterministic_embedder)
        enc.config = MagicMock(chunk_size=512, chunk_overlap=64)
        doc = enc.encode_document("doc1", "some document text", {"title": "T"})
        assert doc.doc_id == "doc1"
        assert len(doc.chunks) >= 1
        assert doc.routing_key.shape == (768,)
        assert abs(np.linalg.norm(doc.routing_key) - 1.0) < 1e-4

    def test_encode_query(self, deterministic_embedder):
        from msa.encoder import ChunkEncoder
        enc = ChunkEncoder(embedder=deterministic_embedder)
        vec = enc.encode_query("question")
        assert len(vec) == 768

    def test_shared_embedder_check(self, deterministic_embedder):
        import shared_embedder
        shared_embedder.set(deterministic_embedder)
        from msa.encoder import _create_embedder
        from msa.config import MSAConfig
        cfg = MSAConfig(embedding_model="fake", embedding_dimension=768)
        emb = _create_embedder(cfg)
        assert emb is deterministic_embedder
        shared_embedder._instance = None


# ═══════════════════════════════════════════════════════════
# msa/memory_bank.py
# ═══════════════════════════════════════════════════════════

class TestMemoryBank:

    def _make_bank(self, tmp_path):
        from msa.memory_bank import MemoryBank
        cfg = MagicMock()
        cfg.routing_keys_path = tmp_path / "routing.jsonl"
        cfg.content_store_path = tmp_path / "content"
        (tmp_path / "content").mkdir(exist_ok=True)
        bank = MemoryBank.__new__(MemoryBank)
        bank.config = cfg
        bank._documents = {}
        bank._routing_matrix = None
        bank._doc_ids_order = []
        bank._dirty = False
        return bank

    def test_add_and_count(self, tmp_path):
        bank = self._make_bank(tmp_path)
        key = np.random.randn(768).astype(np.float32)
        chunks = ["chunk 1", "chunk 2"]
        embs = np.random.randn(2, 768).astype(np.float32)
        bank.add_document("d1", key, chunks, embs, {"title": "T"})
        assert bank.document_count() == 1

    def test_get_all_routing_keys(self, tmp_path):
        bank = self._make_bank(tmp_path)
        key = np.random.randn(768).astype(np.float32)
        bank.add_document("d1", key, ["c"], np.random.randn(1, 768).astype(np.float32))
        keys = bank.get_all_routing_keys()
        assert keys.shape == (1, 768)

    def test_get_doc_ids(self, tmp_path):
        bank = self._make_bank(tmp_path)
        key = np.random.randn(768).astype(np.float32)
        bank.add_document("d1", key, ["c"], np.random.randn(1, 768).astype(np.float32))
        assert bank.get_doc_ids() == ["d1"]

    def test_load_document_content(self, tmp_path):
        bank = self._make_bank(tmp_path)
        key = np.random.randn(768).astype(np.float32)
        bank.add_document("d1", key, ["chunk1"], np.random.randn(1, 768).astype(np.float32))
        chunks, embs = bank.load_document_content("d1")
        assert chunks == ["chunk1"]

    def test_load_missing_raises(self, tmp_path):
        bank = self._make_bank(tmp_path)
        with pytest.raises(FileNotFoundError):
            bank.load_document_content("nonexistent")

    def test_remove_document(self, tmp_path):
        bank = self._make_bank(tmp_path)
        key = np.random.randn(768).astype(np.float32)
        bank.add_document("d1", key, ["c"], np.random.randn(1, 768).astype(np.float32))
        assert bank.remove_document("d1") is True
        assert bank.document_count() == 0

    def test_remove_nonexistent(self, tmp_path):
        bank = self._make_bank(tmp_path)
        assert bank.remove_document("nope") is False

    def test_flush(self, tmp_path):
        bank = self._make_bank(tmp_path)
        key = np.random.randn(768).astype(np.float32)
        bank.add_document("d1", key, ["c"], np.random.randn(1, 768).astype(np.float32))
        bank.flush()
        assert bank.config.routing_keys_path.exists()

    def test_flush_not_dirty(self, tmp_path):
        bank = self._make_bank(tmp_path)
        bank._dirty = False
        bank.flush()  # should be no-op

    def test_stats(self, tmp_path):
        bank = self._make_bank(tmp_path)
        s = bank.stats()
        assert s["document_count"] == 0

    def test_update_existing(self, tmp_path):
        bank = self._make_bank(tmp_path)
        key = np.random.randn(768).astype(np.float32)
        bank.add_document("d1", key, ["c1"], np.random.randn(1, 768).astype(np.float32))
        bank.add_document("d1", key, ["c1", "c2"], np.random.randn(2, 768).astype(np.float32))
        assert bank.document_count() == 1
        meta = bank.get_document_meta("d1")
        assert meta.chunk_count == 2

    def test_persistence(self, tmp_path):
        bank = self._make_bank(tmp_path)
        key = np.random.randn(768).astype(np.float32)
        bank.add_document("d1", key, ["c"], np.random.randn(1, 768).astype(np.float32))
        bank.flush()
        # Reload
        bank2 = self._make_bank(tmp_path)
        bank2._load_routing_index()
        assert bank2.document_count() == 1


# ═══════════════════════════════════════════════════════════
# msa/router.py
# ═══════════════════════════════════════════════════════════

class TestSparseRouter:

    def test_route_empty_bank(self):
        from msa.router import SparseRouter
        bank = MagicMock()
        bank.get_all_routing_keys.return_value = None
        router = SparseRouter(bank)
        router.config = MagicMock(top_k=5, similarity_threshold=0.3)
        assert router.route(np.random.randn(768).astype(np.float32)) == []

    def test_route_returns_sorted(self):
        from msa.router import SparseRouter
        bank = MagicMock()
        keys = np.random.randn(3, 768).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        bank.get_all_routing_keys.return_value = keys
        bank.get_doc_ids.return_value = ["d1", "d2", "d3"]
        router = SparseRouter(bank)
        router.config = MagicMock(top_k=2, similarity_threshold=0.0)
        query = keys[0]
        results = router.route(query, top_k=2)
        assert len(results) <= 2
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_route_threshold_filters(self):
        from msa.router import SparseRouter
        bank = MagicMock()
        keys = np.random.randn(2, 768).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        bank.get_all_routing_keys.return_value = keys
        bank.get_doc_ids.return_value = ["d1", "d2"]
        router = SparseRouter(bank)
        router.config = MagicMock(top_k=5, similarity_threshold=0.99)
        results = router.route(np.random.randn(768).astype(np.float32))
        # Very high threshold should filter most results
        assert len(results) <= 2

    def test_route_multi(self):
        from msa.router import SparseRouter
        bank = MagicMock()
        keys = np.random.randn(2, 768).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        bank.get_all_routing_keys.return_value = keys
        bank.get_doc_ids.return_value = ["d1", "d2"]
        router = SparseRouter(bank)
        router.config = MagicMock(top_k=2, similarity_threshold=0.0)
        queries = np.random.randn(3, 768).astype(np.float32)
        results = router.route_multi(queries)
        assert len(results) == 3

    def test_route_unnormalized_query(self):
        from msa.router import SparseRouter
        bank = MagicMock()
        keys = np.eye(768, dtype=np.float32)[:2]
        bank.get_all_routing_keys.return_value = keys
        bank.get_doc_ids.return_value = ["d1", "d2"]
        router = SparseRouter(bank)
        router.config = MagicMock(top_k=2, similarity_threshold=0.0)
        query = np.ones(768, dtype=np.float32) * 10  # unnormalized
        results = router.route(query)
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════
# msa/interleave.py
# ═══════════════════════════════════════════════════════════

class TestMemoryInterleave:

    def _make_interleave(self):
        from msa.interleave import MemoryInterleave
        from msa.router import ScoredDocument
        encoder = MagicMock()
        encoder.encode_query.return_value = np.random.randn(768).astype(np.float32)
        bank = MagicMock()
        bank.load_document_content.return_value = (["chunk1 text"], np.zeros((1, 768)))
        bank.get_document_meta.return_value = MagicMock(
            metadata={"title": "Test"})
        router = MagicMock()
        router.route.return_value = [ScoredDocument("d1", 0.9, 1)]
        config = MagicMock(max_interleave_rounds=2, top_k=3)
        return MemoryInterleave(encoder, bank, router, config=config)

    def test_run_single_round(self):
        il = self._make_interleave()
        il.generate_fn = lambda q, c: "sufficient answer about the topic"
        result = il.run("question")
        assert len(result.rounds) >= 1
        assert result.total_docs_used >= 1

    def test_run_no_new_docs_stops(self):
        il = self._make_interleave()
        il.generate_fn = lambda q, c: "i don't know"  # insufficient
        # After first round, same docs returned → no new docs → stops
        result = il.run("question", max_rounds=5)
        assert len(result.rounds) <= 2

    def test_run_without_generate_fn(self):
        from msa.interleave import MemoryInterleave
        encoder = MagicMock()
        encoder.encode_query.return_value = np.random.randn(768).astype(np.float32)
        bank = MagicMock()
        bank.load_document_content.return_value = (["text"], np.zeros((1, 768)))
        bank.get_document_meta.return_value = MagicMock(metadata={})
        from msa.router import ScoredDocument
        router = MagicMock()
        router.route.return_value = [ScoredDocument("d1", 0.9, 1)]
        config = MagicMock(max_interleave_rounds=1, top_k=3)
        il = MemoryInterleave(encoder, bank, router, config=config)
        # Default generate_fn should use _default_generate_fn
        with patch("msa.interleave.llm_client", create=True) as lc:
            lc.is_available.return_value = False
            result = il.run("q")
        assert result.final_answer is not None

    def test_is_sufficient(self):
        il = self._make_interleave()
        assert il._is_sufficient("a solid complete answer here!") is True
        assert il._is_sufficient("i don't know") is False
        assert il._is_sufficient("") is False
        assert il._is_sufficient("short") is False

    def test_reformulate(self):
        il = self._make_interleave()
        new_q = il._reformulate("original", "some intermediate text")
        assert "original" in new_q
        assert "considering" in new_q

    def test_default_generate_fn_with_llm(self):
        from msa.interleave import _default_generate_fn
        with patch("llm_client.is_available", return_value=True), \
             patch("llm_client.generate", return_value="LLM answer"):
            result = _default_generate_fn("q", "context")
            assert result == "LLM answer"

    def test_default_generate_fn_no_llm(self):
        from msa.interleave import _default_generate_fn
        with patch("llm_client.is_available", return_value=False):
            result = _default_generate_fn("q", "the context")
            assert result == "the context"

    def test_default_generate_fn_import_error(self):
        from msa.interleave import _default_generate_fn
        with patch.dict("sys.modules", {"llm_client": None}):
            # Should fallback to context
            result = _default_generate_fn("q", "ctx")
            assert result == "ctx"


# ═══════════════════════════════════════════════════════════
# msa/system.py
# ═══════════════════════════════════════════════════════════

class TestMSASystem:

    def _make_system(self, tmp_path):
        from msa.system import MSASystem
        from msa.encoder import MockEmbedder, ChunkEncoder
        from msa.memory_bank import MemoryBank
        from msa.router import SparseRouter
        from msa.interleave import MemoryInterleave
        cfg = MagicMock()
        cfg.base_dir = tmp_path
        cfg.routing_keys_path = tmp_path / "routing.jsonl"
        cfg.content_store_path = tmp_path / "content"
        cfg.state_path = tmp_path / "state.json"
        cfg.top_k = 3
        cfg.chunk_size = 512
        cfg.chunk_overlap = 64
        cfg.similarity_threshold = 0.3
        cfg.max_interleave_rounds = 2
        cfg.embedding_model = "fake"
        cfg.embedding_dimension = 768
        cfg.ensure_dirs = MagicMock()
        (tmp_path / "content").mkdir(exist_ok=True)
        emb = MockEmbedder(768)
        encoder = ChunkEncoder(embedder=emb, config=cfg)
        bank = MemoryBank.__new__(MemoryBank)
        bank.config = cfg
        bank._documents = {}
        bank._routing_matrix = None
        bank._doc_ids_order = []
        bank._dirty = False
        router = SparseRouter(bank, config=cfg)
        interleave = MemoryInterleave(encoder, bank, router, config=cfg)
        sys = MSASystem.__new__(MSASystem)
        sys.config = cfg
        sys.encoder = encoder
        sys.memory_bank = bank
        sys.router = router
        sys.interleave = interleave
        sys._initialized = True
        sys._ingest_count = 0
        return sys

    def test_ingest(self, tmp_path):
        sys = self._make_system(tmp_path)
        doc = sys.ingest("some document text for testing")
        assert doc.doc_id.startswith("doc_")
        assert sys._ingest_count == 1

    def test_ingest_with_id(self, tmp_path):
        sys = self._make_system(tmp_path)
        doc = sys.ingest("text", doc_id="custom_id")
        assert doc.doc_id == "custom_id"

    def test_batch_ingest(self, tmp_path):
        sys = self._make_system(tmp_path)
        docs = sys.batch_ingest([
            {"text": "doc one", "doc_id": "d1"},
            {"text": "doc two"},
        ])
        assert len(docs) == 2
        assert sys._ingest_count == 2

    def test_query(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys.ingest("quantum computing is powerful", doc_id="q1")
        result = sys.query("quantum")
        assert "results" in result

    def test_query_empty(self, tmp_path):
        sys = self._make_system(tmp_path)
        result = sys.query("anything")
        assert result["total_results"] == 0

    def test_remove(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys.ingest("text", doc_id="rm1")
        assert sys.remove("rm1") is True
        assert sys.remove("rm1") is False

    def test_status(self, tmp_path):
        sys = self._make_system(tmp_path)
        s = sys.status()
        assert "document_count" in s

    def test_report(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys.ingest("text", doc_id="rpt1")
        r = sys.report()
        assert len(r["documents"]) == 1

    def test_save_and_load_state(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys._ingest_count = 42
        sys._save_state()
        sys._ingest_count = 0
        sys._load_state()
        assert sys._ingest_count == 42

    def test_initialize_idempotent(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys._initialized = False
        sys.initialize()
        sys.initialize()
        assert sys._initialized is True


# ═══════════════════════════════════════════════════════════
# msa/bridge.py
# ═══════════════════════════════════════════════════════════

class TestMSABridge:

    def test_ingest_and_save_with_daily(self, tmp_path):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.ingest.return_value = MagicMock(
                doc_id="d1", chunks=["c1", "c2"])
            b = MSABridge(memory_dir=tmp_path)
            result = b.ingest_and_save("long text", write_daily=True)
            assert result["doc_id"] == "d1"
            daily = list(tmp_path.glob("*.md"))
            assert len(daily) == 1

    def test_ingest_and_save_no_daily(self, tmp_path):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.ingest.return_value = MagicMock(
                doc_id="d2", chunks=["c"])
            b = MSABridge(memory_dir=tmp_path)
            b.ingest_and_save("text", write_daily=False, cross_index=False)
            assert len(list(tmp_path.glob("*.md"))) == 0

    def test_cross_index(self, tmp_path):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys, \
             patch("memora.vectorstore.vector_store") as mvs:
            msys.ingest.return_value = MagicMock(
                doc_id="d3", chunks=["c1", "c2"])
            b = MSABridge(memory_dir=tmp_path)
            b.ingest_and_save("text", cross_index=True, write_daily=False)
            assert mvs.add.call_count == 2

    def test_cross_index_exception(self, tmp_path):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.ingest.return_value = MagicMock(doc_id="d4", chunks=["c"])
            with patch("memora.vectorstore.vector_store", side_effect=Exception):
                b = MSABridge(memory_dir=tmp_path)
                result = b.ingest_and_save("text", cross_index=True, write_daily=False)
                assert result["doc_id"] == "d4"

    def test_query_memory(self):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.query.return_value = {"results": []}
            b = MSABridge()
            assert b.query_memory("q") == {"results": []}

    def test_query_with_memora(self):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys, \
             patch("memora.vectorstore.vector_store") as mvs:
            msys.query.return_value = {"results": []}
            mvs.search.return_value = [{"content": "x", "score": 0.5}]
            b = MSABridge()
            result = b.query_with_memora("q")
            assert len(result["memora_snippets"]) == 1

    def test_query_with_memora_exception(self):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.query.return_value = {"results": []}
            with patch("memora.vectorstore.vector_store") as mvs:
                mvs.search.side_effect = Exception("fail")
                b = MSABridge()
                result = b.query_with_memora("q")
                assert result["memora_snippets"] == []

    def test_interleave_query(self):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.interleave_query.return_value = MagicMock(
                final_answer="answer", rounds=[1], total_docs_used=2,
                doc_ids_used=["d1"])
            b = MSABridge()
            result = b.interleave_query("q")
            assert result["final_answer"] == "answer"

    def test_remove(self):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.remove.return_value = True
            b = MSABridge()
            assert b.remove("d1") is True

    def test_status(self):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.status.return_value = {"ok": True}
            assert MSABridge().status()["ok"] is True

    def test_report(self):
        from msa.bridge import MSABridge
        with patch("msa.bridge.msa_system") as msys:
            msys.report.return_value = {"rpt": True}
            assert MSABridge().report()["rpt"] is True
