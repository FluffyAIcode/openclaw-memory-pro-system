"""Tests for memory_hub.py — 100% coverage."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


def _get_hub_mod():
    import memory_hub
    return sys.modules["memory_hub"]


class TestMemoryHub:

    def _make_hub(self, tmp_path):
        from memory_hub import MemoryHub
        hub = MemoryHub()
        hub._memora_bridge = MagicMock()
        hub._chronos_bridge = MagicMock()
        hub._msa_bridge = MagicMock()
        return hub

    def test_remember_short_text(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            mock_collector = MagicMock()
            mock_collector.collect.return_value = {"timestamp": "t"}
            mock_vs = MagicMock()

            with patch.dict(sys.modules, {
                "memora.collector": MagicMock(collector=mock_collector),
                "memora.vectorstore": MagicMock(vector_store=mock_vs),
            }):
                result = hub.remember("short text", importance=0.5)
        finally:
            hmod._WORKSPACE = orig_ws
        assert "memora" in result["systems_used"]
        assert "daily_file" in result["systems_used"]

    def test_remember_long_text_routes_msa(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            hub._msa_bridge.ingest_and_save.return_value = {"doc_id": "d1", "chunks": 3}
            mock_collector = MagicMock()
            mock_collector.collect.return_value = {"timestamp": "t"}
            mock_vs = MagicMock()
            with patch.dict(sys.modules, {
                "memora.collector": MagicMock(collector=mock_collector),
                "memora.vectorstore": MagicMock(vector_store=mock_vs),
            }):
                long_text = "word " * 200
                result = hub.remember(long_text, importance=0.5)
        finally:
            hmod._WORKSPACE = orig_ws
        assert "msa" in result["systems_used"]

    def test_remember_high_importance_routes_chronos(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            hub._chronos_bridge.learn_and_save.return_value = MagicMock(importance=0.95)
            mock_collector = MagicMock()
            mock_collector.collect.return_value = {"timestamp": "t"}
            mock_vs = MagicMock()
            with patch.dict(sys.modules, {
                "memora.collector": MagicMock(collector=mock_collector),
                "memora.vectorstore": MagicMock(vector_store=mock_vs),
            }):
                result = hub.remember("important", importance=0.95)
        finally:
            hmod._WORKSPACE = orig_ws
        assert "chronos" in result["systems_used"]

    def test_remember_force_systems(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            hub._msa_bridge.ingest_and_save.return_value = {"doc_id": "d1", "chunks": 1}
            result = hub.remember("text", force_systems=["msa"])
        finally:
            hmod._WORKSPACE = orig_ws
        assert "msa" in result["systems_used"]

    def test_remember_memora_exception(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            with patch.dict(sys.modules, {
                "memora.collector": MagicMock(
                    collector=MagicMock(collect=MagicMock(side_effect=Exception("fail")))),
                "memora.vectorstore": MagicMock(),
            }):
                result = hub.remember("text")
        finally:
            hmod._WORKSPACE = orig_ws
        assert "memora" not in result["systems_used"]

    def test_remember_msa_exception(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            hub._msa_bridge.ingest_and_save.side_effect = Exception("fail")
            mock_collector = MagicMock()
            mock_collector.collect.return_value = {"timestamp": "t"}
            with patch.dict(sys.modules, {
                "memora.collector": MagicMock(collector=mock_collector),
                "memora.vectorstore": MagicMock(),
            }):
                result = hub.remember("word " * 200, importance=0.5)
        finally:
            hmod._WORKSPACE = orig_ws
        assert "msa" not in result["systems_used"]

    def test_remember_chronos_exception(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            hub._chronos_bridge.learn_and_save.side_effect = Exception("fail")
            mock_collector = MagicMock()
            mock_collector.collect.return_value = {"timestamp": "t"}
            with patch.dict(sys.modules, {
                "memora.collector": MagicMock(collector=mock_collector),
                "memora.vectorstore": MagicMock(),
            }):
                result = hub.remember("text", importance=0.95)
        finally:
            hmod._WORKSPACE = orig_ws
        assert "chronos" not in result["systems_used"]

    def test_recall(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = [
            {"content": "x", "score": 0.8}
        ]
        hub._msa_bridge.query_memory.return_value = {
            "results": [{"chunks": ["c1"], "score": 0.7, "doc_id": "d1", "title": "T"}]
        }
        result = hub.recall("query")
        assert len(result["merged"]) == 2
        assert result["merged"][0]["score"] >= result["merged"][1]["score"]

    def test_recall_memora_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.side_effect = Exception("fail")
        hub._msa_bridge.query_memory.return_value = {"results": []}
        result = hub.recall("query")
        assert result["memora"] == []

    def test_recall_msa_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = []
        hub._msa_bridge.query_memory.side_effect = Exception("fail")
        result = hub.recall("query")
        assert result["msa"] == []

    def test_deep_recall(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._msa_bridge.interleave_query.return_value = {"answer": "yes"}
        hub._memora_bridge.search_across.return_value = [{"content": "x", "score": 0.5}]
        result = hub.deep_recall("complex question")
        assert result["interleave"] == {"answer": "yes"}
        assert len(result["memora_context"]) == 1

    def test_deep_recall_msa_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._msa_bridge.interleave_query.side_effect = Exception("fail")
        hub._memora_bridge.search_across.return_value = []
        result = hub.deep_recall("q")
        assert result["interleave"] is None

    def test_deep_recall_memora_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._msa_bridge.interleave_query.return_value = {}
        hub._memora_bridge.search_across.side_effect = Exception("fail")
        result = hub.deep_recall("q")
        assert result["memora_context"] == []

    def test_status(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._chronos_bridge.status.return_value = {"ok": True}
        hub._msa_bridge.status.return_value = {"docs": 3}
        mock_vs = MagicMock()
        mock_vs.count.return_value = 5
        with patch.dict(sys.modules, {
            "memora.vectorstore": MagicMock(vector_store=mock_vs),
        }):
            result = hub.status()
        assert result["systems"]["memora"]["entries"] == 5
        assert result["systems"]["chronos"]["ok"] is True

    def test_status_exceptions(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._chronos_bridge.status.side_effect = Exception("c fail")
        hub._msa_bridge.status.side_effect = Exception("m fail")
        with patch.dict(sys.modules, {
            "memora.vectorstore": MagicMock(
                vector_store=MagicMock(count=MagicMock(side_effect=Exception("v fail")))),
        }):
            result = hub.status()
        assert "error" in result["systems"]["memora"]
        assert "error" in result["systems"]["chronos"]
        assert "error" in result["systems"]["msa"]

    def test_route_ingestion(self, tmp_path):
        hub = self._make_hub(tmp_path)
        assert "memora" in hub._route_ingestion(50, 0.5)
        assert "msa" in hub._route_ingestion(200, 0.5)
        assert "chronos" in hub._route_ingestion(50, 0.9)

    def test_write_daily(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            from memory_hub import MemoryHub
            hub = MemoryHub()
            hub._write_daily("content", "test", ["memora"])
            daily = list((tmp_path / "memory").glob("*.md"))
            assert len(daily) == 1
        finally:
            hmod._WORKSPACE = orig_ws

    def test_lazy_properties(self):
        from memory_hub import MemoryHub
        hub = MemoryHub()
        # memora
        hub._memora_bridge = None
        with patch("memora.bridge.bridge", new="mock_memora"):
            assert hub.memora == "mock_memora"
        # chronos
        hub._chronos_bridge = None
        import chronos.bridge as cbmod
        orig = cbmod.bridge
        cbmod.bridge = "mock_chronos"
        try:
            from memory_hub import MemoryHub as MH
            h2 = MH()
            assert h2.chronos == "mock_chronos"
        finally:
            cbmod.bridge = orig
        # msa
        hub._msa_bridge = None
        import msa.bridge as mbmod
        orig_m = mbmod.bridge
        mbmod.bridge = "mock_msa"
        try:
            h3 = MemoryHub()
            assert h3.msa == "mock_msa"
        finally:
            mbmod.bridge = orig_m

    def test_chinese_word_count(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            mock_collector = MagicMock()
            mock_collector.collect.return_value = {"timestamp": "t"}
            with patch.dict(sys.modules, {
                "memora.collector": MagicMock(collector=mock_collector),
                "memora.vectorstore": MagicMock(),
            }):
                result = hub.remember("这是一段中文测试内容")
        finally:
            hmod._WORKSPACE = orig_ws
        assert result["word_count"] > 1
