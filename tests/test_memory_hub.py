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

    def test_remember_high_importance_auto_routes_chronos_and_msa(self, tmp_path):
        """High importance (>=0.85) auto-routes to Chronos AND MSA."""
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            hub._msa_bridge.ingest_and_save.return_value = {"doc_id": "d1", "chunks": 1}
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
        assert "msa" in result["systems_used"]
        assert "memora" in result["systems_used"]

    def test_remember_force_chronos(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            hub = self._make_hub(tmp_path)
            hub._chronos_bridge.learn_and_save.return_value = MagicMock(importance=0.95)
            result = hub.remember("text", force_systems=["chronos"])
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
            result = hub.remember("text", force_systems=["chronos"])
        finally:
            hmod._WORKSPACE = orig_ws
        assert "chronos" not in result["systems_used"]

    def test_recall(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = [
            {"content": "memory recall query result alpha", "score": 0.8}
        ]
        hub._msa_bridge.query_memory.return_value = {
            "results": [{"chunks": ["memory recall query chunk"],
                         "score": 0.7, "doc_id": "d1", "title": "T"}]
        }
        with patch("memory_hub.MemoryHub._recall_skills", return_value=[]), \
             patch("memory_hub.MemoryHub._recall_kg", return_value=[]):
            result = hub.recall("memory recall query")
        assert len(result["merged"]) == 2
        assert result["merged"][0]["score"] >= result["merged"][1]["score"]
        assert "skills" in result
        assert "kg_relations" in result
        assert "evidence" in result

    def test_recall_memora_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.side_effect = Exception("fail")
        hub._msa_bridge.query_memory.return_value = {"results": []}
        with patch("memory_hub.MemoryHub._recall_skills", return_value=[]), \
             patch("memory_hub.MemoryHub._recall_kg", return_value=[]):
            result = hub.recall("query")
        assert result["memora"] == []

    def test_recall_msa_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = []
        hub._msa_bridge.query_memory.side_effect = Exception("fail")
        with patch("memory_hub.MemoryHub._recall_skills", return_value=[]), \
             patch("memory_hub.MemoryHub._recall_kg", return_value=[]):
            result = hub.recall("query")
        assert result["msa"] == []

    def test_recall_with_skills(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = []
        hub._msa_bridge.query_memory.return_value = {"results": []}
        mock_skill = MagicMock()
        mock_skill.name = "AI记忆系统设计"
        mock_skill.content = "关于向量存储和嵌入模型的技能"
        mock_skill.tags = ["memory", "ai"]
        mock_skill.to_dict.return_value = {
            "id": "s1", "name": "AI记忆系统设计",
            "content": "关于向量存储和嵌入模型的技能",
            "status": "active", "tags": ["memory", "ai"],
        }
        mock_registry = MagicMock()
        mock_registry.list_active.return_value = [mock_skill]
        with patch.dict(sys.modules, {"skill_registry": MagicMock(registry=mock_registry)}), \
             patch("memory_hub.MemoryHub._recall_kg", return_value=[]):
            result = hub.recall("记忆系统")
        assert len(result["skills"]) == 1
        assert result["skills"][0]["name"] == "AI记忆系统设计"
        assert result["merged"][0]["system"] == "skill"

    def test_recall_with_kg_relations(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = []
        hub._msa_bridge.query_memory.return_value = {"results": []}
        mock_relations = [{
            "description": "[concept] RAG -[supports]→ [concept] Memory System",
            "edge_type": "supports",
            "source_content": "RAG",
            "target_content": "Memory System",
            "weight": 0.8,
            "is_critical": False,
            "metadata": {"source_id": "n1", "target_id": "n2"},
        }]
        with patch("memory_hub.MemoryHub._recall_skills", return_value=[]), \
             patch("memory_hub.MemoryHub._recall_kg", return_value=mock_relations):
            result = hub.recall("RAG系统")
        assert len(result["kg_relations"]) == 1
        assert result["merged"][0]["system"] == "kg"

    def test_recall_assembled_ordering(self, tmp_path):
        """Layered ordering: L1 Core (skills + evidence) → L2 Concept (KG) → L3 Background."""
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = [
            {"content": "memory system architecture design", "score": 0.6}
        ]
        hub._msa_bridge.query_memory.return_value = {"results": []}
        mock_skill = MagicMock()
        mock_skill.name = "Memory Architecture"
        mock_skill.content = "memory system architecture design skill"
        mock_skill.tags = ["memory"]
        mock_skill.to_dict.return_value = {
            "id": "s1", "name": "Memory Architecture",
            "content": "memory system architecture design skill",
            "status": "active", "tags": ["memory"],
        }
        mock_registry = MagicMock()
        mock_registry.list_active.return_value = [mock_skill]
        mock_kg = [{
            "description": "memory system architecture supports vector storage",
            "edge_type": "supports",
            "source_content": "memory system", "target_content": "architecture",
            "weight": 0.7, "relevance": 0.7, "is_critical": False, "metadata": {},
        }]
        with patch.dict(sys.modules, {"skill_registry": MagicMock(registry=mock_registry)}), \
             patch("memory_hub.MemoryHub._recall_kg", return_value=mock_kg):
            result = hub.recall("memory system architecture")
        merged = result["merged"]
        assert merged[0]["system"] == "skill"
        systems = [m["system"] for m in merged]
        assert "memora" in systems, "evidence should appear in merged output"
        layers = [m.get("layer") for m in merged]
        core_end = max(i for i, l in enumerate(layers) if l == "core") if "core" in layers else -1
        concept_start = min((i for i, l in enumerate(layers) if l == "concept"), default=len(merged))
        assert core_end < concept_start or concept_start == len(merged), \
            "L1 Core items should precede L2 Concept items"

    def test_recall_skills_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = []
        hub._msa_bridge.query_memory.return_value = {"results": []}
        with patch.dict(sys.modules, {"skill_registry": MagicMock(
                registry=MagicMock(list_active=MagicMock(side_effect=Exception("fail"))))}), \
             patch("memory_hub.MemoryHub._recall_kg", return_value=[]):
            result = hub.recall("query")
        assert result["skills"] == []

    def test_recall_kg_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._memora_bridge.search_across.return_value = []
        hub._msa_bridge.query_memory.return_value = {"results": []}
        with patch("memory_hub.MemoryHub._recall_skills", return_value=[]), \
             patch.dict(sys.modules, {"second_brain.knowledge_graph": MagicMock(
                 kg=MagicMock(find_node_by_content=MagicMock(side_effect=Exception("fail"))))}):
            result = hub.recall("query")
        assert result["kg_relations"] == []

    def test_recall_skills_tag_match(self, tmp_path):
        hub = self._make_hub(tmp_path)
        mock_skill = MagicMock()
        mock_skill.name = "Unrelated Skill"
        mock_skill.content = "nothing relevant here"
        mock_skill.tags = ["memory"]
        mock_skill.to_dict.return_value = {
            "id": "s1", "name": "Unrelated Skill",
            "content": "nothing relevant here",
            "status": "active", "tags": ["memory"],
        }
        mock_registry = MagicMock()
        mock_registry.list_active.return_value = [mock_skill]
        with patch.dict(sys.modules, {"skill_registry": MagicMock(registry=mock_registry)}):
            result = hub._recall_skills("memory")
        assert len(result) == 1

    def test_recall_skills_no_match(self, tmp_path):
        hub = self._make_hub(tmp_path)
        mock_skill = MagicMock()
        mock_skill.name = "Cooking Guide"
        mock_skill.content = "how to make pasta"
        mock_skill.tags = ["cooking"]
        mock_skill.to_dict.return_value = {
            "id": "s1", "name": "Cooking Guide",
            "content": "how to make pasta",
            "status": "active", "tags": ["cooking"],
        }
        mock_registry = MagicMock()
        mock_registry.list_active.return_value = [mock_skill]
        with patch.dict(sys.modules, {"skill_registry": MagicMock(registry=mock_registry)}):
            result = hub._recall_skills("量子计算")
        assert len(result) == 0

    def test_recall_skills_empty_registry(self, tmp_path):
        hub = self._make_hub(tmp_path)
        mock_registry = MagicMock()
        mock_registry.list_active.return_value = []
        with patch.dict(sys.modules, {"skill_registry": MagicMock(registry=mock_registry)}):
            result = hub._recall_skills("query")
        assert result == []

    def test_recall_kg_with_nodes(self, tmp_path):
        hub = self._make_hub(tmp_path)
        mock_node = MagicMock()
        mock_node.id = "n1"
        mock_node.content = "RAG vector retrieval"
        mock_node.node_type = MagicMock(value="concept")
        mock_tgt_node = MagicMock()
        mock_tgt_node.id = "n2"
        mock_tgt_node.content = "Embedding model"
        mock_tgt_node.node_type = MagicMock(value="concept")
        mock_kg = MagicMock()
        mock_kg.find_node_by_content.return_value = [mock_node]
        mock_kg.get_edges.return_value = [("n1", "n2", {"edge_type": "supports", "weight": 0.8})]
        mock_kg.get_node.side_effect = lambda nid: {"n1": mock_node, "n2": mock_tgt_node}.get(nid)
        mock_edge_type = MagicMock()
        mock_edge_type.CONTRADICTS = MagicMock(value="contradicts")
        mock_edge_type.ADDRESSES = MagicMock(value="addresses")
        with patch.dict(sys.modules, {
            "second_brain.knowledge_graph": MagicMock(kg=mock_kg, KGEdgeType=mock_edge_type),
            "shared_embedder": MagicMock(get=MagicMock(return_value=None)),
        }):
            result = hub._recall_kg("RAG")
        assert len(result) == 1
        assert "supports" in result[0]["edge_type"]

    def test_recall_kg_embedding_path(self, tmp_path):
        """When shared embedder is available, use vector similarity."""
        hub = self._make_hub(tmp_path)
        mock_node = MagicMock()
        mock_node.id = "n1"
        mock_node.content = "AI memory system"
        mock_node.node_type = MagicMock(value="concept")
        mock_kg = MagicMock()
        mock_kg.get_all_nodes.return_value = [mock_node]
        mock_kg.get_edges.return_value = []
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.5, 0.5]
        mock_emb.embed_document.return_value = [0.5, 0.5]
        mock_edge_type = MagicMock()
        mock_edge_type.CONTRADICTS = MagicMock(value="contradicts")
        mock_edge_type.ADDRESSES = MagicMock(value="addresses")
        with patch.dict(sys.modules, {
            "second_brain.knowledge_graph": MagicMock(kg=mock_kg, KGEdgeType=mock_edge_type),
            "shared_embedder": MagicMock(get=MagicMock(return_value=mock_emb)),
        }):
            result = hub._recall_kg("AI memory")
        assert isinstance(result, list)

    def test_recall_kg_no_nodes(self, tmp_path):
        hub = self._make_hub(tmp_path)
        mock_kg = MagicMock()
        mock_kg.find_node_by_content.return_value = []
        mock_edge_type = MagicMock()
        with patch.dict(sys.modules, {
            "second_brain.knowledge_graph": MagicMock(kg=mock_kg, KGEdgeType=mock_edge_type),
            "shared_embedder": MagicMock(get=MagicMock(return_value=None)),
        }):
            result = hub._recall_kg("nothing")
        assert result == []

    def test_deep_recall(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._msa_bridge.interleave_query.return_value = {"answer": "yes"}
        hub._memora_bridge.search_across.return_value = [{"content": "x", "score": 0.5}]
        with patch("memory_hub.MemoryHub._recall_skills", return_value=[]):
            result = hub.deep_recall("complex question")
        assert result["interleave"] == {"answer": "yes"}
        assert len(result["memora_context"]) == 1
        assert "skills" in result

    def test_deep_recall_msa_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._msa_bridge.interleave_query.side_effect = Exception("fail")
        hub._memora_bridge.search_across.return_value = []
        with patch("memory_hub.MemoryHub._recall_skills", return_value=[]):
            result = hub.deep_recall("q")
        assert result["interleave"] is None

    def test_deep_recall_memora_exception(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._msa_bridge.interleave_query.return_value = {}
        hub._memora_bridge.search_across.side_effect = Exception("fail")
        with patch("memory_hub.MemoryHub._recall_skills", return_value=[]):
            result = hub.deep_recall("q")
        assert result["memora_context"] == []

    def test_deep_recall_with_skills(self, tmp_path):
        hub = self._make_hub(tmp_path)
        hub._msa_bridge.interleave_query.return_value = {}
        hub._memora_bridge.search_across.return_value = []
        mock_skills = [{"name": "Test Skill", "id": "s1", "content": "c"}]
        with patch("memory_hub.MemoryHub._recall_skills", return_value=mock_skills):
            result = hub.deep_recall("q")
        assert len(result["skills"]) == 1

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
        r_low = hub._route_ingestion(50, 0.5)
        assert "memora" in r_low
        assert "msa" not in r_low
        assert "chronos" not in r_low

        r_long = hub._route_ingestion(200, 0.5)
        assert "msa" in r_long
        assert "chronos" not in r_long

        r_important = hub._route_ingestion(50, 0.9)
        assert "msa" in r_important
        assert "chronos" in r_important
        assert "memora" in r_important

        r_both = hub._route_ingestion(200, 0.95)
        assert r_both == {"memora", "msa", "chronos"}

    def test_write_daily(self, tmp_path):
        hmod = _get_hub_mod()
        orig_ws = hmod._WORKSPACE
        hmod._WORKSPACE = tmp_path
        try:
            from memory_hub import MemoryHub
            hub = MemoryHub()
            hub._write_daily("content", "test", ["memora"], tag="thought")
            daily = list((tmp_path / "memory").glob("*.md"))
            assert len(daily) == 1
            text = daily[0].read_text()
            assert "#thought" in text
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

    def test_remember_with_tag(self, tmp_path):
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
                result = hub.remember("my thought", tag="thought")
                assert result["tag"] == "thought"
                result2 = hub.remember("bad tag", tag="invalid_tag")
                assert result2["tag"] is None
        finally:
            hmod._WORKSPACE = orig_ws

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
