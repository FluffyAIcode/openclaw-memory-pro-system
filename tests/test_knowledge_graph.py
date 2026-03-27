"""Tests for second_brain.knowledge_graph module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from second_brain.knowledge_graph import (
    KGEdge, KGEdgeType, KGNode, KGNodeType, KnowledgeGraph,
)


@pytest.fixture
def tmp_kg(tmp_path):
    """Create a KnowledgeGraph backed by a temp directory."""
    kg = KnowledgeGraph(kg_path=tmp_path)
    return kg


class TestKGNode:
    def test_create_node(self):
        n = KGNode("test fact", KGNodeType.FACT, importance=0.8)
        assert n.content == "test fact"
        assert n.node_type == KGNodeType.FACT
        assert n.importance == 0.8
        assert len(n.id) == 12
        assert n.access_count == 0

    def test_to_dict_roundtrip(self):
        n = KGNode("test decision", KGNodeType.DECISION, confidence=0.9)
        d = n.to_dict()
        n2 = KGNode.from_dict(d)
        assert n2.content == n.content
        assert n2.node_type == n.node_type
        assert n2.id == n.id
        assert n2.confidence == n.confidence

    def test_sentiment_field(self):
        n = KGNode("user is excited", KGNodeType.PREFERENCE, sentiment="excited")
        assert n.sentiment == "excited"
        d = n.to_dict()
        assert d["sentiment"] == "excited"
        n2 = KGNode.from_dict(d)
        assert n2.sentiment == "excited"

    def test_sentiment_empty_not_in_dict(self):
        n = KGNode("neutral fact", KGNodeType.FACT)
        assert n.sentiment == ""
        d = n.to_dict()
        assert "sentiment" not in d

    def test_touch_increments_access(self):
        n = KGNode("fact", KGNodeType.FACT)
        assert n.access_count == 0
        n.touch()
        assert n.access_count == 1
        n.touch()
        assert n.access_count == 2

    def test_update_maturity(self):
        n = KGNode("fact", KGNodeType.FACT, confidence=0.9)
        n.access_count = 5
        n.update_maturity()
        assert n.maturity > 0


class TestKGEdge:
    def test_create_edge(self):
        e = KGEdge("a", "b", KGEdgeType.SUPPORTS, weight=0.8)
        assert e.source_id == "a"
        assert e.target_id == "b"
        assert e.edge_type == KGEdgeType.SUPPORTS
        assert e.weight == 0.8

    def test_to_dict_roundtrip(self):
        e = KGEdge("a", "b", KGEdgeType.CONTRADICTS, evidence="test reason")
        d = e.to_dict()
        e2 = KGEdge.from_dict(d)
        assert e2.source_id == e.source_id
        assert e2.edge_type == e.edge_type
        assert e2.evidence == "test reason"


class TestKnowledgeGraph:
    def test_add_and_get_node(self, tmp_kg):
        n = KGNode("ESP32 supports WiFi", KGNodeType.FACT)
        nid = tmp_kg.add_node(n)
        assert tmp_kg.get_node(nid) is n
        assert nid == n.id

    def test_add_edge(self, tmp_kg):
        n1 = KGNode("ESP32 supports WiFi", KGNodeType.FACT)
        n2 = KGNode("chose ESP32", KGNodeType.DECISION)
        tmp_kg.add_node(n1)
        tmp_kg.add_node(n2)
        e = KGEdge(n1.id, n2.id, KGEdgeType.SUPPORTS, weight=0.9)
        assert tmp_kg.add_edge(e) is True

    def test_add_edge_unknown_node(self, tmp_kg):
        e = KGEdge("nonexistent", "also_nonexistent", KGEdgeType.SUPPORTS)
        assert tmp_kg.add_edge(e) is False

    def test_get_edges(self, tmp_kg):
        n1 = KGNode("fact A", KGNodeType.FACT)
        n2 = KGNode("decision B", KGNodeType.DECISION)
        tmp_kg.add_node(n1)
        tmp_kg.add_node(n2)
        tmp_kg.add_edge(KGEdge(n1.id, n2.id, KGEdgeType.SUPPORTS))
        edges = tmp_kg.get_edges(n2.id, edge_type=KGEdgeType.SUPPORTS, direction="in")
        assert len(edges) == 1
        assert edges[0][0] == n1.id

    def test_get_neighbors(self, tmp_kg):
        n1 = KGNode("fact", KGNodeType.FACT)
        n2 = KGNode("decision", KGNodeType.DECISION)
        tmp_kg.add_node(n1)
        tmp_kg.add_node(n2)
        tmp_kg.add_edge(KGEdge(n1.id, n2.id, KGEdgeType.SUPPORTS))
        neighbors = tmp_kg.get_neighbors(n2.id, direction="in")
        assert len(neighbors) == 1
        assert neighbors[0].id == n1.id

    def test_get_decisions(self, tmp_kg):
        tmp_kg.add_node(KGNode("fact", KGNodeType.FACT))
        tmp_kg.add_node(KGNode("decision", KGNodeType.DECISION))
        decisions = tmp_kg.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].node_type == KGNodeType.DECISION

    def test_get_contradictions(self, tmp_kg):
        n1 = KGNode("high power", KGNodeType.FACT)
        n2 = KGNode("chose low power chip", KGNodeType.DECISION)
        tmp_kg.add_node(n1)
        tmp_kg.add_node(n2)
        tmp_kg.add_edge(KGEdge(n1.id, n2.id, KGEdgeType.CONTRADICTS, weight=0.8))
        contras = tmp_kg.get_contradictions()
        assert len(contras) == 1

    def test_find_paths(self, tmp_kg):
        n1 = KGNode("A", KGNodeType.FACT)
        n2 = KGNode("B", KGNodeType.FACT)
        n3 = KGNode("C", KGNodeType.DECISION)
        tmp_kg.add_node(n1)
        tmp_kg.add_node(n2)
        tmp_kg.add_node(n3)
        tmp_kg.add_edge(KGEdge(n1.id, n2.id, KGEdgeType.EXTENDS))
        tmp_kg.add_edge(KGEdge(n2.id, n3.id, KGEdgeType.SUPPORTS))
        paths = tmp_kg.find_paths(n1.id, n3.id)
        assert len(paths) == 1
        assert len(paths[0]) == 3

    def test_get_descendants(self, tmp_kg):
        n1 = KGNode("root", KGNodeType.FACT)
        n2 = KGNode("child", KGNodeType.FACT)
        tmp_kg.add_node(n1)
        tmp_kg.add_node(n2)
        tmp_kg.add_edge(KGEdge(n1.id, n2.id, KGEdgeType.EXTENDS))
        descs = tmp_kg.get_descendants(n1.id)
        assert n2.id in descs

    def test_persistence(self, tmp_kg):
        n = KGNode("persist me", KGNodeType.FACT)
        tmp_kg.add_node(n)
        n2 = KGNode("persist edge target", KGNodeType.DECISION)
        tmp_kg.add_node(n2)
        tmp_kg.add_edge(KGEdge(n.id, n2.id, KGEdgeType.SUPPORTS))

        # Create a new instance loading from same path
        kg2 = KnowledgeGraph(kg_path=tmp_kg._path)
        kg2.load()
        assert kg2.get_node(n.id) is not None
        assert kg2.get_node(n.id).content == "persist me"
        edges = kg2.get_edges(n2.id, direction="in")
        assert len(edges) == 1

    def test_save_compaction(self, tmp_kg):
        for i in range(5):
            tmp_kg.add_node(KGNode(f"node {i}", KGNodeType.FACT))
        tmp_kg.save()
        kg2 = KnowledgeGraph(kg_path=tmp_kg._path)
        kg2.load()
        assert len(kg2.get_all_nodes()) == 5

    def test_stats(self, tmp_kg):
        tmp_kg.add_node(KGNode("fact", KGNodeType.FACT))
        tmp_kg.add_node(KGNode("decision", KGNodeType.DECISION))
        s = tmp_kg.stats()
        assert s["total_nodes"] == 2
        assert s["total_edges"] == 0
        assert "fact" in s["node_types"]

    def test_find_node_by_content(self, tmp_kg):
        tmp_kg.add_node(KGNode("ESP32 WiFi module", KGNodeType.FACT))
        tmp_kg.add_node(KGNode("nRF52 BLE module", KGNodeType.FACT))
        results = tmp_kg.find_node_by_content("ESP32")
        assert len(results) == 1
        assert "ESP32" in results[0].content

    def test_get_communities(self, tmp_kg):
        n1 = KGNode("A", KGNodeType.FACT)
        n2 = KGNode("B", KGNodeType.FACT)
        tmp_kg.add_node(n1)
        tmp_kg.add_node(n2)
        tmp_kg.add_edge(KGEdge(n1.id, n2.id, KGEdgeType.EXTENDS))
        communities = tmp_kg.get_communities()
        assert len(communities) >= 1

    def test_get_mature_patterns(self, tmp_kg):
        n = KGNode("mature fact", KGNodeType.FACT, confidence=0.95)
        n.access_count = 15
        tmp_kg.add_node(n)
        mature = tmp_kg.get_mature_patterns(min_maturity=0.5)
        assert len(mature) >= 1

    def test_empty_graph(self, tmp_kg):
        assert tmp_kg.stats()["total_nodes"] == 0
        assert tmp_kg.get_decisions() == []
        assert tmp_kg.get_contradictions() == []

    def test_has_lock(self, tmp_kg):
        import threading
        assert type(tmp_kg._lock) is type(threading.Lock())

    def test_concurrent_add_nodes(self, tmp_kg):
        import threading
        from second_brain.knowledge_graph import KGNode, KGNodeType
        errors = []

        def add_node(i):
            try:
                tmp_kg.add_node(KGNode(f"concurrent node {i}", KGNodeType.FACT))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_node, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert tmp_kg.stats()["total_nodes"] == 10
