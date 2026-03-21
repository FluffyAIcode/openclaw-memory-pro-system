"""Tests for second_brain.inference module."""

import pytest
from unittest.mock import patch, MagicMock

from second_brain.knowledge_graph import (
    KGEdge, KGEdgeType, KGNode, KGNodeType, KnowledgeGraph,
)
from second_brain.inference import InferenceEngine


@pytest.fixture
def kg_with_data(tmp_path):
    """Create a KG with some test data for inference."""
    kg = KnowledgeGraph(kg_path=tmp_path)

    fact1 = KGNode("ESP32-S3 supports WiFi+BLE", KGNodeType.FACT, importance=0.8)
    fact2 = KGNode("ESP32-S3 功耗在深度睡眠下 10uA", KGNodeType.FACT, importance=0.7)
    fact3 = KGNode("nRF52840 功耗只有 1uA", KGNodeType.FACT, importance=0.7)
    decision = KGNode("选择 ESP32-S3 作为主芯片", KGNodeType.DECISION, importance=0.9)
    pref = KGNode("偏好低功耗方案", KGNodeType.PREFERENCE, importance=0.8)
    goal = KGNode("做一个 IoT 传感器网络", KGNodeType.GOAL, importance=0.9)
    question = KGNode("mesh 网络用什么协议?", KGNodeType.QUESTION, importance=0.6)

    for n in [fact1, fact2, fact3, decision, pref, goal, question]:
        kg.add_node(n)

    kg.add_edge(KGEdge(fact1.id, decision.id, KGEdgeType.SUPPORTS, weight=0.9))
    kg.add_edge(KGEdge(fact3.id, decision.id, KGEdgeType.CONTRADICTS, weight=0.7,
                        evidence="nRF52840 功耗远低于 ESP32"))
    kg.add_edge(KGEdge(pref.id, decision.id, KGEdgeType.CONTRADICTS, weight=0.5,
                        evidence="ESP32 不符合低功耗偏好"))
    kg.add_edge(KGEdge(decision.id, goal.id, KGEdgeType.SUPPORTS, weight=0.8))
    kg.add_edge(KGEdge(fact2.id, decision.id, KGEdgeType.SUPPORTS, weight=0.6))

    return kg, {
        "fact1": fact1, "fact2": fact2, "fact3": fact3,
        "decision": decision, "pref": pref, "goal": goal,
        "question": question,
    }


class TestContradictionDetection:
    def test_scan_finds_contradictions(self, kg_with_data):
        kg, nodes = kg_with_data
        engine = InferenceEngine(kg)
        reports = engine.scan_contradictions()
        assert len(reports) == 1
        r = reports[0]
        assert r.decision.id == nodes["decision"].id
        assert len(r.contradicting) == 2
        assert len(r.supporting) == 2
        assert r.risk_score > 0

    def test_risk_score_computation(self, kg_with_data):
        kg, nodes = kg_with_data
        engine = InferenceEngine(kg)
        reports = engine.scan_contradictions()
        r = reports[0]
        total_support = sum(e["weight"] for e in r.supporting)
        total_contra = sum(e["weight"] for e in r.contradicting)
        expected = total_contra / (total_support + total_contra + 0.001)
        assert abs(r.risk_score - expected) < 0.01

    def test_no_contradictions(self, tmp_path):
        kg = KnowledgeGraph(kg_path=tmp_path)
        n1 = KGNode("fact", KGNodeType.FACT)
        n2 = KGNode("decision", KGNodeType.DECISION)
        kg.add_node(n1)
        kg.add_node(n2)
        kg.add_edge(KGEdge(n1.id, n2.id, KGEdgeType.SUPPORTS))
        engine = InferenceEngine(kg)
        reports = engine.scan_contradictions()
        assert len(reports) == 0

    def test_to_dict(self, kg_with_data):
        kg, _ = kg_with_data
        engine = InferenceEngine(kg)
        reports = engine.scan_contradictions()
        d = reports[0].to_dict()
        assert "decision_id" in d
        assert "risk_score" in d
        assert "contradicting" in d


class TestAbsenceReasoning:
    def test_detect_blind_spots_with_llm(self, kg_with_data):
        kg, nodes = kg_with_data
        engine = InferenceEngine(kg)
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = '["成本", "功耗", "生态系统", "供应链稳定性"]'
        import sys
        with patch.dict(sys.modules, {"llm_client": mock_llm}):
            report = engine.detect_blind_spots(nodes["decision"].id)
            assert report is not None
            assert len(report.missing_dimensions) > 0

    def test_detect_blind_spots_heuristic(self, kg_with_data):
        kg, nodes = kg_with_data
        engine = InferenceEngine(kg)
        report = engine.detect_blind_spots(nodes["decision"].id)
        assert report is not None
        assert len(report.expected_dimensions) > 0
        assert len(report.missing_dimensions) > 0

    def test_detect_all_blind_spots(self, kg_with_data):
        kg, _ = kg_with_data
        engine = InferenceEngine(kg)
        reports = engine.detect_all_blind_spots()
        assert len(reports) >= 1

    def test_nonexistent_node(self, kg_with_data):
        kg, _ = kg_with_data
        engine = InferenceEngine(kg)
        report = engine.detect_blind_spots("nonexistent")
        assert report is None

    def test_non_decision_node(self, kg_with_data):
        kg, nodes = kg_with_data
        engine = InferenceEngine(kg)
        report = engine.detect_blind_spots(nodes["fact1"].id)
        assert report is None

    def test_to_dict(self, kg_with_data):
        kg, nodes = kg_with_data
        engine = InferenceEngine(kg)
        report = engine.detect_blind_spots(nodes["decision"].id)
        d = report.to_dict()
        assert "coverage_ratio" in d
        assert "missing" in d


class TestForwardPropagation:
    def test_propagate_new_fact(self, kg_with_data):
        kg, nodes = kg_with_data
        new_fact = KGNode("ESP32 实测功耗 50uA 远超规格", KGNodeType.FACT)
        kg.add_node(new_fact)
        kg.add_edge(KGEdge(new_fact.id, nodes["decision"].id,
                           KGEdgeType.CONTRADICTS, weight=0.9))
        engine = InferenceEngine(kg)
        alerts = engine.propagate(new_fact.id)
        assert len(alerts) >= 1
        decision_affected = any(
            a.affected_node.id == nodes["decision"].id for a in alerts
        )
        assert decision_affected

    def test_propagate_short_chain(self, kg_with_data):
        kg, nodes = kg_with_data
        new_fact = KGNode("WiFi 6 规范变更", KGNodeType.FACT)
        kg.add_node(new_fact)
        kg.add_edge(KGEdge(new_fact.id, nodes["decision"].id,
                           KGEdgeType.SUPPORTS, weight=0.6))
        engine = InferenceEngine(kg)
        alerts = engine.propagate(new_fact.id)
        goal_affected = any(
            a.affected_node.id == nodes["goal"].id for a in alerts
        )
        assert goal_affected

    def test_propagate_no_affected(self, tmp_path):
        kg = KnowledgeGraph(kg_path=tmp_path)
        n1 = KGNode("isolated", KGNodeType.FACT)
        kg.add_node(n1)
        engine = InferenceEngine(kg)
        alerts = engine.propagate(n1.id)
        assert len(alerts) == 0

    def test_to_dict(self, kg_with_data):
        kg, nodes = kg_with_data
        new = KGNode("new info", KGNodeType.FACT)
        kg.add_node(new)
        kg.add_edge(KGEdge(new.id, nodes["decision"].id, KGEdgeType.CONTRADICTS))
        engine = InferenceEngine(kg)
        alerts = engine.propagate(new.id)
        if alerts:
            d = alerts[0].to_dict()
            assert "message" in d
            assert "has_contradiction" in d


class TestThreadDiscovery:
    def test_discover_threads(self, kg_with_data):
        kg, _ = kg_with_data
        engine = InferenceEngine(kg)
        threads = engine.discover_threads()
        assert len(threads) >= 1

    def test_thread_has_title(self, kg_with_data):
        kg, _ = kg_with_data
        engine = InferenceEngine(kg)
        threads = engine.discover_threads()
        for t in threads:
            assert t.title != ""
            assert t.node_count > 0

    def test_thread_status(self, kg_with_data):
        kg, _ = kg_with_data
        engine = InferenceEngine(kg)
        threads = engine.discover_threads()
        for t in threads:
            assert t.status in ("exploring", "decided", "nascent", "developing", "active")

    def test_empty_graph(self, tmp_path):
        kg = KnowledgeGraph(kg_path=tmp_path)
        engine = InferenceEngine(kg)
        threads = engine.discover_threads()
        assert len(threads) == 0

    def test_to_dict(self, kg_with_data):
        kg, _ = kg_with_data
        engine = InferenceEngine(kg)
        threads = engine.discover_threads()
        if threads:
            d = threads[0].to_dict()
            assert "title" in d
            assert "node_count" in d
