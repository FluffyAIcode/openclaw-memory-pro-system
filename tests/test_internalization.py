"""Tests for second_brain.internalization module."""

import pytest
from pathlib import Path

from second_brain.knowledge_graph import (
    KGEdge, KGEdgeType, KGNode, KGNodeType, KnowledgeGraph,
)
from second_brain.internalization import InternalizationManager, Pattern, TrainingPair


@pytest.fixture
def kg_with_mature(tmp_path):
    """Create a KG with mature nodes for internalization."""
    kg = KnowledgeGraph(kg_path=tmp_path)

    pref = KGNode("偏好 Python 开发", KGNodeType.PREFERENCE,
                   importance=0.9, confidence=0.95)
    pref.access_count = 12
    pref.update_maturity()

    fact = KGNode("Python 3.9 是当前版本", KGNodeType.FACT,
                   importance=0.7, confidence=0.9)
    fact.access_count = 8
    fact.update_maturity()

    decision = KGNode("选择 FastAPI 作为后端框架", KGNodeType.DECISION,
                       importance=0.85, confidence=0.88)
    decision.access_count = 10
    decision.update_maturity()

    evidence = KGNode("FastAPI 性能优于 Flask", KGNodeType.FACT,
                       importance=0.7, confidence=0.8)
    evidence.access_count = 5
    evidence.update_maturity()

    for n in [pref, fact, decision, evidence]:
        kg.add_node(n)

    kg.add_edge(KGEdge(fact.id, pref.id, KGEdgeType.SUPPORTS, weight=0.7))
    kg.add_edge(KGEdge(evidence.id, decision.id, KGEdgeType.SUPPORTS, weight=0.8))

    return kg, {"pref": pref, "fact": fact, "decision": decision, "evidence": evidence}


class TestInternalizationManager:
    def test_find_candidates(self, kg_with_mature):
        kg, _ = kg_with_mature
        mgr = InternalizationManager(kg)
        candidates = mgr.find_candidates(min_maturity=0.3)
        assert len(candidates) >= 1

    def test_extract_patterns(self, kg_with_mature):
        kg, _ = kg_with_mature
        mgr = InternalizationManager(kg)
        candidates = mgr.find_candidates(min_maturity=0.3)
        patterns = mgr.extract_patterns(candidates)
        assert len(patterns) >= 1
        types = {p.pattern_type for p in patterns}
        assert types.intersection({"preference", "decision_rationale", "knowledge_anchor"})

    def test_preference_pattern(self, kg_with_mature):
        kg, nodes = kg_with_mature
        mgr = InternalizationManager(kg)
        pattern = mgr._build_pattern(nodes["pref"])
        assert pattern is not None
        assert pattern.pattern_type == "preference"
        assert "偏好" in pattern.summary

    def test_decision_pattern(self, kg_with_mature):
        kg, nodes = kg_with_mature
        mgr = InternalizationManager(kg)
        pattern = mgr._build_pattern(nodes["decision"])
        assert pattern is not None
        assert pattern.pattern_type == "decision_rationale"
        assert len(pattern.supporting_nodes) >= 1

    def test_fact_pattern(self, kg_with_mature):
        kg, nodes = kg_with_mature
        mgr = InternalizationManager(kg)
        pattern = mgr._build_pattern(nodes["fact"])
        assert pattern is not None
        assert pattern.pattern_type == "knowledge_anchor"

    def test_generate_training_data(self, kg_with_mature, tmp_path):
        kg, _ = kg_with_mature
        mgr = InternalizationManager(kg)
        mgr._training_dir = tmp_path / "training"
        candidates = mgr.find_candidates(min_maturity=0.3)
        patterns = mgr.extract_patterns(candidates)
        pairs = mgr.generate_training_data(patterns)
        assert len(pairs) >= 1
        assert all(isinstance(p, TrainingPair) for p in pairs)
        training_files = list((tmp_path / "training").glob("*.jsonl"))
        assert len(training_files) == 1

    def test_get_patterns_for_personality(self, kg_with_mature):
        kg, _ = kg_with_mature
        mgr = InternalizationManager(kg)
        formatted = mgr.get_patterns_for_personality()
        assert isinstance(formatted, list)
        if formatted:
            assert "type" in formatted[0]
            assert "content" in formatted[0]
            assert "maturity" in formatted[0]

    def test_status(self, kg_with_mature):
        kg, _ = kg_with_mature
        mgr = InternalizationManager(kg)
        s = mgr.status()
        assert "mature_candidates" in s
        assert "extracted_patterns" in s
        assert "kg_stats" in s

    def test_pattern_to_dict(self, kg_with_mature):
        kg, nodes = kg_with_mature
        mgr = InternalizationManager(kg)
        pattern = mgr._build_pattern(nodes["pref"])
        d = pattern.to_dict()
        assert "core_content" in d
        assert "pattern_type" in d
        assert "maturity" in d

    def test_training_pair_to_dict(self):
        pair = TrainingPair("question", "answer", "node123")
        d = pair.to_dict()
        assert d["instruction"] == "question"
        assert d["response"] == "answer"

    def test_empty_kg(self, tmp_path):
        kg = KnowledgeGraph(kg_path=tmp_path)
        mgr = InternalizationManager(kg)
        assert mgr.find_candidates() == []
        assert mgr.extract_patterns() == []
