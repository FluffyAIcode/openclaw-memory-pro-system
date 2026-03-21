"""Tests for second_brain.relation_extractor module."""

import json
import sys
import pytest
from unittest.mock import patch, MagicMock

from second_brain.knowledge_graph import KGNode, KGNodeType, KnowledgeGraph
from second_brain.relation_extractor import RelationExtractor, ExtractionResult


@pytest.fixture
def tmp_kg(tmp_path):
    return KnowledgeGraph(kg_path=tmp_path)


class TestExtractionResult:
    def test_empty_result(self):
        r = ExtractionResult.empty("too short")
        assert r.skipped is True
        assert r.reason == "too short"
        assert len(r.new_nodes) == 0

    def test_populated_result(self):
        r = ExtractionResult(
            new_nodes=[KGNode("test", KGNodeType.FACT)],
            new_edges=[],
        )
        assert r.skipped is False
        assert len(r.new_nodes) == 1


class TestRelationExtractor:
    def test_skip_short_content(self, tmp_kg):
        ext = RelationExtractor(tmp_kg)
        result = ext.extract("hi", importance=0.8)
        assert result.skipped is True
        assert "too short" in result.reason

    def test_skip_low_importance(self, tmp_kg):
        ext = RelationExtractor(tmp_kg)
        result = ext.extract("This is some content with enough length", importance=0.1)
        assert result.skipped is True
        assert "below threshold" in result.reason

    def _make_mock_llm(self, return_value=""):
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = return_value
        return mock_llm

    def test_extract_with_llm(self, tmp_kg):
        ext = RelationExtractor(tmp_kg)
        llm_response = json.dumps({
            "new_nodes": [
                {"content": "ESP32 选型完成", "type": "decision", "confidence": 0.9}
            ],
            "edges": []
        })
        mock_llm = self._make_mock_llm(llm_response)
        with patch.dict(sys.modules, {"llm_client": mock_llm}):
            result = ext.extract("我们决定用 ESP32-S3 作为主控芯片", importance=0.8)
            assert not result.skipped
            assert len(result.new_nodes) == 1
            assert result.new_nodes[0].node_type == KGNodeType.DECISION

    def test_extract_with_edges(self, tmp_kg):
        existing = KGNode("低功耗是关键需求", KGNodeType.PREFERENCE, node_id="pref001")
        tmp_kg.add_node(existing)

        ext = RelationExtractor(tmp_kg)
        llm_response = json.dumps({
            "new_nodes": [
                {"content": "选择 ESP32-S3", "type": "decision", "confidence": 0.8}
            ],
            "edges": [
                {"from_content": "pref001", "to_content": "选择 ESP32-S3",
                 "type": "contradicts", "weight": 0.7,
                 "evidence": "ESP32 功耗较高，与低功耗需求矛盾"}
            ]
        })
        mock_llm = self._make_mock_llm(llm_response)
        with patch.dict(sys.modules, {"llm_client": mock_llm}):
            result = ext.extract("决定采用 ESP32-S3 方案", importance=0.8)
            assert len(result.new_nodes) == 1
            assert len(result.new_edges) == 1

    def test_llm_unavailable(self, tmp_kg):
        ext = RelationExtractor(tmp_kg)
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = False
        with patch.dict(sys.modules, {"llm_client": mock_llm}):
            result = ext.extract("important content here with enough length",
                                 importance=0.8)
            assert result.skipped is True

    def test_llm_returns_empty(self, tmp_kg):
        ext = RelationExtractor(tmp_kg)
        mock_llm = self._make_mock_llm("")
        with patch.dict(sys.modules, {"llm_client": mock_llm}):
            result = ext.extract("some content with enough length here", importance=0.8)
            assert result.skipped is True

    def test_parse_malformed_json(self, tmp_kg):
        ext = RelationExtractor(tmp_kg)
        mock_llm = self._make_mock_llm("not valid json at all")
        with patch.dict(sys.modules, {"llm_client": mock_llm}):
            result = ext.extract("some content with length", importance=0.8)
            assert result.skipped is True

    def test_parse_json_with_markdown(self, tmp_kg):
        ext = RelationExtractor(tmp_kg)
        llm_response = '```json\n{"new_nodes": [{"content": "test", "type": "fact"}], "edges": []}\n```'
        mock_llm = self._make_mock_llm(llm_response)
        with patch.dict(sys.modules, {"llm_client": mock_llm}):
            result = ext.extract("content with enough text here", importance=0.8)
            assert not result.skipped
            assert len(result.new_nodes) == 1

    def test_fallback_candidates(self, tmp_kg):
        tmp_kg.add_node(KGNode("ESP32 WiFi", KGNodeType.FACT))
        tmp_kg.add_node(KGNode("nRF52 BLE", KGNodeType.FACT))
        ext = RelationExtractor(tmp_kg)
        candidates = ext._fallback_candidates("ESP32 module", tmp_kg.get_all_nodes(), 5)
        assert len(candidates) <= 5
