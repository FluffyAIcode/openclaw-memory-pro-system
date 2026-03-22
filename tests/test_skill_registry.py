"""Tests for skill_registry — registry, Skill, SkillStatus."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestSkillStatus:

    def test_enum_values(self):
        from skill_registry.registry import SkillStatus
        assert SkillStatus.DRAFT.value == "draft"
        assert SkillStatus.ACTIVE.value == "active"
        assert SkillStatus.DEPRECATED.value == "deprecated"


class TestSkill:

    def test_create_defaults(self):
        from skill_registry.registry import Skill, SkillStatus
        s = Skill("test skill", "content here")
        assert s.name == "test skill"
        assert s.content == "content here"
        assert s.status == SkillStatus.DRAFT
        assert s.version == 1
        assert len(s.id) == 12

    def test_to_dict_roundtrip(self):
        from skill_registry.registry import Skill
        s = Skill("test", "body", tags=["python", "ml"],
                  source_memories=["mem1"])
        d = s.to_dict()
        s2 = Skill.from_dict(d)
        assert s2.name == s.name
        assert s2.tags == s.tags
        assert s2.source_memories == s.source_memories
        assert s2.id == s.id

    def test_repr(self):
        from skill_registry.registry import Skill
        s = Skill("test", "body")
        r = repr(s)
        assert "test" in r
        assert "draft" in r


class TestSkillRegistry:

    def test_add_and_list(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("skill1", "content1", tags=["python"])
        assert s.name == "skill1"
        assert len(reg.list_all()) == 1

    def test_promote(self, tmp_path):
        from skill_registry.registry import SkillRegistry, SkillStatus
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("skill1", "content1")
        result = reg.promote(s.id)
        assert result.status == SkillStatus.ACTIVE
        assert len(reg.list_active()) == 1

    def test_deprecate(self, tmp_path):
        from skill_registry.registry import SkillRegistry, SkillStatus
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("skill1", "content1")
        reg.promote(s.id)
        result = reg.deprecate(s.id)
        assert result.status == SkillStatus.DEPRECATED
        assert len(reg.list_active()) == 0

    def test_update_content(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("skill1", "v1")
        result = reg.update_content(s.id, "v2")
        assert result.content == "v2"
        assert result.version == 2

    def test_get(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("skill1", "content1")
        assert reg.get(s.id).name == "skill1"
        assert reg.get("nonexistent") is None

    def test_search(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        reg.add("python tips", "use list comprehension")
        reg.add("rust basics", "ownership model")
        results = reg.search("python")
        assert len(results) == 1
        assert results[0].name == "python tips"

    def test_stats(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        reg.add("s1", "c1")
        s2 = reg.add("s2", "c2")
        reg.promote(s2.id)
        st = reg.stats()
        assert st["total"] == 2
        assert st["by_status"]["draft"] == 1
        assert st["by_status"]["active"] == 1

    def test_persistence(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg1 = SkillRegistry(registry_dir=tmp_path / "skills")
        reg1.add("s1", "c1")
        s2 = reg1.add("s2", "c2")
        reg1.promote(s2.id)

        reg2 = SkillRegistry(registry_dir=tmp_path / "skills")
        assert len(reg2.list_all()) == 2
        active = reg2.list_active()
        assert len(active) == 1
        assert active[0].name == "s2"

    def test_promote_nonexistent(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        assert reg.promote("nonexistent") is None

    def test_deprecate_nonexistent(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        assert reg.deprecate("nonexistent") is None

    def test_update_nonexistent(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        assert reg.update_content("nonexistent", "x") is None

    def test_corrupt_jsonl(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        d = tmp_path / "skills"
        d.mkdir()
        (d / "registry.jsonl").write_text("bad json\n")
        reg = SkillRegistry(registry_dir=d)
        assert len(reg.list_all()) == 0


# ═══════════════════════════════════════════════════════════
# KG structural_gain
# ═══════════════════════════════════════════════════════════

class TestStructuralGain:

    def test_empty_extraction_returns_zero(self):
        from second_brain.relation_extractor import ExtractionResult
        r = ExtractionResult.empty("test")
        assert r.structural_gain == 0.0

    def test_gain_with_contradiction(self, tmp_path):
        from second_brain.knowledge_graph import KnowledgeGraph, KGNode, KGNodeType, KGEdge, KGEdgeType
        from second_brain.relation_extractor import RelationExtractor

        kg = KnowledgeGraph(kg_path=tmp_path / "kg")
        existing = KGNode("AI 需要大量数据", KGNodeType.FACT)
        kg.add_node(existing)

        extractor = RelationExtractor(knowledge_graph=kg)
        new_nodes = [KGNode("小样本学习可以减少数据需求", KGNodeType.FACT)]
        new_edges = [KGEdge("placeholder", existing.id, KGEdgeType.CONTRADICTS)]

        node_id_map = {new_nodes[0].content: kg.add_node(new_nodes[0])}
        new_edges[0].source_id = node_id_map[new_nodes[0].content]
        kg.add_edge(new_edges[0])

        gain = extractor._calc_structural_gain(new_nodes, new_edges, node_id_map)
        assert gain >= 0.5, f"Expected ≥0.5 with contradiction, got {gain}"

    def test_gain_with_no_connections(self, tmp_path):
        from second_brain.knowledge_graph import KnowledgeGraph, KGNode, KGNodeType
        from second_brain.relation_extractor import RelationExtractor

        kg = KnowledgeGraph(kg_path=tmp_path / "kg")
        extractor = RelationExtractor(knowledge_graph=kg)

        new_nodes = [KGNode("孤立事实", KGNodeType.FACT)]
        node_id_map = {new_nodes[0].content: kg.add_node(new_nodes[0])}

        gain = extractor._calc_structural_gain(new_nodes, [], node_id_map)
        assert gain == 0.0


# ═══════════════════════════════════════════════════════════
# Digest compression_value
# ═══════════════════════════════════════════════════════════

class TestCompressionValue:

    def test_no_summary_gives_low_score(self):
        from second_brain.digest import _calc_compression_value
        score = _calc_compression_value("raw text " * 100, None, "", 3, 7)
        assert score == 0.1

    def test_good_summary_scores_higher(self):
        from second_brain.digest import _calc_compression_value
        raw = "记忆日志内容" * 200
        summary = "关键决策：选择了方案A。重要发现：系统性能提升30%。确认用户偏好深色主题。"
        score = _calc_compression_value(raw, summary, "", 5, 7)
        assert score > 0.3, f"Expected >0.3 for good summary, got {score}"

    def test_compare_with_last_digest_no_prior(self, tmp_path):
        from second_brain.digest import _compare_with_last_digest
        import second_brain.digest as dmod
        orig = dmod.config
        mock_cfg = MagicMock()
        mock_cfg.long_term_dir = tmp_path / "long_term"
        mock_cfg.long_term_dir.mkdir()
        dmod.config = mock_cfg
        try:
            score = _compare_with_last_digest("new summary content")
            assert score == 0.8
        finally:
            dmod.config = orig


# ═══════════════════════════════════════════════════════════
# Skill Proposer
# ═══════════════════════════════════════════════════════════

class TestSkillProposer:

    def test_no_scores_means_no_proposal(self, tmp_path):
        from second_brain.skill_proposer import SkillProposer
        import second_brain.skill_proposer as sp_mod
        orig = sp_mod._WORKSPACE
        sp_mod._WORKSPACE = tmp_path
        try:
            p = SkillProposer()
            import second_brain.skill_proposer as spmod
            orig_config = spmod.config
            mock_cfg = MagicMock()
            mock_cfg.long_term_dir = tmp_path / "lt"
            mock_cfg.insights_path = tmp_path / "insights"
            mock_cfg.kg_maturity_threshold = 0.7
            spmod.config = mock_cfg
            try:
                results = p.scan_and_propose(days=7)
                assert results == []
            finally:
                spmod.config = orig_config
        finally:
            sp_mod._WORKSPACE = orig

    def test_two_of_three_triggers_proposal(self, tmp_path):
        from second_brain.skill_proposer import SkillProposer
        from unittest.mock import patch

        p = SkillProposer()
        with patch.object(p, "_get_best_kg_score", return_value=0.8), \
             patch.object(p, "_get_best_digest_score", return_value=0.9), \
             patch.object(p, "_get_best_collision_score", return_value=2), \
             patch.object(p, "_build_proposals") as mock_build, \
             patch.object(p, "_register_draft") as mock_reg:
            from second_brain.skill_proposer import SkillProposal
            proposal = SkillProposal("test", "content",
                                     sources={"kg": 0.8, "digest": 0.9})
            mock_build.return_value = [proposal]
            mock_reg.return_value = MagicMock()

            results = p.scan_and_propose(days=7)
            assert len(results) == 1
            mock_build.assert_called_once()
            mock_reg.assert_called_once()

    def test_one_of_three_not_enough(self, tmp_path):
        from second_brain.skill_proposer import SkillProposer
        from unittest.mock import patch

        p = SkillProposer()
        with patch.object(p, "_get_best_kg_score", return_value=0.8), \
             patch.object(p, "_get_best_digest_score", return_value=0.3), \
             patch.object(p, "_get_best_collision_score", return_value=2):
            results = p.scan_and_propose(days=7)
            assert results == []

    def test_save_and_read_kg_score(self, tmp_path):
        from second_brain.skill_proposer import save_kg_score, SkillProposer
        import second_brain.skill_proposer as sp_mod
        orig = sp_mod._WORKSPACE
        sp_mod._WORKSPACE = tmp_path

        (tmp_path / "memory" / "kg").mkdir(parents=True)
        save_kg_score(0.75, "test content")
        save_kg_score(0.45, "low score")

        p = SkillProposer()
        best = p._get_best_kg_score(days=1)
        sp_mod._WORKSPACE = orig
        assert best == 0.75
