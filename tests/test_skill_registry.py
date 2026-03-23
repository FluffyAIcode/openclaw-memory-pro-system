"""Tests for skill_registry — registry, Skill, SkillStatus."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


    def test_record_feedback_success(self, tmp_path):
        from skill_registry.registry import SkillRegistry, SkillStatus
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("skill1", "content1")
        reg.promote(s.id)
        result = reg.record_feedback(s.id, "test query", "success")
        assert result.successes == 1
        assert result.failures == 0
        assert result.utility_rate == 1.0

    def test_record_feedback_failure(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("skill1", "content1")
        reg.promote(s.id)
        reg.record_feedback(s.id, "q1", "failure")
        reg.record_feedback(s.id, "q2", "failure")
        skill = reg.get(s.id)
        assert skill.failures == 2
        assert skill.utility_rate == 0.0

    def test_record_feedback_nonexistent(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        assert reg.record_feedback("nonexistent", "q", "success") is None

    def test_utility_rate_default(self):
        from skill_registry.registry import Skill
        s = Skill("test", "content")
        assert s.utility_rate == 0.5
        assert s.total_uses == 0

    def test_utility_rate_calculated(self):
        from skill_registry.registry import Skill
        s = Skill("test", "content", successes=3, failures=1)
        assert s.utility_rate == 0.75
        assert s.total_uses == 4

    def test_usage_log_written(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("skill1", "content1")
        reg.record_feedback(s.id, "my query", "success", "worked well")
        log_file = tmp_path / "skills" / "usage_log.jsonl"
        assert log_file.exists()
        import json
        entry = json.loads(log_file.read_text().strip())
        assert entry["skill_id"] == s.id
        assert entry["query"] == "my query"
        assert entry["outcome"] == "success"

    def test_get_usage_stats(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s1 = reg.add("skill1", "c1")
        s2 = reg.add("skill2", "c2")
        reg.record_feedback(s1.id, "q", "success")
        reg.record_feedback(s1.id, "q", "failure")
        stats = reg.get_usage_stats()
        assert len(stats) == 1
        assert stats[0]["id"] == s1.id
        assert stats[0]["total_uses"] == 2

    def test_trigger_rewrite_on_low_utility(self, tmp_path):
        from skill_registry.registry import SkillRegistry, SkillStatus
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("bad skill", "incorrect info")
        reg.promote(s.id)
        reg.record_feedback(s.id, "q1", "failure", "wrong answer")
        reg.record_feedback(s.id, "q2", "failure", "not helpful")
        with patch.object(reg, "_trigger_rewrite") as mock_rw:
            reg.record_feedback(s.id, "q3", "failure", "still bad")
            assert mock_rw.called

    def test_structured_content(self):
        from skill_registry.registry import Skill
        s = Skill("AI Memory", "核心知识内容",
                   prerequisites="需要向量数据库",
                   procedures="1. 嵌入 2. 存储 3. 召回",
                   applicable_scenarios="碎片化学习",
                   inapplicable_scenarios="实时流式数据",
                   tags=["ai", "memory"],
                   successes=5, failures=1)
        text = s.structured_content()
        assert "# AI Memory" in text
        assert "## 前提条件" in text
        assert "## 核心知识" in text
        assert "## 操作步骤" in text
        assert "## 适用场景" in text
        assert "## 不适用场景" in text
        assert "83%" in text

    def test_to_dict_includes_new_fields(self):
        from skill_registry.registry import Skill
        s = Skill("test", "content",
                   successes=2, failures=1,
                   prerequisites="prereq",
                   procedures="proc",
                   applicable_scenarios="good",
                   inapplicable_scenarios="bad")
        d = s.to_dict()
        assert d["successes"] == 2
        assert d["failures"] == 1
        assert d["utility_rate"] == round(2/3, 3)
        assert d["total_uses"] == 3
        assert d["prerequisites"] == "prereq"
        assert d["procedures"] == "proc"

    def test_from_dict_with_new_fields(self):
        from skill_registry.registry import Skill
        d = {
            "name": "test", "content": "c",
            "successes": 5, "failures": 2,
            "prerequisites": "p", "procedures": "pr",
            "applicable_scenarios": "a", "inapplicable_scenarios": "i",
        }
        s = Skill.from_dict(d)
        assert s.successes == 5
        assert s.failures == 2
        assert s.prerequisites == "p"

    def test_stats_includes_usage(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("s1", "c1")
        reg.record_feedback(s.id, "q", "success")
        st = reg.stats()
        assert st["total_uses"] == 1
        assert st["avg_utility"] == 1.0

    def test_add_with_structured_fields(self, tmp_path):
        from skill_registry.registry import SkillRegistry
        reg = SkillRegistry(registry_dir=tmp_path / "skills")
        s = reg.add("structured", "content",
                     prerequisites="need X",
                     procedures="step 1, step 2",
                     applicable_scenarios="scenario A")
        assert s.prerequisites == "need X"
        assert s.procedures == "step 1, step 2"


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
