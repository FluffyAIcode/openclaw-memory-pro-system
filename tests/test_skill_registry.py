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
