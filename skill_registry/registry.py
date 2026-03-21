"""
Skill Registry — the bridge between memories and actionable capabilities.

A Skill is a distilled, versioned, reusable piece of knowledge or procedure
that was promoted from the memory corpus. Skills can be:
  - draft:      auto-extracted, awaiting review
  - active:     confirmed and usable by agents
  - deprecated: superseded or invalidated

Persistence: JSONL at memory/skills/registry.jsonl
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).parent.parent


class SkillStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class Skill:
    __slots__ = ("id", "name", "content", "status", "tags",
                 "source_memories", "version", "created_at", "updated_at")

    def __init__(self, name: str, content: str, *,
                 skill_id: str = "",
                 status: SkillStatus = SkillStatus.DRAFT,
                 tags: List[str] = None,
                 source_memories: List[str] = None,
                 version: int = 1,
                 created_at: str = "",
                 updated_at: str = ""):
        self.id = skill_id or uuid.uuid4().hex[:12]
        self.name = name
        self.content = content
        self.status = SkillStatus(status) if isinstance(status, str) else status
        self.tags = tags or []
        self.source_memories = source_memories or []
        self.version = version
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or self.created_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "status": self.status.value,
            "tags": self.tags,
            "source_memories": self.source_memories,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            name=d["name"],
            content=d["content"],
            skill_id=d.get("id", ""),
            status=d.get("status", "draft"),
            tags=d.get("tags", []),
            source_memories=d.get("source_memories", []),
            version=d.get("version", 1),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )

    def __repr__(self):
        return f"Skill({self.name!r}, status={self.status.value}, v{self.version})"


class SkillRegistry:
    """JSONL-backed skill store with promote/deprecate lifecycle."""

    def __init__(self, registry_dir: Path = None):
        self._dir = registry_dir or (_WORKSPACE / "memory" / "skills")
        self._file = self._dir / "registry.jsonl"
        self._skills: Optional[Dict[str, Skill]] = None

    def _ensure_dir(self):
        self._dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, Skill]:
        if self._skills is not None:
            return self._skills
        self._skills = {}
        if self._file.exists():
            for line in self._file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        d = json.loads(line)
                        s = Skill.from_dict(d)
                        self._skills[s.id] = s
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipped bad skill entry: %s", e)
        return self._skills

    def _save_all(self):
        self._ensure_dir()
        with open(self._file, "w", encoding="utf-8") as f:
            for s in self._load().values():
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    def add(self, name: str, content: str, *,
            tags: List[str] = None,
            source_memories: List[str] = None,
            status: SkillStatus = SkillStatus.DRAFT) -> Skill:
        """Register a new skill (defaults to draft)."""
        skill = Skill(name, content, tags=tags,
                      source_memories=source_memories, status=status)
        skills = self._load()
        skills[skill.id] = skill
        self._ensure_dir()
        with open(self._file, "a", encoding="utf-8") as f:
            f.write(json.dumps(skill.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Skill registered: %s (%s)", skill.name, skill.id)
        return skill

    def promote(self, skill_id: str) -> Optional[Skill]:
        """Promote a draft skill to active."""
        skills = self._load()
        skill = skills.get(skill_id)
        if not skill:
            return None
        skill.status = SkillStatus.ACTIVE
        skill.updated_at = datetime.now().isoformat()
        self._save_all()
        logger.info("Skill promoted: %s → active", skill.name)
        return skill

    def deprecate(self, skill_id: str) -> Optional[Skill]:
        """Mark a skill as deprecated."""
        skills = self._load()
        skill = skills.get(skill_id)
        if not skill:
            return None
        skill.status = SkillStatus.DEPRECATED
        skill.updated_at = datetime.now().isoformat()
        self._save_all()
        logger.info("Skill deprecated: %s", skill.name)
        return skill

    def update_content(self, skill_id: str, content: str) -> Optional[Skill]:
        """Update skill content and bump version."""
        skills = self._load()
        skill = skills.get(skill_id)
        if not skill:
            return None
        skill.content = content
        skill.version += 1
        skill.updated_at = datetime.now().isoformat()
        self._save_all()
        logger.info("Skill updated: %s → v%d", skill.name, skill.version)
        return skill

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._load().get(skill_id)

    def list_active(self) -> List[Skill]:
        return [s for s in self._load().values()
                if s.status == SkillStatus.ACTIVE]

    def list_all(self) -> List[Skill]:
        return list(self._load().values())

    def search(self, query: str) -> List[Skill]:
        """Simple keyword search across skill names and content."""
        q = query.lower()
        return [s for s in self._load().values()
                if q in s.name.lower() or q in s.content.lower()]

    def stats(self) -> dict:
        skills = self._load()
        by_status = {}
        for s in skills.values():
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
        return {"total": len(skills), "by_status": by_status}


registry = SkillRegistry()
