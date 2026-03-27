"""
Skill Registry — the bridge between memories and actionable capabilities.

A Skill is a distilled, versioned, reusable piece of knowledge or procedure
that was promoted from the memory corpus. Skills can be:
  - draft:      auto-extracted, awaiting review
  - active:     confirmed and usable by agents
  - deprecated: superseded or invalidated

Lifecycle with utility tracking (Memento-Skills inspired):
  - Each recall of a skill logs a usage event (query, outcome)
  - utility_rate = successes / (successes + failures)
  - When utility_rate < LOW_UTILITY_THRESHOLD, trigger LLM-based rewrite
  - Rewrite bumps version and resets counters

Persistence: JSONL at memory/skills/registry.jsonl
Usage log:   JSONL at memory/skills/usage_log.jsonl
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).parent.parent

LOW_UTILITY_THRESHOLD = 0.3
MIN_USES_FOR_REWRITE = 3


class SkillStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class Skill:
    __slots__ = ("id", "name", "content", "status", "tags",
                 "source_memories", "version", "created_at", "updated_at",
                 "successes", "failures", "prerequisites", "procedures",
                 "applicable_scenarios", "inapplicable_scenarios",
                 "action_type", "action_config")

    # action_type values:
    #   "none"            — passive knowledge text (default, backward-compatible)
    #   "prompt_template" — agent can use this as a system prompt
    #   "tool_call"       — binds to an OpenClaw tool
    #   "webhook"         — calls an external HTTP endpoint

    def __init__(self, name: str, content: str, *,
                 skill_id: str = "",
                 status: SkillStatus = SkillStatus.DRAFT,
                 tags: List[str] = None,
                 source_memories: List[str] = None,
                 version: int = 1,
                 created_at: str = "",
                 updated_at: str = "",
                 successes: int = 0,
                 failures: int = 0,
                 prerequisites: str = "",
                 procedures: str = "",
                 applicable_scenarios: str = "",
                 inapplicable_scenarios: str = "",
                 action_type: str = "none",
                 action_config: Dict = None):
        self.id = skill_id or uuid.uuid4().hex[:12]
        self.name = name
        self.content = content
        self.status = SkillStatus(status) if isinstance(status, str) else status
        self.tags = tags or []
        self.source_memories = source_memories or []
        self.version = version
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or self.created_at
        self.successes = successes
        self.failures = failures
        self.prerequisites = prerequisites
        self.procedures = procedures
        self.applicable_scenarios = applicable_scenarios
        self.inapplicable_scenarios = inapplicable_scenarios
        self.action_type = action_type
        self.action_config = action_config or {}

    @property
    def utility_rate(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.5
        return self.successes / total

    @property
    def total_uses(self) -> int:
        return self.successes + self.failures

    def structured_content(self) -> str:
        """Return SKILL.md-style structured representation."""
        parts = [f"# {self.name}\n"]
        if self.prerequisites:
            parts.append(f"## 前提条件\n{self.prerequisites}\n")
        parts.append(f"## 核心知识\n{self.content}\n")
        if self.procedures:
            parts.append(f"## 操作步骤\n{self.procedures}\n")
        if self.applicable_scenarios:
            parts.append(f"## 适用场景\n{self.applicable_scenarios}\n")
        if self.inapplicable_scenarios:
            parts.append(f"## 不适用场景\n{self.inapplicable_scenarios}\n")
        if self.tags:
            parts.append(f"## 标签\n{', '.join(self.tags)}\n")
        parts.append(f"## 元信息\n- utility: {self.utility_rate:.0%} ({self.total_uses} uses)\n"
                     f"- version: v{self.version}\n")
        return "\n".join(parts)

    def executable_prompt(self) -> Optional[str]:
        """Build an executable system prompt if action_type is prompt_template."""
        if self.action_type != "prompt_template" or not self.action_config:
            return None
        tpl = self.action_config.get("template", "")
        if not tpl:
            return None
        try:
            return tpl.format(
                name=self.name, content=self.content,
                procedures=self.procedures, prerequisites=self.prerequisites,
                applicable_scenarios=self.applicable_scenarios,
            )
        except (KeyError, IndexError):
            return tpl

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "status": self.status.value,
            "tags": self.tags,
            "source_memories": self.source_memories,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "successes": self.successes,
            "failures": self.failures,
            "utility_rate": round(self.utility_rate, 3),
            "total_uses": self.total_uses,
            "prerequisites": self.prerequisites,
            "procedures": self.procedures,
            "applicable_scenarios": self.applicable_scenarios,
            "inapplicable_scenarios": self.inapplicable_scenarios,
            "action_type": self.action_type,
            "action_config": self.action_config,
        }
        prompt = self.executable_prompt()
        if prompt:
            d["executable_prompt"] = prompt
        return d

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
            successes=d.get("successes", 0),
            failures=d.get("failures", 0),
            prerequisites=d.get("prerequisites", ""),
            procedures=d.get("procedures", ""),
            applicable_scenarios=d.get("applicable_scenarios", ""),
            inapplicable_scenarios=d.get("inapplicable_scenarios", ""),
            action_type=d.get("action_type", "none"),
            action_config=d.get("action_config", {}),
        )

    def __repr__(self):
        return (f"Skill({self.name!r}, status={self.status.value}, "
                f"v{self.version}, utility={self.utility_rate:.0%})")


class SkillRegistry:
    """JSONL-backed skill store with promote/deprecate lifecycle and utility tracking."""

    def __init__(self, registry_dir: Path = None):
        self._dir = registry_dir or (_WORKSPACE / "memory" / "skills")
        self._file = self._dir / "registry.jsonl"
        self._usage_log = self._dir / "usage_log.jsonl"
        self._skills: Optional[Dict[str, Skill]] = None
        self._lock = threading.Lock()

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
        with self._lock:
            with open(self._file, "w", encoding="utf-8") as f:
                for s in self._load().values():
                    f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    def add(self, name: str, content: str, *,
            tags: List[str] = None,
            source_memories: List[str] = None,
            status: SkillStatus = SkillStatus.DRAFT,
            prerequisites: str = "",
            procedures: str = "",
            applicable_scenarios: str = "",
            inapplicable_scenarios: str = "",
            action_type: str = "",
            action_config: Dict = None) -> Skill:
        """Register a new skill (defaults to draft). Auto-generates prompt_template if not provided."""
        try:
            from memory_security import check_content_safety, SecurityAuditLogger
            scan_text = f"{name} {content} {procedures} {prerequisites}"
            safety = check_content_safety(scan_text)
            if not safety.is_safe:
                _audit = SecurityAuditLogger()
                _audit.log_write_rejection(
                    "skill_add_blocked", safety.reason,
                    scan_text[:200], "skill_registry")
                raise ValueError(f"Skill content rejected: {safety.reason}")
        except ImportError:
            pass

        if not action_type and (procedures or content):
            action_type = "prompt_template"
            action_config = self._build_default_prompt_template(
                name, content, prerequisites, procedures, applicable_scenarios)
        skill = Skill(name, content, tags=tags,
                      source_memories=source_memories, status=status,
                      prerequisites=prerequisites,
                      procedures=procedures,
                      applicable_scenarios=applicable_scenarios,
                      inapplicable_scenarios=inapplicable_scenarios,
                      action_type=action_type or "none",
                      action_config=action_config or {})
        skills = self._load()
        skills[skill.id] = skill
        self._ensure_dir()
        with self._lock:
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(json.dumps(skill.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Skill registered: %s (%s)", skill.name, skill.id)
        return skill

    @staticmethod
    def _build_default_prompt_template(name, content, prerequisites, procedures, applicable) -> Dict:
        parts = [f"You are an expert in「{name}」."]
        if prerequisites:
            parts.append(f"Prerequisites: {prerequisites}")
        if procedures:
            parts.append(f"Steps:\n{procedures}")
        else:
            parts.append(f"Knowledge:\n{content[:500]}")
        if applicable:
            parts.append(f"Apply when: {applicable}")
        parts.append("Use the above to help the user solve their problem.")
        return {"template": "\n\n".join(parts)}

    _PROMOTE_COOLDOWN_HOURS = 1.0

    def promote(self, skill_id: str, force: bool = False) -> Optional[Skill]:
        """Promote a draft skill to active.

        Requires the skill to have existed for at least _PROMOTE_COOLDOWN_HOURS
        unless force=True. Re-scans content for injection patterns.
        """
        skills = self._load()
        skill = skills.get(skill_id)
        if not skill:
            return None

        if not force:
            try:
                created = datetime.fromisoformat(skill.created_at)
                age_hours = (datetime.now() - created).total_seconds() / 3600
                if age_hours < self._PROMOTE_COOLDOWN_HOURS:
                    logger.warning("Skill promote blocked: %s is %.1f hours old (min=%.1f)",
                                   skill.name, age_hours, self._PROMOTE_COOLDOWN_HOURS)
                    return None
            except (ValueError, TypeError):
                pass

        try:
            from memory_security import check_content_safety, SecurityAuditLogger
            scan_text = f"{skill.name} {skill.content} {skill.procedures}"
            exec_prompt = skill.executable_prompt()
            if exec_prompt:
                scan_text += f" {exec_prompt}"
            safety = check_content_safety(scan_text)
            if not safety.is_safe:
                _audit = SecurityAuditLogger()
                _audit.log_write_rejection(
                    "skill_promote_blocked", safety.reason,
                    scan_text[:200], "skill_registry")
                logger.warning("Skill promote blocked (injection): %s — %s",
                               skill.name, safety.reason)
                return None
        except ImportError:
            pass

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

    def record_feedback(self, skill_id: str, query: str,
                        outcome: str, context: str = "") -> Optional[Skill]:
        """Record a usage outcome for a skill.

        Args:
            skill_id: Which skill was used
            query: The query that triggered recall
            outcome: "success" or "failure"
            context: Optional context about why it succeeded/failed
        """
        skills = self._load()
        skill = skills.get(skill_id)
        if not skill:
            return None

        if outcome == "success":
            skill.successes += 1
        else:
            skill.failures += 1
        skill.updated_at = datetime.now().isoformat()
        self._save_all()

        self._append_usage_log(skill_id, query, outcome, context)

        logger.info("Skill feedback: %s → %s (utility=%.0f%%, %d uses)",
                     skill.name, outcome, skill.utility_rate * 100, skill.total_uses)

        if (skill.utility_rate < LOW_UTILITY_THRESHOLD
                and skill.total_uses >= MIN_USES_FOR_REWRITE
                and skill.status == SkillStatus.ACTIVE):
            self._trigger_rewrite(skill)

        return skill

    def _append_usage_log(self, skill_id: str, query: str,
                          outcome: str, context: str):
        """Append a structured usage event for future router training."""
        self._ensure_dir()
        entry = {
            "skill_id": skill_id,
            "query": query,
            "outcome": outcome,
            "context": context[:500] if context else "",
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with self._lock:
                with open(self._usage_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write usage log: %s", e)

    def _trigger_rewrite(self, skill: Skill):
        """Use LLM to rewrite a low-utility skill based on failure context."""
        logger.info("Skill rewrite triggered: %s (utility=%.0f%%)",
                     skill.name, skill.utility_rate * 100)
        try:
            failures = self._get_recent_failures(skill.id, limit=5)
            if not failures:
                return

            import llm_client
            if not llm_client.is_available():
                logger.debug("LLM unavailable, skipping skill rewrite")
                return

            failure_context = "\n".join(
                f"- 查询: {f['query']}\n  原因: {f.get('context', '未知')}"
                for f in failures
            )

            prompt = (
                f"以下技能的实际使用效果很差（utility={skill.utility_rate:.0%}）。\n\n"
                f"技能名称: {skill.name}\n"
                f"当前内容:\n{skill.content[:1500]}\n\n"
                f"最近失败记录:\n{failure_context}\n\n"
                f"请改写这个技能，使它更精确、更有用。保持核心知识，但：\n"
                f"1. 修正不准确的信息\n"
                f"2. 添加使用前提条件\n"
                f"3. 明确适用和不适用的场景\n"
                f"4. 如果有可操作步骤，列出来\n\n"
                f"只输出改写后的技能内容，不要额外解释。"
            )

            rewritten = llm_client.generate(
                prompt=prompt,
                system="你是一个技能优化器，目标是让技能更精确、更实用。",
                max_tokens=1500,
                temperature=0.3,
            )

            if rewritten and len(rewritten.strip()) > 50:
                try:
                    from memory_security import check_content_safety, SecurityAuditLogger
                    safety = check_content_safety(rewritten)
                    if not safety.is_safe:
                        _audit = SecurityAuditLogger()
                        _audit.log_write_rejection(
                            "skill_rewrite_blocked", safety.reason,
                            rewritten[:200], "skill_rewrite")
                        logger.warning("Skill rewrite rejected (injection): %s — %s",
                                       skill.name, safety.reason)
                        return
                except ImportError:
                    pass

                skill.content = rewritten.strip()
                skill.version += 1
                skill.successes = 0
                skill.failures = 0
                skill.updated_at = datetime.now().isoformat()
                self._save_all()
                logger.info("Skill rewritten: %s → v%d", skill.name, skill.version)
        except Exception as e:
            logger.warning("Skill rewrite failed for %s: %s", skill.name, e)

    def _get_recent_failures(self, skill_id: str, limit: int = 5) -> List[dict]:
        """Read recent failure entries from the usage log."""
        if not self._usage_log.exists():
            return []
        failures = []
        try:
            for line in self._usage_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("skill_id") == skill_id and entry.get("outcome") == "failure":
                    failures.append(entry)
        except Exception:
            pass
        return failures[-limit:]

    def get_usage_stats(self) -> List[dict]:
        """Return usage statistics for all skills with at least 1 use."""
        skills = self._load()
        return [
            {
                "id": s.id,
                "name": s.name,
                "status": s.status.value,
                "utility_rate": round(s.utility_rate, 3),
                "successes": s.successes,
                "failures": s.failures,
                "total_uses": s.total_uses,
                "version": s.version,
            }
            for s in skills.values()
            if s.total_uses > 0
        ]

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
        total_uses = 0
        total_successes = 0
        for s in skills.values():
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
            total_uses += s.total_uses
            total_successes += s.successes
        avg_utility = (total_successes / total_uses) if total_uses > 0 else 0
        return {
            "total": len(skills),
            "by_status": by_status,
            "total_uses": total_uses,
            "avg_utility": round(avg_utility, 3),
        }


registry = SkillRegistry()
