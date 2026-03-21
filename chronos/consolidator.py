"""
Periodic memory consolidation — strengthens important memories
by replaying them through the EWC engine and updating LoRA adapters.

Also generates PERSONALITY.yaml from high-importance memories using
xAI Grok API, which gets injected into the agent's system prompt
at session startup.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .ewc import ewc_engine
from .replay_buffer import replay_buffer
from .dynamic_lora import dynamic_lora
from .config import load_config

logger = logging.getLogger(__name__)

_PERSONALITY_PROMPT = """\
你是一个人格档案生成器。基于以下 {count} 条高重要性记忆，生成一份 YAML 格式的人格档案。

记忆列表：
{memories}

请生成以下结构的 YAML（直接输出 YAML，不要加 ```yaml 标记）：

core_beliefs:
  - "..."

communication_style:
  language: "..."
  tone: "..."

knowledge_anchors:
  - topic: "..."
    insight: "..."

learned_preferences:
  - "..."

important_decisions:
  - "..."
"""


class MemoryConsolidator:

    def __init__(self):
        cfg = load_config()
        self._interval = timedelta(hours=cfg.consolidation_interval_hours)
        self._state_file: Optional[Path] = (
            cfg.state_path / "consolidator.json" if cfg.state_path else None
        )
        self._last: Optional[datetime] = self._load_last()
        self._count: int = 0

    def should_consolidate(self) -> bool:
        if self._last is None:
            return True
        return datetime.now() - self._last > self._interval

    def consolidate(self, force: bool = False) -> dict:
        if not force and not self.should_consolidate():
            logger.info("巩固间隔未到（上次: %s），跳过", self._last)
            return {"skipped": True}

        important = replay_buffer.get_important_memories(threshold=0.8)
        if not important:
            logger.info("无高重要性记忆需要巩固")
            return {"consolidated": 0}

        logger.info("开始记忆巩固 — %d 条", len(important))
        result = ewc_engine.consolidate(important)
        dynamic_lora.update(important)

        kg_patterns = None
        try:
            from second_brain.internalization import internalization_manager
            kg_patterns = internalization_manager.get_patterns_for_personality()
            if kg_patterns:
                logger.info("从知识图谱获取 %d 个稳定模式用于人格档案", len(kg_patterns))
        except Exception as e:
            logger.debug("KG patterns unavailable: %s", e)

        profile_generated = self._generate_personality_profile(important, kg_patterns)

        self._last = datetime.now()
        self._count += 1
        self._save_last()
        logger.info("巩固完成 (第 %d 次, 人格档案: %s)", self._count,
                     "已生成" if profile_generated else "跳过")
        return {
            "consolidated": len(important),
            "ewc_result": result,
            "consolidation_number": self._count,
            "personality_profile_updated": profile_generated,
        }

    def _generate_personality_profile(self, important_memories,
                                       kg_patterns=None) -> bool:
        """Use LLM to generate PERSONALITY.yaml from high-importance memories
        and stable KG patterns (if available).

        Args:
            important_memories: Chronos replay buffer entries
            kg_patterns: Optional list of dicts from InternalizationManager
        """
        try:
            import llm_client
            if not llm_client.is_available():
                logger.info("LLM unavailable, skipping personality profile generation")
                return False
        except ImportError:
            return False

        memories_text = "\n".join(
            f"[重要性 {m.importance:.2f}] {m.raw_text[:300]}"
            for m in sorted(important_memories, key=lambda x: x.importance, reverse=True)[:30]
        )

        kg_section = ""
        if kg_patterns:
            kg_lines = []
            for p in kg_patterns[:20]:
                kg_lines.append(
                    f"[{p['type']}] (置信度 {p.get('confidence', 0):.2f}, "
                    f"成熟度 {p.get('maturity', 0):.2f}) {p['content'][:200]}"
                )
            kg_section = (
                "\n\n已验证的稳定模式（来自知识图谱，高成熟度）：\n"
                + "\n".join(kg_lines)
            )

        prompt = _PERSONALITY_PROMPT.format(
            count=len(important_memories),
            memories=memories_text + kg_section,
        )
        result = llm_client.generate(
            prompt=prompt,
            system="只输出 YAML 格式内容，不要添加任何前缀、后缀或代码块标记。",
            max_tokens=1500,
            temperature=0.5,
        )

        if not result:
            logger.warning("LLM returned empty result for personality profile")
            return False

        yaml_content = result.strip()
        if yaml_content.startswith("```"):
            lines = yaml_content.split("\n")
            yaml_content = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        workspace = Path(__file__).parent.parent
        profile_path = workspace / "PERSONALITY.yaml"

        header = (
            f"# Auto-generated by Chronos Memory Consolidation\n"
            f"# Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Based on {len(important_memories)} high-importance memories\n"
            f"# DO NOT EDIT MANUALLY — regenerated on each consolidation\n\n"
        )

        profile_path.write_text(header + yaml_content + "\n", encoding="utf-8")
        logger.info("Personality profile written to %s", profile_path)
        return True

    def report(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "last_consolidation": self._last.isoformat() if self._last else None,
            "consolidation_count": self._count,
            "buffer": replay_buffer.stats(),
            "ewc": ewc_engine.stats,
            "lora": dynamic_lora.stats,
        }

    # ---- persistence ----

    def _load_last(self) -> Optional[datetime]:
        if not self._state_file or not self._state_file.exists():
            return None
        try:
            with open(self._state_file, "r") as f:
                ts = json.load(f).get("last_consolidation")
            return datetime.fromisoformat(ts) if ts else None
        except Exception:
            return None

    def _save_last(self):
        if not self._state_file:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump({
                "last_consolidation": self._last.isoformat() if self._last else None,
                "consolidation_count": self._count,
            }, f, indent=2)


consolidator = MemoryConsolidator()
