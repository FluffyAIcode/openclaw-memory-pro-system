"""Skill Proposer — promotes high-quality Second Brain outputs to draft skills.

Trigger rule: when at least 2 of the 3 scores meet their thresholds,
propose a draft skill to the Skill Registry.

Score sources:
  - KG structural_gain (0–1):     stored per extraction in memory/kg/scores.jsonl
  - Digest compression_value (0–1): stored per digest as .score.json
  - Collision novelty (1–5):       stored in daily insight .md files

Thresholds (configurable via SecondBrainConfig):
  - kg_structural_gain_threshold:   0.6
  - digest_compression_threshold:   0.7
  - collision_novelty_threshold:    4   (existing config: insight_novelty_threshold)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()

_WORKSPACE = Path(__file__).parent.parent

KG_SCORE_THRESHOLD = 0.6
DIGEST_SCORE_THRESHOLD = 0.7
COLLISION_NOVELTY_THRESHOLD = 4

KG_SCORES_FILE = "memory/kg/scores.jsonl"


class SkillProposal:
    """A candidate for automatic skill creation."""

    def __init__(self, title: str, content: str, *,
                 sources: Dict[str, float],
                 source_details: List[str] = None,
                 tags: List[str] = None):
        self.title = title
        self.content = content
        self.sources = sources
        self.source_details = source_details or []
        self.tags = tags or []

    def __repr__(self):
        scores = ", ".join(f"{k}={v}" for k, v in self.sources.items())
        return f"SkillProposal({self.title!r}, {scores})"


class SkillProposer:
    """Scans Second Brain outputs and proposes draft skills when thresholds are met."""

    def __init__(self):
        self._kg_threshold = KG_SCORE_THRESHOLD
        self._digest_threshold = DIGEST_SCORE_THRESHOLD
        self._collision_threshold = COLLISION_NOVELTY_THRESHOLD

    def scan_and_propose(self, days: int = 7) -> List[SkillProposal]:
        """Main entry: collect recent scores, check 2-of-3 rule, propose skills."""
        kg_score = self._get_best_kg_score(days)
        digest_score = self._get_best_digest_score(days)
        collision_score = self._get_best_collision_score(days)

        kg_pass = kg_score >= self._kg_threshold
        digest_pass = digest_score >= self._digest_threshold
        collision_pass = collision_score >= self._collision_threshold

        passes = sum([kg_pass, digest_pass, collision_pass])

        logger.info(
            "SkillProposer: kg=%.2f(%s) digest=%.2f(%s) collision=%d(%s) → %d/3 pass",
            kg_score, "✓" if kg_pass else "✗",
            digest_score, "✓" if digest_pass else "✗",
            collision_score, "✓" if collision_pass else "✗",
            passes,
        )

        if passes < 2:
            logger.info("SkillProposer: need ≥2 passes, skipping proposal")
            return []

        proposals = self._build_proposals(
            kg_score=kg_score if kg_pass else None,
            digest_score=digest_score if digest_pass else None,
            collision_score=collision_score if collision_pass else None,
            days=days,
        )

        registered = []
        for proposal in proposals:
            skill = self._register_draft(proposal)
            if skill:
                registered.append(proposal)

        return registered

    def _get_best_kg_score(self, days: int) -> float:
        """Read the highest structural_gain from recent KG extraction scores."""
        scores_path = _WORKSPACE / KG_SCORES_FILE
        if not scores_path.exists():
            return 0.0

        cutoff = datetime.now() - timedelta(days=days)
        best = 0.0
        try:
            for line in scores_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts = entry.get("timestamp", "")
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        continue
                except (ValueError, TypeError):
                    continue
                score = entry.get("structural_gain", 0.0)
                best = max(best, score)
        except Exception as e:
            logger.warning("SkillProposer: failed to read KG scores: %s", e)
        return best

    def _get_best_digest_score(self, days: int) -> float:
        """Read the highest compression_value from recent digest score files."""
        long_term_dir = config.long_term_dir
        if not long_term_dir or not long_term_dir.exists():
            return 0.0

        cutoff = datetime.now() - timedelta(days=days)
        best = 0.0
        try:
            for sf in long_term_dir.glob("digest_*.score.json"):
                data = json.loads(sf.read_text(encoding="utf-8"))
                ts = data.get("timestamp", "")
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        continue
                except (ValueError, TypeError):
                    continue
                score = data.get("compression_value", 0.0)
                best = max(best, score)
        except Exception as e:
            logger.warning("SkillProposer: failed to read digest scores: %s", e)
        return best

    def _get_best_collision_score(self, days: int) -> int:
        """Read the highest novelty from recent insight files."""
        insights_dir = config.insights_path
        if not insights_dir or not insights_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=days)
        best = 0
        try:
            for md_file in insights_dir.glob("*.md"):
                try:
                    file_date = datetime.strptime(md_file.stem, "%Y-%m-%d")
                    if file_date < cutoff:
                        continue
                except ValueError:
                    continue

                text = md_file.read_text(encoding="utf-8")
                for line in text.splitlines():
                    if "novelty" in line.lower() or "新颖度" in line:
                        for ch in line:
                            if ch.isdigit():
                                val = int(ch)
                                if 1 <= val <= 5:
                                    best = max(best, val)
        except Exception as e:
            logger.warning("SkillProposer: failed to read collision scores: %s", e)
        return best

    def _build_proposals(self, *, kg_score: Optional[float],
                         digest_score: Optional[float],
                         collision_score: Optional[int],
                         days: int) -> List[SkillProposal]:
        """Construct skill proposals from the passing sources."""
        sources: Dict[str, float] = {}
        content_parts = []
        tags = []

        if kg_score is not None:
            sources["kg_structural_gain"] = kg_score
            tags.append("knowledge-graph")
            kg_content = self._extract_kg_mature_content()
            if kg_content:
                content_parts.append(f"## 知识图谱成熟节点\n\n{kg_content}")

        if digest_score is not None:
            sources["digest_compression_value"] = digest_score
            tags.append("digest")
            digest_content = self._extract_latest_digest_summary()
            if digest_content:
                content_parts.append(f"## 蒸馏总结\n\n{digest_content}")

        if collision_score is not None:
            sources["collision_novelty"] = float(collision_score)
            tags.append("collision")
            collision_content = self._extract_high_novelty_insights(days)
            if collision_content:
                content_parts.append(f"## 高质量灵感\n\n{collision_content}")

        if not content_parts:
            return []

        title = self._generate_skill_title(content_parts, tags)
        full_content = "\n\n".join(content_parts)

        return [SkillProposal(
            title=title,
            content=full_content,
            sources=sources,
            source_details=[f"{k}={v}" for k, v in sources.items()],
            tags=tags,
        )]

    def _extract_kg_mature_content(self) -> str:
        """Get mature KG nodes and their key relationships."""
        try:
            from .knowledge_graph import kg
            mature = kg.get_mature_patterns(min_maturity=config.kg_maturity_threshold)
            if not mature:
                return ""
            lines = []
            for node in mature[:10]:
                edges = kg.get_edges(node.id, direction="both")
                edge_desc = []
                for src, tgt, data in edges[:3]:
                    other_id = tgt if src == node.id else src
                    other = kg.get_node(other_id)
                    if other:
                        edge_desc.append(
                            f"  -[{data.get('edge_type', '?')}]→ {other.content[:60]}"
                        )
                lines.append(f"• [{node.node_type.value}] {node.content}")
                lines.extend(edge_desc)
            return "\n".join(lines)
        except Exception as e:
            logger.warning("SkillProposer: KG content extraction failed: %s", e)
            return ""

    def _extract_latest_digest_summary(self) -> str:
        """Get the AI summary from the most recent digest."""
        long_term_dir = config.long_term_dir
        if not long_term_dir or not long_term_dir.exists():
            return ""
        digests = sorted(long_term_dir.glob("digest_*.md"), reverse=True)
        if not digests:
            return ""
        try:
            text = digests[0].read_text(encoding="utf-8")
            if "## AI 摘要" in text:
                start = text.index("## AI 摘要") + len("## AI 摘要")
                end = text.find("---", start)
                if end == -1:
                    end = len(text)
                return text[start:end].strip()[:1500]
        except Exception:
            pass
        return ""

    def _extract_high_novelty_insights(self, days: int) -> str:
        """Get recent high-novelty collision insights."""
        insights_dir = config.insights_path
        if not insights_dir or not insights_dir.exists():
            return ""

        cutoff = datetime.now() - timedelta(days=days)
        results = []
        for md_file in sorted(insights_dir.glob("*.md"), reverse=True):
            try:
                file_date = datetime.strptime(md_file.stem, "%Y-%m-%d")
                if file_date < cutoff:
                    continue
            except ValueError:
                continue

            text = md_file.read_text(encoding="utf-8")
            sections = text.split("### 灵感碰撞")
            for section in sections[1:]:
                if any(f"新颖度 {n}" in section or f"novelty={n}" in section
                       for n in [4, 5]):
                    clean = section.strip()[:500]
                    results.append(f"• {clean}")
                    if len(results) >= 3:
                        break
            if len(results) >= 3:
                break

        return "\n\n".join(results)

    def _generate_skill_title(self, content_parts: List[str],
                              tags: List[str]) -> str:
        """Use LLM to generate a concise skill title, fallback to tag-based."""
        combined = "\n".join(content_parts)[:2000]
        try:
            import llm_client
            if llm_client.is_available():
                title = llm_client.generate(
                    prompt=(
                        "根据以下知识内容，生成一个简洁的技能标题（10字以内）：\n\n"
                        f"{combined[:1000]}"
                    ),
                    system="只输出技能标题，不要其他内容。",
                    max_tokens=30,
                    temperature=0.3,
                )
                if title and len(title.strip()) > 2:
                    return title.strip()[:50]
        except Exception:
            pass

        today = datetime.now().strftime("%m-%d")
        return f"{'·'.join(tags)} 技能 ({today})"

    def _register_draft(self, proposal: SkillProposal) -> Optional[object]:
        """Write the proposal into the Skill Registry as a draft."""
        try:
            from skill_registry import registry
            skill = registry.add(
                name=proposal.title,
                content=proposal.content,
                tags=proposal.tags,
                source_memories=proposal.source_details,
            )
            logger.info("SkillProposer: registered draft skill '%s' [%s]",
                        skill.name, skill.id)
            return skill
        except Exception as e:
            logger.error("SkillProposer: failed to register skill: %s", e)
            return None


proposer = SkillProposer()


def save_kg_score(structural_gain: float, content_preview: str = ""):
    """Persist a KG extraction score for the Skill Proposer to read."""
    scores_path = _WORKSPACE / KG_SCORES_FILE
    try:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "structural_gain": structural_gain,
                "content_preview": content_preview[:200],
                "timestamp": datetime.now().isoformat(),
            }, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to save KG score: %s", e)
