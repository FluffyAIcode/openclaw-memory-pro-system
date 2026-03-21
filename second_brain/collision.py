"""Collision Engine — 7 strategies across all memory layers + knowledge graph.

RAG-based strategies (find connections by similarity):
  1. semantic_bridge:     Memora → Memora (moderate cosine similarity)
  2. dormant_revival:     dormant memory → recent memory
  3. temporal_echo:       today → 7/30 days ago
  4. chronos_crossref:    Chronos structured facts/preferences → recent Memora
  5. digest_bridge:       Long-term digest theme → recent raw memory

KG-based strategies (find connections by logical reasoning — RAG cannot do this):
  6. contradiction_based: KG contradiction edges → decisions at risk
  7. blind_spot_based:    KG absence reasoning → unexplored dimensions
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import load_config
from .tracker import tracker as _tracker

logger = logging.getLogger(__name__)
config = load_config()


# ── LLM Prompt ────────────────────────────────────────────────

_COLLISION_PROMPT = """\
你是一个创意思维助手。以下是来自不同记忆层次的两条记忆片段。
请分析它们之间可能存在的深层联系，生成 1-2 个可行的灵感或行动建议。

记忆 A（来源: {source_a}, 时间: {time_a}）：
{memory_a}

记忆 B（来源: {source_b}, 时间: {time_b}）：
{memory_b}
{structured_context}
请严格按照以下格式输出：

## 联系发现
（一句话概括联系）

## 灵感
（1-2 个具体可行的想法或行动）

## 新颖度
（只输出 1-5 的数字，5 分最新颖）

## 情感共鸣
（只输出 1-5 的数字。5 分=这个联系能真正触动用户、与其目标和价值观深度相关；1 分=纯技术关联、无情感价值）
"""

_STRUCTURED_SECTION = """
补充结构化信息：
- 已提取的事实: {facts}
- 已识别的偏好: {preferences}
- 情感标签: {emotions}
- 因果链: {causal_links}
"""

_SOURCE_NAMES = {
    "memora": "Memora向量库",
    "chronos": "Chronos深度记忆",
    "digest": "长期记忆摘要",
    "msa": "MSA文档库",
}


# ── Insight ───────────────────────────────────────────────────

class Insight:
    """A single collision result."""

    __slots__ = ("memory_a", "memory_b", "strategy", "connection",
                 "ideas", "novelty", "emotional_relevance", "timestamp",
                 "raw_llm", "source_a", "source_b")

    def __init__(self, memory_a: str, memory_b: str, strategy: str,
                 connection: str = "", ideas: str = "", novelty: int = 0,
                 emotional_relevance: int = 0,
                 raw_llm: str = "", timestamp: str = "",
                 source_a: str = "", source_b: str = ""):
        self.memory_a = memory_a
        self.memory_b = memory_b
        self.strategy = strategy
        self.connection = connection
        self.ideas = ideas
        self.novelty = novelty
        self.emotional_relevance = emotional_relevance
        self.raw_llm = raw_llm
        self.timestamp = timestamp or datetime.now().isoformat()
        self.source_a = source_a
        self.source_b = source_b

    def to_dict(self) -> dict:
        d = {
            "memory_a": self.memory_a[:200],
            "memory_b": self.memory_b[:200],
            "strategy": self.strategy,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "connection": self.connection,
            "ideas": self.ideas,
            "novelty": self.novelty,
            "timestamp": self.timestamp,
        }
        if self.emotional_relevance:
            d["emotional_relevance"] = self.emotional_relevance
        return d

    def to_markdown(self) -> str:
        src_a = _SOURCE_NAMES.get(self.source_a, self.source_a)
        src_b = _SOURCE_NAMES.get(self.source_b, self.source_b)
        return (
            f"### 灵感碰撞 [{self.strategy}] — {self.timestamp[:16]}\n\n"
            f"**记忆 A** ({src_a}): {self.memory_a[:150]}…\n\n"
            f"**记忆 B** ({src_b}): {self.memory_b[:150]}…\n\n"
            f"{self.raw_llm or self.connection}\n\n"
            f"---\n"
        )


def _parse_novelty(text: str) -> int:
    """Extract novelty score specifically from the '新颖度' section."""
    sections = text.split("##")
    for section in sections:
        header_end = section.find("\n")
        if header_end == -1:
            continue
        header = section[:header_end].strip().lower()
        if "新颖" in header or "novelty" in header:
            body = section[header_end:].strip()
            for ch in body:
                if ch.isdigit() and 1 <= int(ch) <= 5:
                    return int(ch)
    for line in reversed(text.strip().splitlines()):
        line = line.strip().lstrip("#").strip()
        if "情感" in line or "emotional" in line or "共鸣" in line:
            continue
        for ch in line:
            if ch.isdigit() and 1 <= int(ch) <= 5:
                return int(ch)
    return 3


def _parse_llm_output(text: str) -> Tuple[str, str, int, int]:
    """Extract connection, ideas, novelty, emotional_relevance from LLM response."""
    connection = ""
    ideas = ""
    novelty = _parse_novelty(text)
    emotional_relevance = 0

    sections = text.split("##")
    for section in sections:
        header_end = section.find("\n")
        if header_end == -1:
            continue
        header = section[:header_end].strip().lower()
        body = section[header_end:].strip()
        if "联系" in header or "connection" in header:
            connection = body
        elif "灵感" in header or "idea" in header or "inspiration" in header:
            ideas = body
        elif "情感" in header or "emotional" in header or "共鸣" in header:
            try:
                emotional_relevance = int("".join(c for c in body if c.isdigit())[:1] or "0")
                emotional_relevance = max(1, min(5, emotional_relevance))
            except (ValueError, IndexError):
                emotional_relevance = 0

    return connection, ideas, novelty, emotional_relevance


# ── Collision Engine ──────────────────────────────────────────

_INSIGHT_PREFIXES = ("[灵感碰撞-", "[深度碰撞]")


def _is_collision_output(content: str) -> bool:
    """Return True if the content was generated by a previous collision."""
    if not content:
        return False
    for prefix in _INSIGHT_PREFIXES:
        if content.startswith(prefix):
            return True
    return False


def _filter_pool(pool: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Remove collision-generated entries from the memory pool."""
    filtered = {}
    for layer, entries in pool.items():
        filtered[layer] = [
            e for e in entries
            if not _is_collision_output(e.get("content", ""))
        ]
    return filtered


class CollisionEngine:
    """Generates creative insights by colliding memories across layers."""

    def __init__(self):
        self._insights_path = config.insights_path
        self._insights_path.mkdir(parents=True, exist_ok=True)

    def collide_round(self, pool: Dict[str, List[dict]]) -> List[Insight]:
        """Execute one round of collisions across the memory pool.

        Uses adaptive strategy weights for selection when available.

        Args:
            pool: dict with keys memora_vectors, chronos_encoded, digests, msa_docs
        """
        pool = _filter_pool(pool)

        all_flat = []
        for layer, entries in pool.items():
            for e in entries:
                e.setdefault("source_system", layer)
                all_flat.append(e)

        if len(all_flat) < 2:
            logger.warning("Too few memories (%d) for collision", len(all_flat))
            return []

        all_strategies = {
            "semantic_bridge": self._semantic_bridge,
            "chronos_crossref": self._chronos_crossref,
            "digest_bridge": self._digest_bridge,
            "dormant_revival": self._dormant_revival,
            "temporal_echo": self._temporal_echo,
            "contradiction_based": self._contradiction_based,
            "blind_spot_based": self._blind_spot_based,
        }

        try:
            from .strategy_weights import strategy_weights
            ordered = strategy_weights.select_strategies(
                n=len(all_strategies),
                available=list(all_strategies.keys()),
            )
        except Exception:
            ordered = list(all_strategies.keys())

        insights = []
        for name in ordered:
            if len(insights) >= config.collisions_per_round:
                break
            strategy_fn = all_strategies.get(name)
            if strategy_fn is None:
                continue
            try:
                pair = strategy_fn(pool, all_flat)
                if pair is None:
                    logger.info("Strategy %s: no suitable pair found", name)
                    continue
                insight = self._generate_insight(pair[0], pair[1], name)
                insights.append(insight)
            except Exception as e:
                logger.error("Strategy %s failed: %s", name, e)

        return insights

    # ── Strategy 1: Semantic Bridge (Memora ↔ Memora) ─────────

    def _semantic_bridge(self, pool: Dict[str, List[dict]],
                         all_flat: List[dict]) -> Optional[Tuple[dict, dict]]:
        """Find two Memora entries with moderate similarity (0.35-0.65)."""
        memora = pool.get("memora_vectors", [])
        if len(memora) < 2:
            return None

        anchor = random.choice(memora)
        anchor_content = anchor.get("content", "")

        try:
            from memora.vectorstore import vector_store
            results = vector_store.search(anchor_content, limit=config.max_search_candidates)
        except Exception:
            return None

        candidates = []
        for r in results:
            score = r.get("score", 0)
            if config.semantic_bridge_low <= score <= config.semantic_bridge_high:
                if r.get("content", "") != anchor_content:
                    candidates.append(r)

        if not candidates:
            return None

        partner = random.choice(candidates)
        partner["source_system"] = "memora"
        return anchor, partner

    # ── Strategy 2: Chronos Crossref (Chronos → Memora) ───────

    def _chronos_crossref(self, pool: Dict[str, List[dict]],
                           all_flat: List[dict]) -> Optional[Tuple[dict, dict]]:
        """Cross-reference a Chronos structured memory with a recent Memora entry.

        Picks a Chronos memory that has extracted facts/preferences, then
        searches Memora for a semantically related but temporally distant entry.
        """
        chronos = pool.get("chronos_encoded", [])
        memora = pool.get("memora_vectors", [])

        if not chronos or not memora:
            return None

        rich = [c for c in chronos if c.get("facts") or c.get("preferences")]
        if not rich:
            rich = chronos

        pick = random.choice(rich)

        search_query = pick.get("content", "")[:200]
        if pick.get("facts"):
            search_query = " ".join(pick["facts"][:3])
        elif pick.get("preferences"):
            search_query = " ".join(pick["preferences"][:3])

        try:
            from memora.vectorstore import vector_store
            results = vector_store.search(search_query, limit=10)
        except Exception:
            return None

        for r in results:
            if r.get("content", "") != pick.get("content", ""):
                r["source_system"] = "memora"
                return pick, r

        return None

    # ── Strategy 3: Digest Bridge (Digest → Recent) ───────────

    def _digest_bridge(self, pool: Dict[str, List[dict]],
                        all_flat: List[dict]) -> Optional[Tuple[dict, dict]]:
        """Connect a long-term digest insight with a recent raw memory.

        Digests contain summarized patterns/decisions. Pairing them with
        recent entries can reveal how past conclusions connect to new events.
        """
        digests = pool.get("digests", [])
        memora = pool.get("memora_vectors", [])

        if not digests or not memora:
            return None

        digest_pick = random.choice(digests)

        now = datetime.now()
        recent_cutoff = now - timedelta(days=7)
        recent = []
        for e in memora:
            try:
                ts = datetime.fromisoformat(e.get("timestamp", ""))
                if ts >= recent_cutoff:
                    recent.append(e)
            except (ValueError, TypeError):
                pass

        if not recent:
            recent = memora[:5]

        recent_pick = random.choice(recent)
        return digest_pick, recent_pick

    # ── Strategy 4: Dormant Revival (Any Layer) ───────────────

    def _dormant_revival(self, pool: Dict[str, List[dict]],
                          all_flat: List[dict]) -> Optional[Tuple[dict, dict]]:
        """Pair a dormant memory with a recent active one."""
        dormant = _tracker.find_dormant(all_flat)
        if not dormant:
            return None

        now = datetime.now()
        recent_cutoff = now - timedelta(days=3)
        recent = [
            e for e in all_flat
            if _safe_parse_ts(e.get("timestamp", "")) and
               _safe_parse_ts(e.get("timestamp", "")) >= recent_cutoff
        ]

        if not recent:
            recent = all_flat[:5]

        dormant_pick = dormant[0]
        recent_pick = random.choice(recent)

        if dormant_pick.get("content") == recent_pick.get("content"):
            return None

        return dormant_pick, recent_pick

    # ── Strategy 5: Temporal Echo (Any Layer) ─────────────────

    def _temporal_echo(self, pool: Dict[str, List[dict]],
                        all_flat: List[dict]) -> Optional[Tuple[dict, dict]]:
        """Connect recent memory with one from 7 or 30 days ago."""
        now = datetime.now()
        recent = []
        echoes = []

        for e in all_flat:
            ts = _safe_parse_ts(e.get("timestamp", ""))
            if ts is None:
                continue
            age = (now - ts).days
            if age <= 1:
                recent.append(e)
            elif 6 <= age <= 8 or 29 <= age <= 31:
                echoes.append(e)

        if not recent or not echoes:
            return None

        return random.choice(recent), random.choice(echoes)

    # ── Strategy 6: Contradiction-Based (KG-driven) ────────────

    def _contradiction_based(self, pool: Dict[str, List[dict]],
                              all_flat: List[dict]) -> Optional[Tuple[dict, dict]]:
        """Use KG contradiction edges to find decisions at risk.

        This is fundamentally different from RAG: it uses typed logical
        relationships, not cosine similarity.
        """
        try:
            from .inference import inference_engine
            reports = inference_engine.scan_contradictions()
            if not reports:
                return None

            report = reports[0]
            if not report.contradicting:
                return None

            mem_a = {
                "content": report.decision.content,
                "timestamp": report.decision.created_at,
                "source_system": "kg_decision",
                "importance": report.decision.importance,
            }
            contra = report.contradicting[0]
            mem_b = {
                "content": contra["content"],
                "timestamp": "",
                "source_system": "kg_contradiction",
                "importance": contra.get("weight", 0.7),
            }
            return mem_a, mem_b
        except Exception as e:
            logger.warning("contradiction_based strategy failed: %s", e)
            return None

    # ── Strategy 7: Blind Spot-Based (KG-driven) ─────────────

    def _blind_spot_based(self, pool: Dict[str, List[dict]],
                           all_flat: List[dict]) -> Optional[Tuple[dict, dict]]:
        """Use KG absence reasoning to surface unexplored dimensions.

        This is something RAG fundamentally cannot do: detect what is MISSING.
        """
        try:
            from .inference import inference_engine
            reports = inference_engine.detect_all_blind_spots()
            if not reports:
                return None

            report = reports[0]
            if not report.missing_dimensions:
                return None

            mem_a = {
                "content": report.decision.content,
                "timestamp": report.decision.created_at,
                "source_system": "kg_decision",
                "importance": report.decision.importance,
            }
            missing = "、".join(report.missing_dimensions[:4])
            mem_b = {
                "content": f"[盲区] 关于这个决策，以下维度尚未被考虑: {missing}",
                "timestamp": datetime.now().isoformat(),
                "source_system": "kg_blindspot",
                "importance": 0.9,
            }
            return mem_a, mem_b
        except Exception as e:
            logger.warning("blind_spot_based strategy failed: %s", e)
            return None

    # ── Insight Generation ────────────────────────────────────

    def _generate_insight(self, mem_a: dict, mem_b: dict, strategy: str) -> Insight:
        """Use LLM to analyze the collision with full structured context."""
        content_a = mem_a.get("content", "")[:500]
        content_b = mem_b.get("content", "")[:500]
        time_a = str(mem_a.get("timestamp", "未知时间"))[:10]
        time_b = str(mem_b.get("timestamp", "未知时间"))[:10]
        source_a = mem_a.get("source_system", "unknown")
        source_b = mem_b.get("source_system", "unknown")

        structured_context = ""
        for mem in [mem_a, mem_b]:
            facts = mem.get("facts", [])
            prefs = mem.get("preferences", [])
            emotions = mem.get("emotions", [])
            causal = mem.get("causal_links", [])
            if any([facts, prefs, emotions, causal]):
                structured_context = _STRUCTURED_SECTION.format(
                    facts=", ".join(facts[:5]) or "无",
                    preferences=", ".join(prefs[:3]) or "无",
                    emotions=", ".join(emotions[:3]) or "无",
                    causal_links=", ".join(causal[:3]) or "无",
                )
                break

        llm_result = None
        try:
            import llm_client
            if llm_client.is_available():
                prompt = _COLLISION_PROMPT.format(
                    memory_a=content_a, memory_b=content_b,
                    time_a=time_a, time_b=time_b,
                    source_a=_SOURCE_NAMES.get(source_a, source_a),
                    source_b=_SOURCE_NAMES.get(source_b, source_b),
                    structured_context=structured_context,
                )
                llm_result = llm_client.generate(
                    prompt=prompt,
                    system="你是创意思维助手。只输出规定格式的分析，简洁有力。",
                    max_tokens=800,
                    temperature=0.9,
                )
        except ImportError:
            pass

        if llm_result:
            connection, ideas, novelty, emo_rel = _parse_llm_output(llm_result)
            return Insight(
                memory_a=content_a, memory_b=content_b,
                strategy=strategy, connection=connection,
                ideas=ideas, novelty=novelty,
                emotional_relevance=emo_rel, raw_llm=llm_result,
                source_a=source_a, source_b=source_b,
            )

        return Insight(
            memory_a=content_a, memory_b=content_b,
            strategy=strategy,
            connection=f"记忆 A ({_SOURCE_NAMES.get(source_a, source_a)}) 与"
                       f"记忆 B ({_SOURCE_NAMES.get(source_b, source_b)}) 可能存在关联",
            ideas="需要进一步思考它们之间的联系",
            novelty=2,
            source_a=source_a, source_b=source_b,
        )

    # ── Persistence ───────────────────────────────────────────

    def save_insights(self, insights: List[Insight]) -> Path:
        """Persist insights to daily markdown file."""
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = self._insights_path / f"{today}.md"

        header = ""
        if not filepath.exists():
            header = f"# 灵感碰撞 — {today}\n\n"

        with open(filepath, "a", encoding="utf-8") as f:
            if header:
                f.write(header)
            for ins in insights:
                f.write(ins.to_markdown() + "\n")

        logger.info("Saved %d insights to %s", len(insights), filepath)
        return filepath

    def index_high_novelty(self, insights: List[Insight]):
        """Log high-novelty insights. No longer writes to Memora vector store
        to prevent self-referencing pollution in future collision rounds.
        Insights are already persisted in the daily insights .md file."""
        threshold = config.insight_novelty_threshold
        for ins in insights:
            if ins.novelty >= threshold:
                logger.info("High-novelty insight (novelty=%d, %s→%s, strategy=%s): %s",
                            ins.novelty, ins.source_a, ins.source_b,
                            ins.strategy, ins.connection[:100])


def _safe_parse_ts(ts_str: str) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


engine = CollisionEngine()
