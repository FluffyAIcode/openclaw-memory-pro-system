"""Second Brain Bridge — integration entry point for OpenClaw.

Aggregates memories from ALL four systems (Memora vector store, Chronos
replay buffer, Memora digests, MSA document store) and feeds them into
the collision engine for cross-layer inspiration generation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config
from .tracker import tracker
from .collision import engine, Insight

logger = logging.getLogger(__name__)
config = load_config()

_WORKSPACE = Path(__file__).parent.parent


class SecondBrainBridge:
    """Orchestrates tracking and collision across all memory layers."""

    def collide(self) -> dict:
        """Run one round of collisions using memories from all systems."""
        pool = self._build_memory_pool()
        total = sum(len(v) for v in pool.values())
        if total < 2:
            return {"insights": [], "message": f"记忆条目不足 ({total})，无法碰撞",
                    "pool_stats": {k: len(v) for k, v in pool.items()}}

        insights = engine.collide_round(pool)
        if insights:
            filepath = engine.save_insights(insights)
            engine.index_high_novelty(insights)
            return {
                "insights": [i.to_dict() for i in insights],
                "file": str(filepath),
                "message": f"生成 {len(insights)} 条灵感",
                "pool_stats": {k: len(v) for k, v in pool.items()},
            }

        return {"insights": [], "message": "本轮碰撞未产生有效灵感",
                "pool_stats": {k: len(v) for k, v in pool.items()}}

    def deep_collide(self, topic: str = "") -> dict:
        """Use MSA interleave for multi-hop inspiration across document-level memories."""
        query = topic or "这些记忆之间有什么深层联系和可以行动的灵感？"
        try:
            from msa.bridge import bridge as msa_bridge
            result = msa_bridge.interleave_query(query, max_rounds=3)
            if result and result.get("final_answer"):
                insight_text = (
                    f"[深度碰撞] 基于 {result.get('total_docs_used', 0)} 篇文档的"
                    f"多跳推理：\n\n{result['final_answer']}"
                )
                try:
                    from memora.vectorstore import vector_store
                    vector_store.add(insight_text, metadata={
                        "source": "second_brain_deep_collide",
                        "importance": 0.85,
                    })
                except Exception:
                    pass

                return {
                    "answer": result["final_answer"],
                    "rounds": result.get("rounds", 0),
                    "docs_used": result.get("total_docs_used", 0),
                    "doc_ids": result.get("doc_ids_used", []),
                }
        except Exception as e:
            logger.warning("MSA interleave failed for deep_collide: %s", e)

        return {"answer": None, "message": "MSA 文档不足或 interleave 不可用"}

    def track_access(self, memory_id: str, content: str, query: str = ""):
        """Record a memory access event."""
        content_hash = str(hash(content))
        tracker.track(memory_id=memory_id, content_hash=content_hash, query=query)

    def report(self) -> dict:
        """Generate a comprehensive second-brain status report."""
        pool = self._build_memory_pool()
        all_entries = []
        for layer, entries in pool.items():
            for e in entries:
                e["_layer"] = layer
                all_entries.append(e)

        dormant = tracker.find_dormant(all_entries)
        trends = tracker.find_trends()
        stats = tracker.stats()

        vitality_scores = []
        for e in all_entries[:80]:
            v = tracker.vitality(
                importance=e.get("importance", e.get("metadata", {}).get("importance", 0.5)),
                created_at=e.get("timestamp", ""),
                content_hash=str(hash(e.get("content", ""))),
            )
            vitality_scores.append(v)

        avg_vitality = sum(vitality_scores) / len(vitality_scores) if vitality_scores else 0

        recent_insights = self._load_recent_insights()

        return {
            "pool_stats": {k: len(v) for k, v in pool.items()},
            "total_memories": sum(len(v) for v in pool.values()),
            "tracker_stats": stats,
            "dormant_count": len(dormant),
            "dormant_top3": [
                {"content": d["content"][:100], "importance": d.get("importance", 0),
                 "dormant_days": d.get("dormant_days", 0), "layer": d.get("_layer", "?")}
                for d in dormant[:3]
            ],
            "trending": trends,
            "avg_vitality": round(avg_vitality, 4),
            "vitality_sample_size": len(vitality_scores),
            "recent_insights_count": len(recent_insights),
            "recent_insights": recent_insights[:5],
        }

    def daily_briefing(self) -> dict:
        """Generate a human-readable daily briefing for the user.

        Combines overnight insights, dormant memories, trends, and
        vitality distribution into a structured report suitable for
        pushing via Telegram or displaying in CLI.
        """
        pool = self._build_memory_pool()
        all_entries = []
        for layer, entries in pool.items():
            for e in entries:
                e["_layer"] = layer
                all_entries.append(e)

        recent_insights = self._load_recent_insights(days=1)
        dormant = tracker.find_dormant(all_entries, include_never_accessed=False)
        trends = tracker.find_trends()

        vitality_buckets = {"high": [], "medium": [], "low": []}
        for e in all_entries:
            v = tracker.vitality(
                importance=e.get("importance", e.get("metadata", {}).get("importance", 0.5)),
                created_at=e.get("timestamp", ""),
                content_hash=str(hash(e.get("content", ""))),
            )
            entry_info = {"content": e.get("content", "")[:100], "vitality": v,
                          "layer": e.get("_layer", "?")}
            if v >= 0.7:
                vitality_buckets["high"].append(entry_info)
            elif v >= 0.4:
                vitality_buckets["medium"].append(entry_info)
            else:
                vitality_buckets["low"].append(entry_info)

        sections = []

        total = len(all_entries)
        high = len(vitality_buckets["high"])
        medium = len(vitality_buckets["medium"])
        low = len(vitality_buckets["low"])
        sections.append(f"🧠 你的记忆库有 {total} 条记忆，其中 {high} 条活跃")

        if trends:
            seen = set()
            tags = []
            for t in trends[:5]:
                queries = t.get("queries", [])
                label = queries[0] if queries else ""
                if label and label not in seen:
                    seen.add(label)
                    tags.append(label)
            if tags:
                sections.append(f"📈 最近你在关注: {' · '.join(tags[:3])}")

        if recent_insights:
            total_insights = sum(i.get("insight_count", 0) for i in recent_insights)
            if total_insights > 0:
                best_lines = []
                for ins in recent_insights:
                    preview = ins.get("preview", "")
                    in_connection = False
                    for line in preview.split("\n"):
                        stripped = line.strip()
                        if stripped.startswith("## 联系发现") or stripped.startswith("## 联系"):
                            in_connection = True
                            continue
                        if stripped.startswith("## "):
                            in_connection = False
                            continue
                        if in_connection and len(stripped) > 15:
                            best_lines.append(stripped[:100])
                            break
                section = f"💡 昨天发现了 {total_insights} 个新联系"
                for bl in best_lines[:2]:
                    section += f"\n  • {bl}"
                sections.append(section)

        if dormant:
            section = f"💤 {len(dormant)} 条记忆超过 {config.dormancy_age_days} 天没被用到"
            for d in dormant[:2]:
                content = d.get("content", "")[:50]
                section += f"\n  • {content}..."
            sections.append(section)

        if low > 0 and low > high:
            sections.append(f"📌 提示: {low} 条记忆活力较低，试试 recall 唤醒它们")

        today = datetime.now().strftime("%Y-%m-%d")
        text = f"☀️ {today} 记忆简报\n\n" + "\n\n".join(sections)

        return {
            "text": text,
            "date": today,
            "insight_count": sum(i.get("insight_count", 0) for i in recent_insights),
            "dormant_count": len(dormant),
            "trend_count": len(trends),
            "vitality_distribution": {k: len(v) for k, v in vitality_buckets.items()},
            "total_memories": total,
        }

    def vitality_list(self) -> dict:
        """Return per-memory vitality scores with distribution."""
        pool = self._build_memory_pool()
        all_entries = []
        for layer, entries in pool.items():
            for e in entries:
                e["_layer"] = layer
                all_entries.append(e)

        scored = []
        for e in all_entries:
            v = tracker.vitality(
                importance=e.get("importance", e.get("metadata", {}).get("importance", 0.5)),
                created_at=e.get("timestamp", ""),
                content_hash=str(hash(e.get("content", ""))),
            )
            scored.append({
                "content": e.get("content", "")[:120],
                "vitality": v,
                "layer": e.get("_layer", "?"),
                "importance": e.get("importance", 0.5),
                "timestamp": str(e.get("timestamp", ""))[:10],
            })

        scored.sort(key=lambda x: x["vitality"], reverse=True)

        high = [s for s in scored if s["vitality"] >= 0.7]
        medium = [s for s in scored if 0.4 <= s["vitality"] < 0.7]
        low = [s for s in scored if s["vitality"] < 0.4]

        return {
            "total": len(scored),
            "distribution": {"high": len(high), "medium": len(medium), "low": len(low)},
            "top_active": scored[:5],
            "nearly_dormant": [s for s in scored if s["vitality"] < 0.45][-5:],
            "all": scored,
        }

    def memory_lifecycle(self, query: str) -> dict:
        """Inspect memories matching a query — full lifecycle view."""
        results = []
        try:
            from memora.vectorstore import vector_store
            matches = vector_store.search(query, limit=5)
            for m in matches:
                content = m.get("content", "")
                content_hash = str(hash(content))
                hits = tracker._count_hits("", content_hash)
                v = tracker.vitality(
                    importance=m.get("metadata", {}).get("importance", 0.5),
                    created_at=m.get("timestamp", ""),
                    content_hash=content_hash,
                )
                last = tracker._last_access_time(content_hash)

                related_insights = self._find_related_insights(content[:80])

                results.append({
                    "content": content[:300],
                    "score": m.get("score", 0),
                    "source": m.get("metadata", {}).get("source", "?"),
                    "timestamp": m.get("timestamp", ""),
                    "importance": m.get("metadata", {}).get("importance", 0.5),
                    "vitality": v,
                    "access_count": hits,
                    "last_accessed": last.isoformat() if last else None,
                    "related_insights": related_insights[:2],
                })
        except Exception as e:
            logger.warning("memory_lifecycle search failed: %s", e)

        return {"query": query, "matches": results}

    def list_dormant(self) -> dict:
        """Return all dormant memories with full detail."""
        pool = self._build_memory_pool()
        all_entries = []
        for layer, entries in pool.items():
            for e in entries:
                e["_layer"] = layer
                all_entries.append(e)

        stale = tracker.find_dormant(all_entries, include_never_accessed=False)
        never = tracker.find_dormant(all_entries, include_never_accessed=True)
        never_only = [d for d in never if d.get("dormant_reason") == "never_accessed"]

        return {
            "count": len(stale),
            "never_accessed_count": len(never_only),
            "threshold_days": config.dormancy_age_days,
            "threshold_importance": config.dormancy_importance_threshold,
            "memories": [
                {
                    "content": d.get("content", "")[:200],
                    "importance": d.get("importance", 0),
                    "dormant_days": d.get("dormant_days", 0),
                    "layer": d.get("_layer", "?"),
                    "timestamp": str(d.get("timestamp", ""))[:10],
                    "reason": d.get("dormant_reason", "stale"),
                }
                for d in stale
            ],
            "never_accessed": [
                {
                    "content": d.get("content", "")[:200],
                    "importance": d.get("importance", 0),
                    "dormant_days": d.get("dormant_days", 0),
                    "layer": d.get("_layer", "?"),
                    "timestamp": str(d.get("timestamp", ""))[:10],
                }
                for d in never_only[:10]
            ],
        }

    def _find_related_insights(self, content_snippet: str) -> List[dict]:
        """Find insights that reference the given content."""
        results = []
        insights_dir = config.insights_path
        if not insights_dir.exists():
            return results
        for f in sorted(insights_dir.glob("*.md"), reverse=True)[:5]:
            try:
                text = f.read_text(encoding="utf-8")
                if content_snippet[:40] in text:
                    results.append({"date": f.stem, "file": f.name})
            except OSError:
                continue
        return results

    def status(self) -> dict:
        """Quick status check."""
        stats = tracker.stats()
        pool = self._build_memory_pool()
        return {
            "module": "second_brain",
            "version": "2.0.0",
            "tracker": stats,
            "pool": {k: len(v) for k, v in pool.items()},
            "config": {
                "collision_interval_hours": config.collision_interval_hours,
                "collisions_per_round": config.collisions_per_round,
                "vitality_half_life_days": config.vitality_half_life_days,
            },
        }

    # ── Memory Pool Builder ──────────────────────────────────

    def _build_memory_pool(self) -> Dict[str, List[dict]]:
        """Aggregate memories from all four systems into a layered pool.

        Returns:
            {
                "memora_vectors": [...],    # Memora snippet entries
                "chronos_encoded": [...],   # Chronos structured memories
                "digests": [...],           # Memora long-term digests
                "msa_docs": [...],          # MSA document summaries
            }
        """
        pool: Dict[str, List[dict]] = {
            "memora_vectors": [],
            "chronos_encoded": [],
            "digests": [],
            "msa_docs": [],
        }

        pool["memora_vectors"] = self._load_memora_vectors()
        pool["chronos_encoded"] = self._load_chronos_memories()
        pool["digests"] = self._load_digests()
        pool["msa_docs"] = self._load_msa_summaries()

        for layer, entries in pool.items():
            logger.info("Memory pool [%s]: %d entries", layer, len(entries))

        return pool

    def _load_memora_vectors(self) -> List[dict]:
        try:
            from memora.vectorstore import vector_store
            raw = vector_store._load()
            return [
                {
                    "content": e.get("content", ""),
                    "timestamp": e.get("timestamp", ""),
                    "importance": e.get("metadata", {}).get("importance", 0.5),
                    "metadata": e.get("metadata", {}),
                    "source_system": "memora",
                }
                for e in raw if e.get("content")
            ]
        except Exception as e:
            logger.warning("Failed to load Memora vectors: %s", e)
            return []

    def _load_chronos_memories(self) -> List[dict]:
        """Load Chronos replay buffer — structured EncodedMemory objects."""
        try:
            from chronos.replay_buffer import replay_buffer
            memories = replay_buffer._buffer
            return [
                {
                    "content": m.raw_text,
                    "timestamp": m.timestamp.isoformat(),
                    "importance": m.importance,
                    "facts": m.facts,
                    "preferences": m.preferences,
                    "emotions": m.emotions,
                    "causal_links": m.causal_links,
                    "source_system": "chronos",
                }
                for m in memories if m.raw_text
            ]
        except Exception as e:
            logger.warning("Failed to load Chronos memories: %s", e)
            return []

    def _load_digests(self) -> List[dict]:
        """Load Memora long-term digest summaries."""
        digest_dir = _WORKSPACE / "memory" / "long_term"
        if not digest_dir.exists():
            return []

        results = []
        for f in sorted(digest_dir.glob("digest_*.md"), reverse=True)[:10]:
            try:
                text = f.read_text(encoding="utf-8")
                date_part = f.stem.replace("digest_", "")[:10]
                results.append({
                    "content": text[:2000],
                    "timestamp": date_part,
                    "importance": 0.9,
                    "metadata": {"filename": f.name},
                    "source_system": "digest",
                })
            except OSError:
                continue
        return results

    def _load_msa_summaries(self) -> List[dict]:
        """Load MSA routing index as document-level summaries."""
        routing_file = _WORKSPACE / "memory" / "msa" / "routing_index.jsonl"
        if not routing_file.exists():
            return []

        results = []
        try:
            with open(routing_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    doc_id = entry.get("doc_id", "")
                    content_file = _WORKSPACE / "memory" / "msa" / "content" / f"doc_{doc_id}.jsonl"
                    preview = ""
                    if content_file.exists():
                        with open(content_file, "r", encoding="utf-8") as cf:
                            first_line = cf.readline().strip()
                            if first_line:
                                chunk = json.loads(first_line)
                                preview = chunk.get("text", "")[:500]

                    results.append({
                        "content": preview or f"[MSA document: {doc_id}]",
                        "timestamp": entry.get("ingested_at", ""),
                        "importance": 0.7,
                        "metadata": entry.get("metadata", {}),
                        "doc_id": doc_id,
                        "chunk_count": entry.get("chunk_count", 0),
                        "source_system": "msa",
                    })
        except Exception as e:
            logger.warning("Failed to load MSA summaries: %s", e)

        return results

    def _load_recent_insights(self, days: int = 7) -> List[dict]:
        """Load insight summaries from the last N days."""
        insights_dir = config.insights_path
        if not insights_dir.exists():
            return []

        results = []
        now = datetime.now()
        for f in sorted(insights_dir.glob("*.md"), reverse=True):
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
                if (now - file_date).days > days:
                    break
                text = f.read_text(encoding="utf-8")
                block_count = text.count("### 灵感碰撞")
                results.append({
                    "date": f.stem,
                    "insight_count": block_count,
                    "preview": text[:800],
                })
            except (ValueError, OSError):
                continue

        return results


bridge = SecondBrainBridge()
