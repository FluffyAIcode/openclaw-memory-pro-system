"""
Memory Hub — Unified interface for OpenClaw's memory architecture.

Storage systems:
  - Memora   (primary storage: vector RAG, snippet-level search)
  - MSA      (optional long-document storage + multi-hop reasoning)
  - Chronos  (training buffer — not on default ingest path)
  - Second Brain  (intelligence layer: KG, digest, collision — not a storage system)
  - Skill Registry (distilled actionable knowledge — participates in recall)

Recall pipeline (Context Composer architecture):
  1. Raw retrieval   — skills, KG, Memora, MSA (parallel)
  2. Reranking       — cross-encoder on evidence candidates
  3. Conflict scan   — real-time contradiction detection via KG + inference engine
  4. Composition     — 4-layer assembly via context_composer.py:
       L1 Core Facts     — skills + highest-relevance evidence
       L2 Concept Links  — KG relations + conceptual associations
       L3 Background     — long-term summaries, lower-rank evidence
       L4 Conflicts      — contradictions + warnings (always surfaced)

  Strategy router classifies query intent (factual/thinking/planning/review)
  and adjusts per-layer token budget allocation accordingly.

Ingest tags (intent):
  - thought:    user's own thinking / analysis
  - share:      forwarded / curated content
  - reference:  factual reference material
  - to_verify:  unconfirmed / needs checking
  - (none):     untagged legacy content
"""

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class RecallData(TypedDict, total=False):
    """Shared schema for the recall pipeline results dict.

    Used by MemoryHub.recall() and ContextComposer.compose().
    """
    query: str
    skills: List[dict]
    kg_relations: List[dict]
    evidence: List[dict]
    memora: List[dict]
    msa: List[dict]
    contradictions: List[dict]
    merged: List[dict]
    intent: str
    layers: dict
    budget_stats: dict

logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).parent
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

VALID_TAGS = {"thought", "share", "reference", "to_verify"}
VALID_SENSITIVITIES = {"public", "private", "confidential", "secret"}
_IMPORTANCE_CAP_UNTRUSTED = 0.9
_TRUSTED_HIGH_IMPORTANCE_SOURCES = {"openclaw", "agent", "cursor", "manual", "hub"}


class MemoryHub:
    """
    Unified memory interface.

    Ingestion routing:
      - All text → Memora (primary vector store, dedup-aware)
      - Long text (≥100 words) OR high importance (≥0.85) → also MSA
      - High importance (≥0.85) → also Chronos (structured deep encoding)
      - Always writes daily file

    Query routing (recall):
      - Skills recall (active skill registry, vector + keyword)
      - KG recall (Second Brain knowledge graph, semantic similarity)
      - Evidence recall (Memora + MSA, reranked via cross-encoder)
      → Context Composer (4-layer assembly with strategy routing)
      - deep-recall → MSA multi-hop + Memora context + skills
    """

    def __init__(self):
        self._vector_store = None
        self._chronos_bridge = None
        self._msa_bridge = None
        self._post_remember_hooks: list = []
        self._post_recall_hooks: list = []

        self._post_remember_hooks.append(self._default_kg_hook)

    @staticmethod
    def _default_kg_hook(content: str, importance: float, results: dict = None):
        """Base KG extraction hook — works without memory_server."""
        if importance < 0.4:
            return
        try:
            from second_brain.relation_extractor import extractor
            extractor.extract(content, importance=importance)
        except Exception as e:
            logger.debug("KG extraction skipped: %s", e)

    def register_post_remember_hook(self, fn):
        """Register a callback: fn(content: str, importance: float, results: dict)"""
        self._post_remember_hooks.append(fn)

    def register_post_recall_hook(self, fn):
        """Register a callback: fn(merged: list, query: str)"""
        self._post_recall_hooks.append(fn)

    @property
    def vector_store(self):
        if self._vector_store is None:
            from memora.vectorstore import vector_store
            self._vector_store = vector_store
        return self._vector_store

    @property
    def chronos(self):
        if self._chronos_bridge is None:
            from chronos.bridge import bridge
            self._chronos_bridge = bridge
        return self._chronos_bridge

    @property
    def msa(self):
        if self._msa_bridge is None:
            from msa.bridge import bridge
            self._msa_bridge = bridge
        return self._msa_bridge

    def remember(self, content: str, source: str = "openclaw",
                 importance: float = 0.7,
                 tag: Optional[str] = None,
                 sensitivity: Optional[str] = None,
                 doc_id: Optional[str] = None,
                 title: Optional[str] = None,
                 force_systems: Optional[List[str]] = None) -> Dict:
        """
        Smart ingestion with intent tagging and sensitivity classification.

        Args:
            content: The text to remember
            source: Origin channel (e.g. "telegram", "cursor", "auto_ingest")
            importance: 0.0-1.0 importance score
            tag: Intent tag — thought | share | reference | to_verify
            sensitivity: Privacy level — public | private | confidential | secret
                         Auto-classified from content if not provided.
            doc_id: Optional document ID for MSA
            title: Optional title for MSA
            force_systems: Override auto-routing, e.g. ["memora", "msa", "chronos"]
        """
        if tag and tag not in VALID_TAGS:
            logger.warning("Unknown tag '%s', ignoring", tag)
            tag = None

        if sensitivity and sensitivity not in VALID_SENSITIVITIES:
            logger.warning("Unknown sensitivity '%s', ignoring", sensitivity)
            sensitivity = None

        content = _CONTROL_CHARS.sub("", content)
        source = _CONTROL_CHARS.sub("", source)
        timestamp = datetime.now().isoformat()

        importance = max(0.0, min(importance, 1.0))
        if source not in _TRUSTED_HIGH_IMPORTANCE_SOURCES and importance > _IMPORTANCE_CAP_UNTRUSTED:
            logger.info("Importance capped %.2f → %.2f for source='%s'",
                        importance, _IMPORTANCE_CAP_UNTRUSTED, source)
            importance = _IMPORTANCE_CAP_UNTRUSTED

        if sensitivity is None:
            try:
                from memory_security import classify_sensitivity
                sensitivity = classify_sensitivity(content)
            except ImportError:
                sensitivity = "public"

        word_count = max(len(content.split()), len(content) // 2)
        results = {"word_count": word_count, "systems_used": [], "tag": tag,
                   "sensitivity": sensitivity}

        if force_systems:
            systems = set(force_systems)
        else:
            systems = self._route_ingestion(word_count, importance)

        metadata = {
            "source": source,
            "importance": importance,
            "timestamp": timestamp,
            "sensitivity": sensitivity,
        }
        if tag:
            metadata["tag"] = tag

        if "memora" in systems:
            try:
                self.vector_store.add(content, metadata=metadata)
                results["memora"] = {
                    "content": content, "source": source,
                    "importance": importance, "timestamp": timestamp,
                }
                results["systems_used"].append("memora")
            except Exception as e:
                logger.warning("Memora ingestion failed: %s", e)

        if "msa" in systems:
            try:
                msa_meta = {"source": source}
                if title:
                    msa_meta["title"] = title
                if tag:
                    msa_meta["tag"] = tag
                msa_result = self.msa.ingest_and_save(
                    content, source=source, doc_id=doc_id,
                    metadata=msa_meta,
                    cross_index=("memora" not in systems),
                    write_daily=False)
                results["msa"] = msa_result
                results["systems_used"].append("msa")
            except Exception as e:
                logger.warning("MSA ingestion failed: %s", e)

        if "chronos" in systems:
            try:
                chronos_result = self.chronos.learn_and_save(
                    content, source=source, importance=importance,
                    write_daily=False)
                results["chronos"] = {"importance": chronos_result.importance}
                results["systems_used"].append("chronos")
            except Exception as e:
                logger.warning("Chronos ingestion failed: %s", e)

        self._write_daily(content, source, results["systems_used"], tag)
        results["systems_used"].append("daily_file")

        logger.info("Memory Hub: remembered %d words via %s (tag=%s)",
                     word_count, ", ".join(results["systems_used"]), tag)

        if self._post_remember_hooks:
            def _run_hooks():
                for hook in self._post_remember_hooks:
                    try:
                        hook(content, importance, results)
                    except Exception as e:
                        logger.warning("Post-remember hook failed: %s", e)
            threading.Thread(target=_run_hooks, daemon=True,
                             name="post-remember-hooks").start()

        return results

    def recall(self, query: str, top_k: int = 8,
               max_tokens: int = 4000,
               min_score: float = 0.45) -> Dict:
        """Context-composed recall with 4-layer assembly pipeline.

        Pipeline:
          1. Raw retrieval  — skills, KG, Memora, MSA (parallel)
          2. Reranking      — cross-encoder on evidence
          3. Conflict scan  — find contradictions among recalled KG nodes
          4. Composition    — Context Composer (strategy router → layered
                             assembly → quality gate → budget controller)

        max_tokens caps the total content size in composed results.
        min_score filters out low-relevance results at retrieval stage.
        """
        results = {
            "query": query,
            "skills": [],
            "kg_relations": [],
            "evidence": [],
            "memora": [],
            "msa": [],
            "contradictions": [],
            "merged": [],
            "intent": "",
            "layers": {},
            "budget_stats": {},
        }

        # ── Raw retrieval (parallel) ──────────────────────────────
        def _fetch_memora():
            try:
                mrs = self.vector_store.search(
                    query, limit=top_k, min_score=min_score)
                for r in mrs:
                    r["system"] = "memora"
                return mrs
            except Exception as e:
                logger.warning("Memora recall failed: %s", e)
                return []

        def _fetch_msa():
            try:
                msa_result = self.msa.query_memory(query, top_k=min(top_k, 5))
                out = []
                for doc in msa_result.get("results", []):
                    out.append({
                        "content": "\n".join(doc["chunks"][:3]),
                        "score": round(doc["score"], 4),
                        "metadata": {"doc_id": doc["doc_id"], "title": doc["title"]},
                        "system": "msa",
                    })
                return out
            except Exception as e:
                logger.warning("MSA recall failed: %s", e)
                return []

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="recall") as pool:
            f_skills = pool.submit(self._recall_skills, query)
            f_kg = pool.submit(self._recall_kg, query, min_score)
            f_memora = pool.submit(_fetch_memora)
            f_msa = pool.submit(_fetch_msa)

            results["skills"] = f_skills.result()
            results["kg_relations"] = f_kg.result()
            results["memora"] = f_memora.result()
            results["msa"] = f_msa.result()

        # ── Rerank evidence ───────────────────────────────────────
        evidence = results["memora"] + results["msa"]
        evidence.sort(key=lambda x: x.get("score", 0), reverse=True)
        results["evidence"] = self._rerank(query, evidence[:top_k * 2])[:top_k]

        # ── Conflict scan (Step 3: real-time contradiction detection) ──
        results["contradictions"] = self._scan_recall_conflicts(
            results["kg_relations"])

        # ── Context Composition (Steps 1,2,4) ─────────────────────
        try:
            from context_composer import ContextComposer
            composer = ContextComposer()
            composed = composer.compose(query, results, max_tokens=max_tokens)
            results["merged"] = composed["merged"]
            results["intent"] = composed["intent"]
            results["layers"] = composed["layers"]
            results["budget_stats"] = composed["budget_stats"]
            if composed.get("gate_stats"):
                results["gate_stats"] = composed["gate_stats"]
            if composed.get("security_stats"):
                results["security_stats"] = composed["security_stats"]
            if composed.get("warnings"):
                results["composer_warnings"] = composed["warnings"]
                for w in composed["warnings"]:
                    logger.warning("Context Composer: %s", w)
        except ImportError as e:
            logger.error("Context Composer module not found: %s", e)
            results["merged"] = self._fallback_merge(results, top_k, max_tokens)
            results["composer_warnings"] = [f"MODULE_MISSING: {e}"]
        except Exception as e:
            logger.error("Context Composer failed: %s (type=%s)",
                         e, type(e).__name__, exc_info=True)
            results["merged"] = self._fallback_merge(results, top_k, max_tokens)
            results["composer_warnings"] = [f"COMPOSER_ERROR: {type(e).__name__}: {e}"]

        logger.info(
            "Recall: intent=%s, skills=%d kg=%d evidence=%d conflicts=%d "
            "merged=%d (budget %d) for '%s'",
            results.get("intent", "?"),
            len(results["skills"]), len(results["kg_relations"]),
            len(results["evidence"]), len(results["contradictions"]),
            len(results["merged"]), max_tokens, query[:60],
        )

        for hook in self._post_recall_hooks:
            try:
                hook(results.get("merged", []), query)
            except Exception as e:
                logger.warning("Post-recall hook failed: %s", e)

        return results

    def deep_recall(self, query: str, max_rounds: int = 3,
                    max_tokens: int = 4000) -> Dict:
        """MSA cross-document multi-hop reasoning with budget control and conflict detection.

        Pipeline:
          1. MSA interleave  — multi-hop retrieval-generation loop
          2. KG + skills     — supplementary retrieval for conflict scan
          3. Context Composer — budget control + conflict detection on output
        """
        results: Dict = {
            "query": query,
            "interleave": None,
            "skills": [],
            "kg_relations": [],
            "contradictions": [],
            "merged": [],
            "intent": "",
            "budget_stats": {},
        }

        # ── 1. MSA interleave (core multi-hop reasoning) ─────────
        try:
            interleave = self.msa.interleave_query(query, max_rounds=max_rounds)
            results["interleave"] = interleave
        except Exception as e:
            logger.warning("MSA interleave failed: %s", e)
            return results

        # ── 2. Supplementary retrieval for conflict detection ────
        results["skills"] = self._recall_skills(query)
        results["kg_relations"] = self._recall_kg(query)
        results["contradictions"] = self._scan_recall_conflicts(
            results["kg_relations"])

        # ── 3. Context Composer: budget control + conflict layer ──
        final_answer = interleave.get("final_answer", "") if isinstance(
            interleave, dict) else ""
        composer_input = {
            "skills": results["skills"],
            "kg_relations": results["kg_relations"],
            "evidence": [{
                "content": final_answer,
                "score": 1.0,
                "system": "msa_interleave",
                "metadata": {
                    "rounds": interleave.get("rounds", 0),
                    "docs_used": interleave.get("total_docs_used", 0),
                },
            }] if final_answer else [],
            "memora": [],
            "msa": [],
            "contradictions": results["contradictions"],
        }

        try:
            from context_composer import ContextComposer
            composer = ContextComposer()
            composed = composer.compose(query, composer_input, max_tokens=max_tokens)
            results["merged"] = composed["merged"]
            results["intent"] = composed["intent"]
            results["budget_stats"] = composed["budget_stats"]
            if composed.get("security_stats"):
                results["security_stats"] = composed["security_stats"]
            if composed.get("warnings"):
                results["composer_warnings"] = composed["warnings"]
        except Exception as e:
            logger.warning("Deep recall composer failed: %s", e)
            merged = []
            if final_answer:
                merged.append({
                    "content": final_answer, "score": 1.0,
                    "system": "msa_interleave", "layer": "core",
                })
            for c in results["contradictions"]:
                merged.append({
                    "content": c.get("decision_content", ""),
                    "score": c.get("risk_score", 0.5),
                    "system": "conflict", "layer": "conflict",
                })
            results["merged"] = merged

        logger.info(
            "Deep recall: rounds=%s docs=%s skills=%d kg=%d conflicts=%d "
            "merged=%d for '%s'",
            interleave.get("rounds", "?"), interleave.get("total_docs_used", "?"),
            len(results["skills"]), len(results["kg_relations"]),
            len(results["contradictions"]), len(results["merged"]),
            query[:60],
        )

        return results

    @staticmethod
    def _scan_recall_conflicts(kg_relations: list) -> list:
        """Find contradictions among recalled KG relations + global scan.

        Returns lightweight dicts suitable for the Context Composer's L4 layer.
        """
        conflicts = []

        for rel in kg_relations:
            if rel.get("edge_type") == "contradicts":
                conflicts.append({
                    "decision_content": rel.get("source_content", "")[:150],
                    "contradicting": [{
                        "content": rel.get("target_content", "")[:150],
                        "weight": rel.get("weight", 0.5),
                    }],
                    "risk_score": rel.get("weight", 0.5),
                    "source": "kg_edge",
                })

        try:
            from second_brain.inference import inference_engine
            reports = inference_engine.scan_contradictions()
            for r in reports[:3]:
                if r.risk_score < 0.3:
                    continue
                conflicts.append({
                    "decision_content": r.decision.content[:150],
                    "contradicting": r.contradicting[:2],
                    "risk_score": r.risk_score,
                    "source": "inference_engine",
                })
        except Exception:
            pass

        return conflicts

    def _fallback_merge(self, results: dict, top_k: int,
                        max_tokens: int) -> list:
        """Fallback merge when Context Composer is unavailable."""
        merged = []
        for s in results.get("skills", []):
            proc = s.get("procedures", "")[:150]
            body = proc if proc else s.get("content", "")[:200]
            merged.append({
                "content": f"[Skill] {s.get('name', '?')}: {body}",
                "score": 1.0, "system": "skill", "layer": "core",
            })
        for ev in results.get("evidence", [])[:top_k]:
            merged.append({**ev, "layer": "core"})
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return self._trim_to_budget(merged[:top_k], max_tokens)

    @staticmethod
    def _rerank(query: str, candidates: list) -> list:
        """Stage 2: cross-encoder reranking over bi-encoder candidates."""
        if len(candidates) <= 1:
            return candidates

        try:
            from reranker import rerank
            reranked = rerank(query, candidates, content_key="content")
            for item in reranked:
                ce_score = item.get("rerank_score")
                if ce_score is not None:
                    bi_score = item.get("score", 0)
                    ce_norm = max(min((ce_score + 10) / 20, 1.0), 0.0)
                    item["score"] = round(0.4 * bi_score + 0.6 * ce_norm, 4)
            reranked.sort(key=lambda x: x.get("score", 0), reverse=True)
            return reranked
        except Exception as e:
            logger.debug("Reranker unavailable (%s), using bi-encoder order", e)
            return candidates

    @staticmethod
    def _trim_to_budget(merged: list, max_tokens: int) -> list:
        """Trim merged results to fit within a token budget."""
        try:
            from context_composer import estimate_tokens
        except ImportError:
            def estimate_tokens(text):
                return len(text) // 3
        result = []
        budget_used = 0
        for item in merged:
            est = estimate_tokens(item.get("content", ""))
            if budget_used + est > max_tokens and result:
                break
            result.append(item)
            budget_used += est
        return result

    def _recall_skills(self, query: str) -> List[Dict]:
        """Search active skills using vector similarity with keyword fallback."""
        try:
            from skill_registry import registry
            active = registry.list_active()
            if not active:
                return []

            scored = self._recall_skills_vector(query, active)
            if not scored:
                scored = self._recall_skills_keyword(query, active)

            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {**s.to_dict(), "match_score": round(sc, 2)}
                for sc, s in scored[:5]
                if sc > 0.25
            ]
        except Exception as e:
            logger.warning("Skill recall failed: %s", e)
            return []

    def _recall_skills_vector(self, query: str,
                              skills: list) -> List[tuple]:
        """Primary path: embed query + skill content, rank by cosine similarity."""
        try:
            import shared_embedder
            emb = shared_embedder.get()
            if emb is None:
                return []

            import numpy as np
            embed_q = emb.embed_query if hasattr(emb, "embed_query") else emb.embed
            embed_d = emb.embed_document if hasattr(emb, "embed_document") else emb.embed

            q_vec = np.array(embed_q(query), dtype=np.float32)
            scored = []
            for skill in skills:
                text = f"{skill.name} {skill.content[:500]}"
                s_vec = np.array(embed_d(text), dtype=np.float32)
                sim = float(np.dot(q_vec, s_vec))
                if sim > 0.25:
                    scored.append((sim, skill))
            return scored
        except Exception as e:
            logger.debug("Vector skill recall unavailable: %s", e)
            return []

    @staticmethod
    def _recall_skills_keyword(query: str, skills: list) -> List[tuple]:
        """Fallback: keyword + tag matching when embedder is unavailable."""
        q_lower = query.lower()
        scored = []
        for skill in skills:
            name_lower = skill.name.lower()
            content_lower = skill.content.lower()

            if q_lower in name_lower or q_lower in content_lower:
                scored.append((1.0, skill))
                continue

            words = q_lower.replace("，", " ").replace("、", " ").split()
            hits = sum(1 for w in words
                       if w in name_lower or w in content_lower)
            if hits > 0:
                score = hits / max(len(words), 1)
                scored.append((score, skill))
                continue

            tag_overlap = any(t.lower() in q_lower for t in skill.tags)
            if tag_overlap:
                scored.append((0.5, skill))

        return scored

    def _recall_kg(self, query: str, max_nodes: int = 8,
                   min_score: float = 0.35) -> List[Dict]:
        """Find KG nodes related to the query and their logical relationships."""
        try:
            from second_brain.knowledge_graph import kg, KGEdgeType

            try:
                import shared_embedder
                emb = shared_embedder.get()
                if emb is not None:
                    import numpy as np
                    embed_q = emb.embed_query if hasattr(emb, "embed_query") else emb.embed
                    embed_d = emb.embed_document if hasattr(emb, "embed_document") else emb.embed
                    q_vec = np.array(embed_q(query), dtype=np.float32)
                    all_nodes = kg.get_all_nodes()
                    scored = []
                    for node in all_nodes:
                        vec = node.embedding if hasattr(node, 'embedding') and node.embedding is not None else None
                        if vec is None:
                            vec = embed_d(node.content)
                        n_vec = np.array(vec, dtype=np.float32)
                        sim = float(np.dot(q_vec, n_vec))
                        if sim > min_score:
                            scored.append((sim, node))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_nodes = [(s, n) for s, n in scored[:max_nodes]]
                else:
                    top_nodes = [(0.5, n) for n in kg.find_node_by_content(query, max_results=max_nodes)]
            except Exception:
                top_nodes = [(0.5, n) for n in kg.find_node_by_content(query, max_results=max_nodes)]

            if not top_nodes:
                return []

            node_scores = {n.id: s for s, n in top_nodes}

            relations = []
            seen_pairs = set()
            for _, node in top_nodes:
                edges = kg.get_edges(node.id, direction="both")
                for src, tgt, data in edges[:5]:
                    pair_key = (min(src, tgt), max(src, tgt))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    src_node = kg.get_node(src)
                    tgt_node = kg.get_node(tgt)
                    if not src_node or not tgt_node:
                        continue

                    edge_type = data.get("edge_type", "?")
                    desc = (f"[{src_node.node_type.value}] {src_node.content[:80]} "
                            f"-[{edge_type}]→ "
                            f"[{tgt_node.node_type.value}] {tgt_node.content[:80]}")

                    is_critical = edge_type in (
                        KGEdgeType.CONTRADICTS.value,
                        KGEdgeType.ADDRESSES.value,
                    )

                    relevance = max(
                        node_scores.get(src, 0.0),
                        node_scores.get(tgt, 0.0),
                    )

                    relations.append({
                        "description": desc,
                        "edge_type": edge_type,
                        "source_content": src_node.content[:150],
                        "target_content": tgt_node.content[:150],
                        "weight": data.get("weight", 0.5),
                        "relevance": round(relevance, 4),
                        "is_critical": is_critical,
                        "metadata": {
                            "source_id": src, "target_id": tgt,
                            "source_type": src_node.node_type.value,
                            "target_type": tgt_node.node_type.value,
                        },
                    })

            relations.sort(key=lambda r: (r["is_critical"], r["weight"]), reverse=True)
            return relations[:10]

        except Exception as e:
            logger.warning("KG recall failed: %s", e)
            return []

    def status(self) -> Dict:
        """Combined status from all systems."""
        st = {"systems": {}}

        try:
            st["systems"]["memora"] = {"entries": self.vector_store.count()}
        except Exception as e:
            st["systems"]["memora"] = {"error": str(e)}

        try:
            st["systems"]["chronos"] = self.chronos.status()
        except Exception as e:
            st["systems"]["chronos"] = {"error": str(e)}

        try:
            st["systems"]["msa"] = self.msa.status()
        except Exception as e:
            st["systems"]["msa"] = {"error": str(e)}

        return st

    def _route_ingestion(self, word_count: int, importance: float) -> set:
        """Smart routing based on text length AND importance.

        Rules:
          - Memora: always (primary vector store)
          - MSA:    long text (≥100 words) OR high importance (≥0.85)
          - Chronos: high importance (≥0.85) — structured deep encoding
        """
        systems = {"memora"}
        if word_count >= 100 or importance >= 0.85:
            systems.add("msa")
        if importance >= 0.85:
            systems.add("chronos")
        return systems

    def _write_daily(self, content: str, source: str,
                     systems: List[str], tag: Optional[str] = None):
        memory_dir = _WORKSPACE / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        today = memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        tags = "+".join(systems) if systems else "direct"
        tag_label = f" #{tag}" if tag else ""
        preview = content[:300] if len(content) > 300 else content
        with open(today, "a", encoding="utf-8") as f:
            f.write(f"\n### {datetime.now().strftime('%H:%M:%S')} "
                    f"[Hub/{tags}]{tag_label}\n{preview}\n")


hub = MemoryHub()
