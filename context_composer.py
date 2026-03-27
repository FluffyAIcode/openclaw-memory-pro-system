"""
Context Composer — 6-stage context assembly pipeline for OpenClaw recall.

Architecture:
  Stage 1 — Strategy Router:    score-based intent classification → layer weights
  Stage 2 — Relevance Gate:     cross-encoder / n-gram query-relevance verification
  Stage 2.5 — Security Gate:    multi-signal trust scoring + PII detection (memory_security)
  Stage 3 — Layered Assembly:   4 layers with MMR dedup
  Stage 4 — Quality Gate:       multi-dim scoring (relevance × recency × importance)
  Stage 5 — Budget Controller:  CJK-aware token estimation + dynamic rebalancing

Layers:
  L1 Core Facts     — skills + highest-relevance evidence (max precision)
  L2 Concept Links  — KG relations + conceptual associations
  L3 Background     — long-term summaries, lower-rank evidence
  L4 Conflicts      — contradictions + warnings (query-relevance-filtered)

Usage:
    composer = ContextComposer()
    result = composer.compose(query, raw_recall_data, max_tokens=4000)
"""

import logging
import math
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Stage 2: Relevance Gate ───────────────────────────────────────


def _get_cross_encoder():
    """Get the shared cross-encoder singleton from reranker module."""
    try:
        from reranker import get_cross_encoder
        return get_cross_encoder()
    except ImportError:
        return None


def _has_cjk(text: str, threshold: float = 0.3) -> bool:
    """Check if text is predominantly CJK (Chinese/Japanese/Korean)."""
    if not text:
        return False
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
              or '\u3400' <= c <= '\u4dbf'
              or '\u3040' <= c <= '\u30ff'
              or '\uac00' <= c <= '\ud7af')
    return cjk / max(len(text.replace(" ", "")), 1) > threshold


def _query_relevance(query: str, content: str) -> float:
    """Score how relevant content is to the query.

    Routing strategy:
      - CJK queries → keyword + character n-gram method (cross-encoder
        ms-marco-MiniLM is English-only, gives false-high scores to all CJK)
      - Latin queries → cross-encoder when available, n-gram fallback
    """
    if not query or not content:
        return 0.0

    if _has_cjk(query):
        return _cjk_query_relevance(query, content)

    ce = _get_cross_encoder()
    if ce is not None:
        try:
            scores = ce.predict([(query, content[:256])])
            raw = float(scores[0]) if hasattr(scores, '__len__') else float(scores)
            return max(min((raw + 10) / 20, 1.0), 0.0)
        except Exception:
            pass

    return _ngram_query_relevance(query, content)


def _cjk_query_relevance(query: str, content: str) -> float:
    """CJK-optimized relevance using bigram coverage only.

    Bigrams are the right unit for Chinese: single characters are too
    ambiguous ("学" in "学习" vs "力学"), but bigrams like "学习" vs "力学"
    are clearly distinct. Returns the fraction of query bigrams found
    in the content.
    """
    q = query.lower()[:80]
    c = content.lower()[:500]

    _PUNCT = set('，。？！、；：""''（）·…— \t\n[]【】')
    n = 2
    q_bigrams = {q[i:i+n] for i in range(max(len(q) - n + 1, 1))
                 if not (q[i] in _PUNCT or q[i+1] in _PUNCT)}
    c_bigrams = {c[i:i+n] for i in range(max(len(c) - n + 1, 1))
                 if not (c[i] in _PUNCT or c[i+1] in _PUNCT)}
    if not q_bigrams:
        return 0.0
    return len(q_bigrams & c_bigrams) / len(q_bigrams)


def _ngram_query_relevance(query: str, content: str) -> float:
    """Latin-text n-gram relevance fallback."""
    n = 2
    q_lower = query.lower()[:100]
    c_lower = content.lower()[:300]
    q_grams = {q_lower[i:i+n] for i in range(max(len(q_lower) - n + 1, 1))}
    c_grams = {c_lower[i:i+n] for i in range(max(len(c_lower) - n + 1, 1))}
    if not q_grams:
        return 0.0
    overlap = len(q_grams & c_grams)
    return overlap / len(q_grams)


RELEVANCE_GATE_THRESHOLD = 0.12

_CJK_FUNCTION_CHARS = frozenset(
    '的了我你他她它们吗呢吧啊呀哦嗯么什怎为也是在不有这那到都和就'
    '人着过被把让给与上下中大小里好些又还可以会要将从对能所该自但'
    '当然已才更比很非常也许应每请帮一二三四五六七八九十个'
)


def _cjk_query_specificity(query: str) -> float:
    """How topic-specific is the CJK query? Returns 0.0-1.0.

    "记忆系统设计" → high (all content words)
    "帮我总结一下" → low (mostly function words)
    """
    cjk_chars = [c for c in query if '\u4e00' <= c <= '\u9fff']
    if not cjk_chars:
        return 0.5
    content_chars = [c for c in cjk_chars if c not in _CJK_FUNCTION_CHARS]
    return len(content_chars) / len(cjk_chars)


def effective_gate_threshold(query: str) -> float:
    """Adaptive threshold: strict for topic-specific queries, disabled otherwise.

    CJK bigram overlap only produces reliable signals when the query has
    enough distinctive content words (specificity >= 0.7). For medium and
    generic queries, bigram matching is too brittle — we rely on the upstream
    bi-encoder + quality gate + budget control instead.

    "记忆系统设计"          → spec=1.0  → threshold=0.12 (strict gate)
    "今天的工作计划"         → spec=0.86 → threshold=0.10 (strict gate)
    "最近学了什么"          → spec=0.5  → threshold=0.0  (gate disabled)
    "帮我总结一下"          → spec=0.33 → threshold=0.0  (gate disabled)
    """
    if not _has_cjk(query):
        return RELEVANCE_GATE_THRESHOLD
    spec = _cjk_query_specificity(query)
    if spec < 0.7:
        return 0.0
    return RELEVANCE_GATE_THRESHOLD * spec


# ── Stage 1: Strategy Router (score-based) ────────────────────────

class QueryIntent:
    FACTUAL = "factual"
    THINKING = "thinking"
    PLANNING = "planning"
    REVIEW = "review"


_INTENT_PATTERNS: Dict[str, re.Pattern] = {
    QueryIntent.REVIEW: re.compile(
        r"(昨天|今天|上周|这周|本周|最近|过去|之前|历史|回顾|review|last\s+week|yesterday|today|"
        r"this\s+week|recently|\d+天前|\d+\s*days?\s*ago)", re.IGNORECASE
    ),
    QueryIntent.THINKING: re.compile(
        r"(为什么|为何|原因|why|because|how\s+come|怎么会|什么导致)", re.IGNORECASE
    ),
    QueryIntent.PLANNING: re.compile(
        r"(怎么[做办优改提搞]|如何|计划|方案|步骤|接下来|下一步|should|how\s+to|plan|next\s+step|strategy|"
        r"建议|推荐|recommend|优化)", re.IGNORECASE
    ),
    QueryIntent.FACTUAL: re.compile(
        r"(是什么|什么是|谁|哪个|哪些|多少|what\s+is|who|which|how\s+many|define|"
        r"告诉我|explain)", re.IGNORECASE
    ),
}

_INTENT_BASE_WEIGHT = {
    QueryIntent.REVIEW: 0.5,
    QueryIntent.THINKING: 1.2,
    QueryIntent.PLANNING: 1.1,
    QueryIntent.FACTUAL: 1.0,
}

LAYER_WEIGHTS = {
    QueryIntent.FACTUAL:  {"core": 0.50, "concept": 0.15, "background": 0.25, "conflict": 0.10},
    QueryIntent.THINKING: {"core": 0.25, "concept": 0.35, "background": 0.25, "conflict": 0.15},
    QueryIntent.PLANNING: {"core": 0.30, "concept": 0.25, "background": 0.20, "conflict": 0.25},
    QueryIntent.REVIEW:   {"core": 0.20, "concept": 0.15, "background": 0.55, "conflict": 0.10},
}


def classify_intent(query: str) -> str:
    """Score-based intent classification — all patterns scored, highest wins.

    Each matching pattern contributes its base weight × match count.
    Resolves ambiguity like "为什么上周进展慢" (THINKING beats REVIEW
    when both match, because "为什么" is the primary question word).
    """
    scores: Dict[str, float] = {}
    for intent, pattern in _INTENT_PATTERNS.items():
        matches = pattern.findall(query)
        if matches:
            base = _INTENT_BASE_WEIGHT[intent]
            scores[intent] = base * len(matches)

    if not scores:
        return QueryIntent.FACTUAL

    _boost_primary_intent(query, scores)

    return max(scores, key=scores.get)


def _boost_primary_intent(query: str, scores: Dict[str, float]):
    """Boost intent whose trigger word appears closest to the question start.

    "为什么上周失败" → THINKING ("为什么" at position 0 beats "上周" at position 3).
    """
    earliest_pos = {}
    for intent, pattern in _INTENT_PATTERNS.items():
        if intent not in scores:
            continue
        m = pattern.search(query)
        if m:
            earliest_pos[intent] = m.start()

    if not earliest_pos:
        return

    min_pos = min(earliest_pos.values())
    for intent, pos in earliest_pos.items():
        if pos == min_pos:
            scores[intent] = scores.get(intent, 0) + 0.5


# ── Token Estimation (canonical, used across modules) ─────────────

def estimate_tokens(text: str) -> int:
    """CJK-aware token estimation — canonical implementation.

    Chinese: ~1.5 tokens per character
    English: ~1.3 tokens per word (≈ 4 chars)
    Mixed: weighted average

    This is the single source of truth; other modules should import this.
    """
    if not text:
        return 0
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
    latin_count = len(text) - cjk_count
    return int(cjk_count * 1.5 + latin_count / 3.0)


# ── Stage 3: Quality Gate (multi-dim scoring) ─────────────────────

def _recency_score(timestamp_str: str, half_life_days: float = 14.0) -> float:
    """Exponential decay: score = exp(-ln2 * age_days / half_life).

    Returns 1.0 for now, 0.5 at half_life, ~0 for very old.
    Uses UTC consistently to avoid timezone drift.
    """
    if not timestamp_str:
        return 0.5
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_seconds = (now - ts).total_seconds()
        age_days = max(age_seconds / 86400.0, 0.0)
        return math.exp(-math.log(2) * age_days / half_life_days)
    except (ValueError, TypeError):
        return 0.5


def score_item(item: dict, query_intent: str) -> float:
    """Multi-dimensional quality score: relevance × recency × importance.

    Intent modulates component weights:
      - REVIEW:   heavy recency
      - FACTUAL:  heavy relevance
      - THINKING: balanced
      - PLANNING: relevance + light recency
    """
    relevance = item.get("score", 0.0)
    timestamp = item.get("timestamp", "")
    recency = _recency_score(timestamp)

    importance = item.get("metadata", {}).get("importance", 0.5)
    if isinstance(importance, str):
        try:
            importance = float(importance)
        except (ValueError, TypeError):
            importance = 0.5

    if query_intent == QueryIntent.REVIEW:
        w_rel, w_rec, w_imp = 0.3, 0.5, 0.2
    elif query_intent == QueryIntent.FACTUAL:
        w_rel, w_rec, w_imp = 0.6, 0.2, 0.2
    elif query_intent == QueryIntent.THINKING:
        w_rel, w_rec, w_imp = 0.45, 0.25, 0.3
    else:  # PLANNING
        w_rel, w_rec, w_imp = 0.5, 0.2, 0.3

    composite = w_rel * relevance + w_rec * recency + w_imp * importance
    return round(composite, 4)


# ── MMR Deduplication ─────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    """Fast character n-gram Jaccard similarity (no embedder needed).

    Uses 3-gram overlap as a proxy for semantic similarity.
    """
    if not a or not b:
        return 0.0
    n = 3
    grams_a = {a[i:i+n] for i in range(max(len(a) - n + 1, 1))}
    grams_b = {b[i:i+n] for i in range(max(len(b) - n + 1, 1))}
    if not grams_a or not grams_b:
        return 0.0
    intersection = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return intersection / union if union > 0 else 0.0


def _mmr_select(candidates: List[dict], budget_fn, layer: str,
                diversity_lambda: float = 0.6,
                min_score: float = 0.0) -> List[dict]:
    """Maximal Marginal Relevance selection with budget awareness.

    MMR = λ × composite_score - (1-λ) × max_sim(candidate, selected)

    Selects items greedily: each new item maximizes relevance while
    minimizing redundancy with already-selected items.
    Skips items below min_score. Uses continue (not break) on budget
    overflow so smaller items later can still fit.
    """
    if not candidates:
        return []

    selected: List[dict] = []
    selected_texts: List[str] = []
    remaining = list(candidates)

    while remaining:
        best_mmr = -float("inf")
        best_idx = -1

        for i, cand in enumerate(remaining):
            cs = cand.get("composite_score", 0.0)
            if cs < min_score:
                continue

            if selected_texts:
                cand_text = cand.get("content", "")[:200]
                max_sim = max(
                    _text_similarity(cand_text, st)
                    for st in selected_texts
                )
            else:
                max_sim = 0.0

            mmr = diversity_lambda * cs - (1 - diversity_lambda) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx < 0:
            break

        item = remaining.pop(best_idx)
        content = item.get("content", "")

        if budget_fn(layer, content):
            selected.append(item)
            selected_texts.append(content[:200])
        # continue (not break): a smaller item later may still fit

    return selected


# ── Stage 4: Budget Controller (with redistribution) ─────────────

class BudgetController:
    """Allocates token budget across layers with hard cap + redistribution."""

    def __init__(self, total_budget: int, weights: dict):
        self.total = total_budget
        raw = {layer: int(total_budget * w) for layer, w in weights.items()}
        total_raw = sum(raw.values())
        if total_raw > total_budget:
            scale = total_budget / total_raw
            raw = {k: max(int(v * scale), 30) for k, v in raw.items()}
        else:
            raw = {k: max(v, 30) for k, v in raw.items()}
        self.allocations = dict(raw)
        self._used = {layer: 0 for layer in self.allocations}
        self._force_used = 0

    @property
    def global_remaining(self) -> int:
        return max(self.total - sum(self._used.values()) - self._force_used, 0)

    def remaining(self, layer: str) -> int:
        return max(self.allocations.get(layer, 0) - self._used.get(layer, 0), 0)

    def try_add(self, layer: str, text: str) -> bool:
        """Try to add text to a layer. Respects both layer and global budgets."""
        tokens = estimate_tokens(text)
        layer_ok = self._used[layer] + tokens <= self.allocations[layer]
        global_ok = (sum(self._used.values()) + self._force_used + tokens
                     <= self.total)
        if layer_ok and global_ok:
            self._used[layer] += tokens
            return True
        return False

    def force_add(self, layer: str, text: str, hard_cap: int = 0) -> bool:
        """Add critical items (skills, conflicts) with a hard cap.

        Returns False and skips the item if the hard cap is exceeded.
        """
        tokens = estimate_tokens(text)
        if hard_cap > 0 and self._force_used + tokens > hard_cap:
            return False
        self._force_used += tokens
        self._used[layer] += tokens
        return True

    def redistribute(self):
        """Move unused budget from completed layers to remaining layers."""
        layers = list(self.allocations.keys())
        surplus = 0
        hungry = []
        for layer in layers:
            unused = self.remaining(layer)
            used = self._used[layer]
            alloc = self.allocations[layer]
            if used > 0 and unused > alloc * 0.3:
                give_back = int(unused * 0.7)
                self.allocations[layer] -= give_back
                surplus += give_back
            elif used == 0 and unused == alloc:
                pass
            else:
                hungry.append(layer)

        if surplus > 0 and hungry:
            per_layer = surplus // len(hungry)
            for layer in hungry:
                self.allocations[layer] += per_layer

    def stats(self) -> dict:
        return {
            layer: {"allocated": self.allocations[layer],
                    "used": self._used[layer],
                    "remaining": self.remaining(layer)}
            for layer in self.allocations
        }

    def global_stats(self) -> dict:
        return {
            "total_budget": self.total,
            "total_used": sum(self._used.values()),
            "force_used": self._force_used,
            "global_remaining": self.global_remaining,
        }


# ── Main Composer ─────────────────────────────────────────────────

MIN_CORE_SCORE = 0.3
MIN_BACKGROUND_SCORE = 0.15
MIN_CONFLICT_RELEVANCE = 0.2
FORCE_ADD_CAP_RATIO = 0.35

class ContextComposer:
    """Orchestrates the 5-stage context assembly pipeline.

    Stage 2 (Relevance Gate) is the key defense against topically-irrelevant
    content: every item must pass a query-relevance check before entering any
    layer. Uses cross-encoder when available, n-gram fallback otherwise.
    """

    def compose(self, query: str, raw: dict, max_tokens: int = 4000) -> dict:
        """Main entry point: raw recall data → composed context.

        Args:
            query: user's question
            raw: dict with keys: skills, kg_relations, memora, msa, evidence
            max_tokens: total token budget

        Returns:
            dict with: merged, layers, intent, budget_stats, warnings,
                       security_stats, gate_stats
        """
        intent = classify_intent(query)

        # ── Stage 2.5: Security Gate ──────────────────────────────
        security_stats = {}
        try:
            from memory_security import SecurityGate
            security_gate = SecurityGate()
            raw, security_stats = security_gate.filter_raw(raw, query)
        except ImportError:
            logger.warning("memory_security module not available, skipping security gate")
        except Exception as e:
            logger.error("Security gate failed: %s", e, exc_info=True)

        weights = LAYER_WEIGHTS[intent]
        budget = BudgetController(max_tokens, weights)
        force_cap = int(max_tokens * FORCE_ADD_CAP_RATIO)
        warnings: List[str] = []
        gate_threshold = effective_gate_threshold(query)
        gate_stats = {"passed": 0, "rejected": 0, "rejected_items": [],
                      "threshold": gate_threshold}

        l1_core = self._build_core_layer(
            raw, query, intent, budget, force_cap, gate_stats, gate_threshold)
        budget.redistribute()

        l2_concept = self._build_concept_layer(
            raw, query, intent, budget, gate_stats, gate_threshold)
        budget.redistribute()

        l3_background = self._build_background_layer(
            raw, query, intent, budget, gate_stats, gate_threshold)
        budget.redistribute()

        l4_conflicts = self._build_conflict_layer(
            raw, query, intent, budget, force_cap)

        merged = l1_core + l2_concept + l3_background + l4_conflicts

        gs = budget.global_stats()
        if gs["total_used"] > max_tokens:
            warnings.append(
                f"Budget overrun: {gs['total_used']}/{max_tokens} tokens "
                f"(force_used={gs['force_used']})")

        if gate_stats["rejected"] > 0:
            warnings.append(
                f"Relevance gate rejected {gate_stats['rejected']} items "
                f"(threshold={gate_threshold})")
            logger.info(
                "Relevance gate: passed=%d, rejected=%d for '%s'. "
                "Rejected samples: %s",
                gate_stats["passed"], gate_stats["rejected"], query[:40],
                [r[:60] for r in gate_stats["rejected_items"][:3]])

        if security_stats.get("blocked", 0) > 0:
            warnings.append(
                f"Security gate blocked {security_stats['blocked']} items")
        if security_stats.get("flagged", 0) > 0:
            warnings.append(
                f"Security gate flagged {security_stats['flagged']} items")

        logger.info(
            "Composed: intent=%s, L1=%d L2=%d L3=%d L4=%d, total=%d items, "
            "budget=%s, global=%s for '%s'",
            intent, len(l1_core), len(l2_concept), len(l3_background),
            len(l4_conflicts), len(merged),
            {k: f"{v['used']}/{v['allocated']}" for k, v in budget.stats().items()},
            f"{gs['total_used']}/{gs['total_budget']}",
            query[:60],
        )

        return {
            "merged": merged,
            "intent": intent,
            "layers": {
                "core": l1_core,
                "concept": l2_concept,
                "background": l3_background,
                "conflict": l4_conflicts,
            },
            "budget_stats": budget.stats(),
            "global_budget": gs,
            "gate_stats": gate_stats,
            "security_stats": security_stats,
            "warnings": warnings,
        }

    def _build_core_layer(self, raw: dict, query: str, intent: str,
                          budget: BudgetController,
                          force_cap: int, gate_stats: dict,
                          gate_threshold: float) -> list:
        """L1: Skills (force-added with cap) + relevance-gated evidence."""
        items = []

        for s in raw.get("skills", []):
            proc = s.get("procedures", "")[:200]
            body = proc if proc else s.get("content", "")[:200]
            entry = {
                "content": f"[Skill] {s.get('name', '?')}: {body}",
                "score": 1.0,
                "composite_score": 1.0,
                "system": "skill",
                "layer": "core",
                "metadata": {"skill_id": s.get("id"), "status": s.get("status")},
            }
            if budget.force_add("core", entry["content"], hard_cap=force_cap):
                items.append(entry)

        evidence = raw.get("evidence", [])
        scored_evidence = []
        for ev in evidence:
            content = ev.get("content", "")
            rel = _query_relevance(query, content)
            if rel < gate_threshold:
                gate_stats["rejected"] += 1
                gate_stats["rejected_items"].append(content[:80])
                continue
            gate_stats["passed"] += 1

            cs = score_item(ev, intent)
            scored_evidence.append({
                "content": content,
                "score": ev.get("score", 0),
                "composite_score": cs,
                "query_relevance": rel,
                "system": ev.get("system", "memora"),
                "layer": "core",
                "timestamp": ev.get("timestamp", ""),
                "metadata": ev.get("metadata", {}),
            })
        scored_evidence.sort(key=lambda x: x["composite_score"], reverse=True)

        deduped = _mmr_select(
            scored_evidence, budget.try_add, "core",
            diversity_lambda=0.7, min_score=MIN_CORE_SCORE)
        items.extend(deduped)

        return items

    def _build_concept_layer(self, raw: dict, query: str, intent: str,
                             budget: BudgetController,
                             gate_stats: dict,
                             gate_threshold: float) -> list:
        """L2: KG relations — relevance-gated, scored, MMR-deduped."""
        kg_relations = raw.get("kg_relations", [])
        candidates = []
        for kg_item in kg_relations:
            if kg_item.get("edge_type") == "contradicts":
                continue

            desc = kg_item.get("description", "")[:200]
            rel = _query_relevance(query, desc)
            if rel < gate_threshold:
                gate_stats["rejected"] += 1
                gate_stats["rejected_items"].append(f"[KG] {desc[:60]}")
                continue
            gate_stats["passed"] += 1

            relevance = kg_item.get("relevance", 0.5)
            kg_as_item = {
                "score": relevance,
                "timestamp": kg_item.get("metadata", {}).get("created_at", ""),
                "metadata": kg_item.get("metadata", {}),
            }
            cs = score_item(kg_as_item, intent)

            candidates.append({
                "content": f"[KG] {desc}",
                "score": relevance,
                "composite_score": cs,
                "query_relevance": rel,
                "system": "kg",
                "layer": "concept",
                "metadata": kg_item.get("metadata", {}),
            })

        candidates.sort(
            key=lambda x: (x.get("composite_score", 0)), reverse=True)

        return _mmr_select(
            candidates, budget.try_add, "concept",
            diversity_lambda=0.65, min_score=MIN_BACKGROUND_SCORE)

    def _build_background_layer(self, raw: dict, query: str, intent: str,
                                budget: BudgetController,
                                gate_stats: dict,
                                gate_threshold: float) -> list:
        """L3: Lower-relevance evidence — relevance-gated, scored, MMR-deduped."""
        all_evidence = raw.get("memora", []) + raw.get("msa", [])
        core_texts = set()
        for ev in raw.get("evidence", []):
            core_texts.add(ev.get("content", "")[:120])

        candidates = []
        for ev in all_evidence:
            content = ev.get("content", "")
            if content[:120] in core_texts:
                continue

            rel = _query_relevance(query, content)
            if rel < gate_threshold:
                gate_stats["rejected"] += 1
                gate_stats["rejected_items"].append(content[:80])
                continue
            gate_stats["passed"] += 1

            cs = score_item(ev, intent)
            if cs < MIN_BACKGROUND_SCORE:
                continue
            candidates.append({
                "content": content,
                "score": ev.get("score", 0),
                "composite_score": cs,
                "query_relevance": rel,
                "system": ev.get("system", "memora"),
                "layer": "background",
                "timestamp": ev.get("timestamp", ""),
                "metadata": ev.get("metadata", {}),
            })

        candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        return _mmr_select(
            candidates, budget.try_add, "background",
            diversity_lambda=0.6, min_score=MIN_BACKGROUND_SCORE)

    def _build_conflict_layer(self, raw: dict, query: str, intent: str,
                              budget: BudgetController,
                              force_cap: int) -> list:
        """L4: Contradictions — query-relevance-filtered, force-added with cap."""
        items = []

        for kg_item in raw.get("kg_relations", []):
            if kg_item.get("edge_type") != "contradicts":
                continue
            desc = kg_item.get("description", "")[:200]
            relevance = kg_item.get("relevance", 0.0)
            if relevance < MIN_CONFLICT_RELEVANCE:
                continue
            entry = {
                "content": f"[CONFLICT] {desc}",
                "score": relevance,
                "composite_score": 0.8 + relevance * 0.2,
                "system": "kg_conflict",
                "layer": "conflict",
                "metadata": kg_item.get("metadata", {}),
            }
            if budget.force_add("conflict", entry["content"],
                                hard_cap=force_cap):
                items.append(entry)

        contradictions = raw.get("contradictions", [])
        for cr in contradictions:
            decision = cr.get("decision_content", "")[:100]
            risk = cr.get("risk_score", 0)
            contra = cr.get("contradicting", [])
            if not contra:
                continue
            contra_text = contra[0].get("content", "")[:100] if contra else ""
            query_sim = _text_similarity(
                query, f"{decision} {contra_text}")
            if query_sim < 0.05 and risk < 0.5:
                continue
            entry = {
                "content": (f"[CONFLICT risk={risk:.0%}] "
                            f"决策「{decision}」存在矛盾证据: {contra_text}"),
                "score": risk,
                "composite_score": 0.8 + risk * 0.2,
                "system": "contradiction",
                "layer": "conflict",
                "metadata": {"risk_score": risk},
            }
            if budget.force_add("conflict", entry["content"],
                                hard_cap=force_cap):
                items.append(entry)

        return items
