"""
Phase 3: End-to-End A/B Test

Compares LLM answer quality WITH vs WITHOUT memory context.

Mode A (no memory): Vanilla LLM — no retrieved context
Mode B (with memory): LLM + assembled recall context

Evaluation dimensions (LLM-as-judge):
- Specificity: Does the answer contain concrete, specific details?
- Accuracy: Are factual claims correct?
- Personalization: Is the answer tailored to the user's actual context?
- Groundedness: Are claims traceable to real stored memories?
- Completeness: Does the answer cover the key aspects?

Also falls back to keyword heuristics when LLM is unavailable.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from benchmarks.test_dataset import AB_TEST_QUERIES

logger = logging.getLogger(__name__)

SERVER = "http://127.0.0.1:18790"

JUDGE_SYSTEM = """You are a quality evaluation judge comparing two answers to the same question. Score each answer on the given criteria, from 1 (worst) to 5 (best). Be strict and objective."""

COMPARE_PROMPT = """Question: {query}

Answer A (no memory context):
{answer_a}

Answer B (with memory context):
{answer_b}

Evaluation criteria: {criteria}

Score EACH answer on EACH criterion (1-5). Then give an overall winner.

Respond in JSON:
{{
  "scores_a": {{"criterion1": score, ...}},
  "scores_b": {{"criterion1": score, ...}},
  "avg_a": float,
  "avg_b": float,
  "winner": "A" or "B" or "tie",
  "reasoning": "brief explanation"
}}

Return ONLY the JSON."""


@dataclass
class ABResult:
    query_id: str
    query: str
    answer_a: str
    answer_b: str
    context_used: str
    scores_a: Dict[str, float] = field(default_factory=dict)
    scores_b: Dict[str, float] = field(default_factory=dict)
    avg_a: float = 0.0
    avg_b: float = 0.0
    winner: str = "unknown"
    reasoning: str = ""
    latency_a_ms: float = 0.0
    latency_b_ms: float = 0.0
    llm_judge_available: bool = True


_JUDGE_MODEL = None

def _get_judge_model() -> str:
    global _JUDGE_MODEL
    if _JUDGE_MODEL is None:
        try:
            from llm_client import FAST_MODEL
            _JUDGE_MODEL = FAST_MODEL
        except ImportError:
            _JUDGE_MODEL = "deepseek/deepseek-chat"
        logger.info("A/B judge using model: %s", _JUDGE_MODEL)
    return _JUDGE_MODEL

def _llm_generate(prompt: str, system: str = "", temperature: float = 0.3) -> Optional[str]:
    try:
        from llm_client import generate
        return generate(prompt, system=system, temperature=temperature,
                        max_tokens=2048, model=_get_judge_model(), timeout=60)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


def _recall(query: str) -> dict:
    try:
        r = requests.post(f"{SERVER}/recall", json={"query": query, "top_k": 8},
                          timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _build_context(recall_result: dict) -> str:
    parts = []
    for sk in recall_result.get("skills", []):
        parts.append(f"[Skill] {sk.get('name', '')}: {sk.get('content', '')}")
    for kg in recall_result.get("kg_relations", []):
        parts.append(f"[KG] {kg.get('content', str(kg))}")
    for ev in recall_result.get("merged", []):
        c = ev.get("content", "")
        if c:
            parts.append(f"[Memory] {c}")
    return "\n\n".join(parts)


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _keyword_judge(answer_a: str, answer_b: str, context: str,
                   criteria: List[str]) -> Dict:
    """Fallback heuristic when LLM judge is unavailable."""
    a_lower = answer_a.lower()
    b_lower = answer_b.lower()
    ctx_lower = context.lower()

    ctx_words = set(w for w in ctx_lower.split() if len(w) > 4)

    overlap_a = len(set(a_lower.split()) & ctx_words)
    overlap_b = len(set(b_lower.split()) & ctx_words)

    scores_a = {}
    scores_b = {}
    for c in criteria:
        if c in ("specificity", "personalization", "grounding"):
            scores_a[c] = min(5, 1 + overlap_a // 3)
            scores_b[c] = min(5, 1 + overlap_b // 3)
        elif c == "completeness":
            scores_a[c] = min(5, 1 + len(answer_a) // 100)
            scores_b[c] = min(5, 1 + len(answer_b) // 100)
        else:
            scores_a[c] = 3
            scores_b[c] = 3

    avg_a = sum(scores_a.values()) / max(len(scores_a), 1)
    avg_b = sum(scores_b.values()) / max(len(scores_b), 1)
    winner = "B" if avg_b > avg_a + 0.3 else ("A" if avg_a > avg_b + 0.3 else "tie")

    return {
        "scores_a": scores_a,
        "scores_b": scores_b,
        "avg_a": avg_a,
        "avg_b": avg_b,
        "winner": winner,
        "reasoning": "keyword heuristic (LLM unavailable)",
    }


def evaluate_single(entry: dict) -> ABResult:
    query = entry["query"]
    criteria = entry["eval_criteria"]

    # Mode A: no memory
    t0 = time.time()
    answer_a = _llm_generate(
        f"Answer this question concisely:\n\n{query}",
        system="You are a helpful AI assistant. Answer based on your general knowledge.",
        temperature=0.3)
    lat_a = (time.time() - t0) * 1000

    if not answer_a:
        answer_a = "(LLM unavailable)"

    # Mode B: with memory context
    recall_data = _recall(query)
    context = _build_context(recall_data)

    t0 = time.time()
    if context.strip():
        answer_b = _llm_generate(
            f"""Based on the following personal memory context AND your general knowledge, answer the question. Prioritize information from the context.

Context:
{context[:4000]}

Question: {query}

Answer:""",
            system="You are a helpful AI assistant with access to the user's personal memory system. Use the provided context to give specific, personalized answers.",
            temperature=0.3)
    else:
        answer_b = _llm_generate(
            f"Answer this question concisely:\n\n{query}",
            system="You are a helpful AI assistant.",
            temperature=0.3)
    lat_b = (time.time() - t0) * 1000

    if not answer_b:
        answer_b = "(LLM unavailable)"

    # Judge
    llm_judge_ok = True
    judge_result = None

    if answer_a != "(LLM unavailable)" and answer_b != "(LLM unavailable)":
        judge_text = _llm_generate(
            COMPARE_PROMPT.format(
                query=query,
                answer_a=answer_a[:2000],
                answer_b=answer_b[:2000],
                criteria=", ".join(criteria)),
            system=JUDGE_SYSTEM, temperature=0.1)
        judge_result = _parse_json(judge_text)

    if not judge_result:
        llm_judge_ok = False
        judge_result = _keyword_judge(answer_a, answer_b, context, criteria)

    return ABResult(
        query_id=entry["id"],
        query=query,
        answer_a=answer_a[:500],
        answer_b=answer_b[:500],
        context_used=context[:300],
        scores_a=judge_result.get("scores_a", {}),
        scores_b=judge_result.get("scores_b", {}),
        avg_a=judge_result.get("avg_a", 0),
        avg_b=judge_result.get("avg_b", 0),
        winner=judge_result.get("winner", "unknown"),
        reasoning=judge_result.get("reasoning", ""),
        latency_a_ms=lat_a,
        latency_b_ms=lat_b,
        llm_judge_available=llm_judge_ok,
    )


def run() -> Dict:
    """Run full Phase 3 A/B benchmark."""
    results = []
    total = len(AB_TEST_QUERIES)
    for i, entry in enumerate(AB_TEST_QUERIES, 1):
        print(f"  [{i}/{total}] A/B testing {entry['id']}: {entry['query'][:50]}...",
              flush=True)
        r = evaluate_single(entry)
        judge_tag = "LLM" if r.llm_judge_available else "heuristic"
        print(f"           winner={r.winner} A={r.avg_a:.1f} B={r.avg_b:.1f} "
              f"({judge_tag})", flush=True)
        results.append(r)

    n = len(results)
    wins_a = sum(1 for r in results if r.winner == "A")
    wins_b = sum(1 for r in results if r.winner == "B")
    ties = sum(1 for r in results if r.winner == "tie")
    avg_a = sum(r.avg_a for r in results) / n if n else 0
    avg_b = sum(r.avg_b for r in results) / n if n else 0
    delta = avg_b - avg_a

    return {
        "phase": "3_ab_test",
        "dataset_size": n,
        "summary": {
            "wins_no_memory": wins_a,
            "wins_with_memory": wins_b,
            "ties": ties,
            "avg_score_no_memory": round(avg_a, 3),
            "avg_score_with_memory": round(avg_b, 3),
            "memory_delta": round(delta, 3),
            "memory_win_rate": round(wins_b / n, 3) if n else 0,
        },
        "details": [
            {
                "id": r.query_id,
                "query": r.query,
                "winner": r.winner,
                "avg_no_memory": round(r.avg_a, 2),
                "avg_with_memory": round(r.avg_b, 2),
                "delta": round(r.avg_b - r.avg_a, 2),
                "scores_a": r.scores_a,
                "scores_b": r.scores_b,
                "reasoning": r.reasoning[:200],
                "llm_judge": r.llm_judge_available,
            }
            for r in results
        ],
    }
