"""
Phase 2: Faithfulness Benchmark

Measures:
- Grounding Score: What % of answer claims are supported by retrieved context?
- Hallucination Rate: What % of answer claims have NO support in retrieved context?
- Trap Avoidance: Does the model avoid known hallucination traps?
- Claim Precision: Among claims made, what fraction are faithful to context?

Uses LLM-as-judge to decompose answers into atomic claims and verify each.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from benchmarks.test_dataset import FAITHFULNESS_DATASET

logger = logging.getLogger(__name__)

SERVER = "http://127.0.0.1:18790"

JUDGE_SYSTEM = """You are a strict factual verification judge. Your job is to check whether claims in an answer are supported by the provided context.

Rules:
1. A claim is SUPPORTED if the context contains evidence for it (even paraphrased).
2. A claim is UNSUPPORTED if the context has no relevant information about it.
3. A claim is CONTRADICTED if the context directly contradicts it.
4. Be strict — vague or tangential matches do NOT count as supported.

Respond in JSON only."""

DECOMPOSE_PROMPT = """Decompose this answer into atomic factual claims (one fact per claim).

Answer: {answer}

Return a JSON array of strings, each being one atomic claim. Example:
["The system uses JSONL storage", "The embedding model is nomic-embed-text"]

Return ONLY the JSON array, nothing else."""

VERIFY_PROMPT = """Given the context and a list of claims, classify each claim.

Context:
{context}

Claims to verify:
{claims}

For each claim, respond with a JSON array of objects:
[{{"claim": "...", "verdict": "supported|unsupported|contradicted", "evidence": "brief quote or null"}}]

Return ONLY the JSON array."""

TRAP_CHECK_PROMPT = """Check if this answer contains any of these specific false claims (hallucination traps).

Answer: {answer}

Known false claims to check for:
{traps}

Return a JSON array of objects:
[{{"trap": "...", "found": true/false, "quote": "relevant quote from answer or null"}}]

Return ONLY the JSON array."""


@dataclass
class FaithfulnessResult:
    query_id: str
    query: str
    answer: str
    retrieved_context: str
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    grounding_score: float = 0.0
    hallucination_rate: float = 0.0
    traps_triggered: int = 0
    traps_total: int = 0
    trap_avoidance_rate: float = 0.0
    latency_ms: float = 0.0
    claims_detail: List[dict] = field(default_factory=list)
    llm_available: bool = True


_JUDGE_MODEL = None

def _get_judge_model() -> str:
    global _JUDGE_MODEL
    if _JUDGE_MODEL is None:
        try:
            from llm_client import FAST_MODEL
            _JUDGE_MODEL = FAST_MODEL
        except ImportError:
            _JUDGE_MODEL = "deepseek/deepseek-chat"
        logger.info("Faithfulness judge using model: %s", _JUDGE_MODEL)
    return _JUDGE_MODEL

def _llm_generate(prompt: str, system: str = "", temperature: float = 0.1) -> Optional[str]:
    try:
        from llm_client import generate
        return generate(prompt, system=system, temperature=temperature,
                        max_tokens=2048, model=_get_judge_model(), timeout=60)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


def _search(query: str, top_k: int = 5) -> List[dict]:
    try:
        r = requests.post(f"{SERVER}/search", json={"query": query, "top_k": top_k},
                          timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


def _recall(query: str) -> dict:
    try:
        r = requests.post(f"{SERVER}/recall", json={"query": query, "top_k": 8},
                          timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _generate_answer_with_context(query: str, context: str) -> Optional[str]:
    prompt = f"""Based on the following context, answer the question. Only use information from the context. If unsure, say so.

Context:
{context}

Question: {query}

Answer:"""
    return _llm_generate(prompt, temperature=0.3)


def _parse_json(text: str) -> Optional[list]:
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
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _keyword_faithfulness(answer: str, context: str,
                          expected_claims: List[str],
                          traps: List[str]) -> Dict:
    """Fallback: keyword-based faithfulness when LLM is unavailable."""
    answer_lower = answer.lower()
    context_lower = context.lower()

    supported = 0
    for claim in expected_claims:
        claim_words = claim.lower().split()
        if any(w in answer_lower for w in claim_words if len(w) > 3):
            if any(w in context_lower for w in claim_words if len(w) > 3):
                supported += 1

    trap_hits = 0
    for trap in traps:
        trap_words = [w for w in trap.lower().split() if len(w) > 3]
        if trap_words and sum(1 for w in trap_words if w in answer_lower) >= len(trap_words) * 0.5:
            trap_hits += 1

    total = max(len(expected_claims), 1)
    return {
        "total_claims": total,
        "supported": supported,
        "unsupported": total - supported,
        "contradicted": 0,
        "grounding_score": supported / total,
        "hallucination_rate": (total - supported) / total,
        "traps_triggered": trap_hits,
        "traps_total": len(traps),
    }


def evaluate_single(entry: dict) -> FaithfulnessResult:
    query = entry["query"]
    expected_claims = entry["expected_grounded_claims"]
    traps = entry["hallucination_traps"]

    t0 = time.time()

    search_results = _search(query, top_k=5)
    context = "\n\n".join(r.get("content", "") for r in search_results)

    answer = _generate_answer_with_context(query, context)
    if not answer:
        latency = (time.time() - t0) * 1000
        kw_result = _keyword_faithfulness("", context, expected_claims, traps)
        return FaithfulnessResult(
            query_id=entry["id"], query=query, answer="(LLM unavailable)",
            retrieved_context=context[:500],
            llm_available=False, latency_ms=latency,
            **{k: v for k, v in kw_result.items()
               if k in ("total_claims", "grounding_score", "hallucination_rate",
                         "traps_triggered", "traps_total")},
            supported_claims=kw_result["supported"],
            unsupported_claims=kw_result["unsupported"],
            trap_avoidance_rate=(1 - kw_result["traps_triggered"] / max(kw_result["traps_total"], 1)),
        )

    decomposed = _parse_json(_llm_generate(
        DECOMPOSE_PROMPT.format(answer=answer), system=JUDGE_SYSTEM))

    if not decomposed:
        decomposed = [s.strip() for s in answer.split(". ") if len(s.strip()) > 10]

    verified = _parse_json(_llm_generate(
        VERIFY_PROMPT.format(context=context[:3000],
                             claims=json.dumps(decomposed, ensure_ascii=False)),
        system=JUDGE_SYSTEM))

    supported = unsupported = contradicted = 0
    claims_detail = []
    if verified:
        for v in verified:
            verdict = v.get("verdict", "unsupported")
            if verdict == "supported":
                supported += 1
            elif verdict == "contradicted":
                contradicted += 1
            else:
                unsupported += 1
            claims_detail.append(v)
    else:
        kw = _keyword_faithfulness(answer, context, expected_claims, traps)
        supported = kw["supported"]
        unsupported = kw["unsupported"]

    trap_results = _parse_json(_llm_generate(
        TRAP_CHECK_PROMPT.format(answer=answer,
                                 traps=json.dumps(traps, ensure_ascii=False)),
        system=JUDGE_SYSTEM))

    traps_hit = 0
    if trap_results:
        traps_hit = sum(1 for t in trap_results if t.get("found"))
    else:
        answer_lower = answer.lower()
        for trap in traps:
            words = [w for w in trap.lower().split() if len(w) > 3]
            if words and sum(1 for w in words if w in answer_lower) >= len(words) * 0.5:
                traps_hit += 1

    total = max(supported + unsupported + contradicted, 1)
    latency = (time.time() - t0) * 1000

    return FaithfulnessResult(
        query_id=entry["id"],
        query=query,
        answer=answer[:500],
        retrieved_context=context[:500],
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        contradicted_claims=contradicted,
        grounding_score=supported / total,
        hallucination_rate=(unsupported + contradicted) / total,
        traps_triggered=traps_hit,
        traps_total=len(traps),
        trap_avoidance_rate=1 - (traps_hit / max(len(traps), 1)),
        latency_ms=latency,
        claims_detail=claims_detail,
    )


def run() -> Dict:
    """Run full Phase 2 benchmark."""
    results = []
    total = len(FAITHFULNESS_DATASET)
    for i, entry in enumerate(FAITHFULNESS_DATASET, 1):
        print(f"  [{i}/{total}] Evaluating {entry['id']}: {entry['query'][:50]}...",
              flush=True)
        r = evaluate_single(entry)
        status = "LLM" if r.llm_available else "heuristic"
        print(f"           grounding={r.grounding_score:.0%} "
              f"hallucination={r.hallucination_rate:.0%} "
              f"({status}, {r.latency_ms:.0f}ms)", flush=True)
        results.append(r)

    n = len(results)
    llm_ok = sum(1 for r in results if r.llm_available)

    avg_grounding = sum(r.grounding_score for r in results) / n if n else 0
    avg_hallucination = sum(r.hallucination_rate for r in results) / n if n else 0
    avg_trap_avoid = sum(r.trap_avoidance_rate for r in results) / n if n else 0
    total_claims = sum(r.total_claims for r in results)
    total_supported = sum(r.supported_claims for r in results)
    total_traps_hit = sum(r.traps_triggered for r in results)
    total_traps = sum(r.traps_total for r in results)

    return {
        "phase": "2_faithfulness",
        "dataset_size": n,
        "llm_available": llm_ok,
        "summary": {
            "avg_grounding_score": round(avg_grounding, 4),
            "avg_hallucination_rate": round(avg_hallucination, 4),
            "avg_trap_avoidance_rate": round(avg_trap_avoid, 4),
            "total_claims_verified": total_claims,
            "total_supported": total_supported,
            "total_traps_triggered": total_traps_hit,
            "total_traps_checked": total_traps,
        },
        "details": [
            {
                "id": r.query_id,
                "query": r.query,
                "grounding_score": round(r.grounding_score, 3),
                "hallucination_rate": round(r.hallucination_rate, 3),
                "trap_avoidance": round(r.trap_avoidance_rate, 3),
                "claims": f"{r.supported_claims}S/{r.unsupported_claims}U/{r.contradicted_claims}C",
                "traps": f"{r.traps_triggered}/{r.traps_total} triggered",
                "latency_ms": round(r.latency_ms, 1),
                "llm_available": r.llm_available,
            }
            for r in results
        ],
    }
