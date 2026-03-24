"""
Phase 4: Context Efficiency Benchmark

Measures:
- Token budget: How many tokens does each pipeline stage consume?
- Information density: Relevant tokens / total tokens
- Compression ratio: Raw memories → session-context → skills
- Latency: Time for each retrieval method
- Coverage: How many memories are "reachable" via different methods?
"""

import json
import logging
import time
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

SERVER = "http://127.0.0.1:18790"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word for English, ~1.5 for Chinese."""
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / max(len(text), 1)
    words = len(text.split())
    if ratio > 0.8:
        return int(words * 1.3)
    else:
        return int(len(text) * 0.6)


def _api_get(path: str, timeout: float = 10):
    try:
        r = requests.get(f"{SERVER}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _api_post(path: str, body: dict = None, timeout: float = 15):
    try:
        r = requests.post(f"{SERVER}{path}", json=body or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def measure_raw_memory_tokens() -> Dict:
    """Measure token cost of dumping all memories."""
    results = _api_post("/search", {"query": "everything", "top_k": 100})
    if not results:
        return {"total_memories": 0, "total_tokens": 0}

    items = results.get("results", [])
    all_text = "\n\n".join(r.get("content", "") for r in items)
    tokens = _estimate_tokens(all_text)

    return {
        "total_memories": len(items),
        "total_chars": len(all_text),
        "total_tokens": tokens,
        "avg_tokens_per_memory": tokens // max(len(items), 1),
    }


def measure_session_context_tokens() -> Dict:
    """Measure token cost of session-context endpoint."""
    t0 = time.time()
    ctx = _api_get("/session-context")
    latency = (time.time() - t0) * 1000

    if not ctx:
        return {"tokens": 0, "latency_ms": latency}

    text = json.dumps(ctx, ensure_ascii=False)
    tokens = _estimate_tokens(text)

    sections = {}
    for key, val in ctx.items():
        section_text = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
        sections[key] = _estimate_tokens(section_text)

    return {
        "total_tokens": tokens,
        "total_chars": len(text),
        "latency_ms": round(latency, 1),
        "sections": sections,
    }


def measure_recall_tokens(query: str) -> Dict:
    """Measure token cost of assembled recall for a query."""
    t0 = time.time()
    result = _api_post("/recall", {"query": query, "top_k": 8})
    latency = (time.time() - t0) * 1000

    if not result:
        return {"tokens": 0, "latency_ms": latency}

    layers = {}
    total_tokens = 0

    for layer_name in ["skills", "kg_relations", "merged"]:
        items = result.get(layer_name, [])
        layer_text = json.dumps(items, ensure_ascii=False)
        t = _estimate_tokens(layer_text)
        layers[layer_name] = {
            "count": len(items),
            "tokens": t,
        }
        total_tokens += t

    return {
        "total_tokens": total_tokens,
        "latency_ms": round(latency, 1),
        "layers": layers,
    }


def measure_search_tokens(query: str) -> Dict:
    """Measure token cost of simple vector search."""
    t0 = time.time()
    result = _api_post("/search", {"query": query, "top_k": 5})
    latency = (time.time() - t0) * 1000

    if not result:
        return {"tokens": 0, "latency_ms": latency}

    items = result.get("results", [])
    all_text = "\n\n".join(r.get("content", "") for r in items)
    tokens = _estimate_tokens(all_text)

    return {
        "count": len(items),
        "tokens": tokens,
        "latency_ms": round(latency, 1),
    }


def measure_briefing_tokens() -> Dict:
    """Measure token cost of daily briefing."""
    t0 = time.time()
    briefing = _api_get("/briefing")
    latency = (time.time() - t0) * 1000

    if not briefing:
        return {"tokens": 0, "latency_ms": latency}

    text = briefing.get("text", json.dumps(briefing, ensure_ascii=False))
    tokens = _estimate_tokens(text)

    return {
        "tokens": tokens,
        "chars": len(text),
        "latency_ms": round(latency, 1),
    }


def measure_kg_tokens() -> Dict:
    """Measure token cost of KG data."""
    kg = _api_get("/kg/status") or {}
    graph = _api_get("/kg/graph") or {"nodes": [], "edges": []}

    node_text = " ".join(n.get("content", "") for n in graph["nodes"])
    edge_text = " ".join(
        f"{e.get('source_id','')} {e.get('edge_type','')} {e.get('target_id','')}"
        for e in graph["edges"])

    node_tokens = _estimate_tokens(node_text)
    edge_tokens = _estimate_tokens(edge_text)

    return {
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "node_tokens": node_tokens,
        "edge_tokens": edge_tokens,
        "total_tokens": node_tokens + edge_tokens,
    }


def measure_skills_tokens() -> Dict:
    """Measure token cost of all skills."""
    skills = _api_get("/skills") or {"skills": []}
    all_text = json.dumps(skills["skills"], ensure_ascii=False)
    tokens = _estimate_tokens(all_text)

    return {
        "skill_count": len(skills["skills"]),
        "tokens": tokens,
    }


def run() -> Dict:
    """Run full Phase 4 benchmark."""
    test_queries = [
        "vector database architecture",
        "quantum computing error correction",
        "memory system engineering lessons",
    ]

    raw = measure_raw_memory_tokens()
    session = measure_session_context_tokens()
    briefing = measure_briefing_tokens()
    kg = measure_kg_tokens()
    skills = measure_skills_tokens()

    recall_measurements = []
    search_measurements = []
    for q in test_queries:
        recall_measurements.append({"query": q, **measure_recall_tokens(q)})
        search_measurements.append({"query": q, **measure_search_tokens(q)})

    avg_recall_tokens = (sum(r["total_tokens"] for r in recall_measurements)
                         / max(len(recall_measurements), 1))
    avg_recall_latency = (sum(r["latency_ms"] for r in recall_measurements)
                          / max(len(recall_measurements), 1))
    avg_search_tokens = (sum(r["tokens"] for r in search_measurements)
                         / max(len(search_measurements), 1))
    avg_search_latency = (sum(r["latency_ms"] for r in search_measurements)
                          / max(len(search_measurements), 1))

    raw_total = raw.get("total_tokens", 1) or 1
    session_total = session.get("total_tokens", 0)
    recall_total = avg_recall_tokens

    compression_session = 1 - (session_total / raw_total) if raw_total > 0 else 0
    compression_recall = 1 - (recall_total / raw_total) if raw_total > 0 else 0

    return {
        "phase": "4_context_efficiency",
        "summary": {
            "raw_memory_tokens": raw.get("total_tokens", 0),
            "session_context_tokens": session_total,
            "avg_recall_tokens": round(avg_recall_tokens),
            "avg_search_tokens": round(avg_search_tokens),
            "briefing_tokens": briefing.get("tokens", 0),
            "kg_total_tokens": kg.get("total_tokens", 0),
            "skills_tokens": skills.get("tokens", 0),
            "compression_ratio_session": round(compression_session, 3),
            "compression_ratio_recall": round(compression_recall, 3),
            "avg_recall_latency_ms": round(avg_recall_latency, 1),
            "avg_search_latency_ms": round(avg_search_latency, 1),
        },
        "token_budget_comparison": {
            "raw_dump_all": raw.get("total_tokens", 0),
            "session_context": session_total,
            "assembled_recall_per_query": round(avg_recall_tokens),
            "search_per_query": round(avg_search_tokens),
            "briefing": briefing.get("tokens", 0),
            "note": "Lower is better for the same information coverage",
        },
        "details": {
            "raw_memories": raw,
            "session_context": session,
            "briefing": briefing,
            "kg": kg,
            "skills": skills,
            "recall_per_query": recall_measurements,
            "search_per_query": search_measurements,
        },
    }
