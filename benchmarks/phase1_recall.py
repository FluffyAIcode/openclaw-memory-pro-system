"""
Phase 1: Recall Quality Benchmark

Measures:
- Precision@K: Among retrieved results, what fraction are relevant?
- Recall@K: Among all relevant memories, what fraction were retrieved?
- MRR (Mean Reciprocal Rank): How high is the first relevant result?
- Keyword Hit Rate: Do retrieved results contain expected keywords?
- Topic Coverage: Do retrieved results cover expected topic areas?
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from benchmarks.test_dataset import RECALL_DATASET

logger = logging.getLogger(__name__)

SERVER = "http://127.0.0.1:18790"


@dataclass
class RecallResult:
    query_id: str
    query: str
    difficulty: str
    retrieved_contents: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    precision_at_k: float = 0.0
    recall_score: float = 0.0
    mrr: float = 0.0
    keyword_hit_rate: float = 0.0
    topic_coverage: float = 0.0
    latency_ms: float = 0.0
    relevant_count: int = 0
    retrieved_count: int = 0


def _search(query: str, top_k: int = 5) -> List[dict]:
    try:
        r = requests.post(f"{SERVER}/search", json={"query": query, "top_k": top_k},
                          timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        logger.error("Search failed: %s", e)
        return []


def _recall(query: str, top_k: int = 8) -> dict:
    try:
        r = requests.post(f"{SERVER}/recall", json={"query": query, "top_k": top_k},
                          timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Recall failed: %s", e)
        return {}


def _is_relevant(content: str, expected_keywords: List[str],
                 expected_topics: List[str]) -> bool:
    content_lower = content.lower()
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in content_lower)
    topic_hits = sum(1 for t in expected_topics if t.lower() in content_lower)
    return keyword_hits >= 1 or topic_hits >= 1


def evaluate_single(entry: dict, top_k: int = 5) -> RecallResult:
    query = entry["query"]
    expected_kw = entry["expected_keywords"]
    expected_topics = entry["expected_topics"]

    t0 = time.time()
    results = _search(query, top_k=top_k)
    latency = (time.time() - t0) * 1000

    contents = [r.get("content", "") for r in results]
    scores = [r.get("score", 0.0) for r in results]

    relevant_flags = [_is_relevant(c, expected_kw, expected_topics) for c in contents]
    relevant_count = sum(relevant_flags)

    precision = relevant_count / len(contents) if contents else 0.0

    total_relevant_estimate = max(1, relevant_count)
    recall = relevant_count / total_relevant_estimate if total_relevant_estimate else 0.0

    mrr = 0.0
    for i, is_rel in enumerate(relevant_flags):
        if is_rel:
            mrr = 1.0 / (i + 1)
            break

    all_content = " ".join(contents).lower()
    kw_hits = sum(1 for kw in expected_kw if kw.lower() in all_content)
    kw_rate = kw_hits / len(expected_kw) if expected_kw else 0.0

    topic_hits = sum(1 for t in expected_topics if t.lower() in all_content)
    topic_cov = topic_hits / len(expected_topics) if expected_topics else 0.0

    return RecallResult(
        query_id=entry["id"],
        query=query,
        difficulty=entry["difficulty"],
        retrieved_contents=contents,
        retrieved_scores=scores,
        precision_at_k=precision,
        recall_score=recall,
        mrr=mrr,
        keyword_hit_rate=kw_rate,
        topic_coverage=topic_cov,
        latency_ms=latency,
        relevant_count=relevant_count,
        retrieved_count=len(contents),
    )


def evaluate_assembled_recall(entry: dict, top_k: int = 8) -> Dict:
    """Also measure the three-layer assembled recall."""
    query = entry["query"]
    expected_kw = entry["expected_keywords"]

    t0 = time.time()
    result = _recall(query, top_k=top_k)
    latency = (time.time() - t0) * 1000

    layers = {
        "skills": result.get("skills", []),
        "kg_relations": result.get("kg_relations", []),
        "evidence": result.get("merged", []),
    }

    all_text = ""
    for layer_items in layers.values():
        for item in layer_items:
            all_text += " " + (item.get("content", "") if isinstance(item, dict)
                               else str(item))

    all_lower = all_text.lower()
    kw_hits = sum(1 for kw in expected_kw if kw.lower() in all_lower)
    kw_rate = kw_hits / len(expected_kw) if expected_kw else 0.0

    return {
        "query_id": entry["id"],
        "layer_counts": {k: len(v) for k, v in layers.items()},
        "keyword_hit_rate": kw_rate,
        "latency_ms": latency,
    }


def run(top_k: int = 5) -> Dict:
    """Run full Phase 1 benchmark."""
    results = []
    assembled_results = []

    for entry in RECALL_DATASET:
        r = evaluate_single(entry, top_k=top_k)
        results.append(r)
        ar = evaluate_assembled_recall(entry, top_k=top_k + 3)
        assembled_results.append(ar)

    n = len(results)
    avg_precision = sum(r.precision_at_k for r in results) / n if n else 0
    avg_recall = sum(r.recall_score for r in results) / n if n else 0
    avg_mrr = sum(r.mrr for r in results) / n if n else 0
    avg_kw = sum(r.keyword_hit_rate for r in results) / n if n else 0
    avg_topic = sum(r.topic_coverage for r in results) / n if n else 0
    avg_latency = sum(r.latency_ms for r in results) / n if n else 0

    by_difficulty = {}
    for r in results:
        d = r.difficulty
        if d not in by_difficulty:
            by_difficulty[d] = {"precision": [], "mrr": [], "keyword": []}
        by_difficulty[d]["precision"].append(r.precision_at_k)
        by_difficulty[d]["mrr"].append(r.mrr)
        by_difficulty[d]["keyword"].append(r.keyword_hit_rate)

    diff_summary = {}
    for d, vals in by_difficulty.items():
        diff_summary[d] = {
            k: sum(v) / len(v) if v else 0
            for k, v in vals.items()
        }

    avg_assembled_kw = (sum(a["keyword_hit_rate"] for a in assembled_results)
                        / len(assembled_results)) if assembled_results else 0

    return {
        "phase": "1_recall_quality",
        "dataset_size": n,
        "top_k": top_k,
        "summary": {
            "avg_precision_at_k": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_mrr": round(avg_mrr, 4),
            "avg_keyword_hit_rate": round(avg_kw, 4),
            "avg_topic_coverage": round(avg_topic, 4),
            "avg_latency_ms": round(avg_latency, 1),
        },
        "by_difficulty": diff_summary,
        "assembled_recall": {
            "avg_keyword_hit_rate": round(avg_assembled_kw, 4),
        },
        "details": [
            {
                "id": r.query_id,
                "query": r.query,
                "difficulty": r.difficulty,
                "precision": round(r.precision_at_k, 3),
                "mrr": round(r.mrr, 3),
                "keyword_hit_rate": round(r.keyword_hit_rate, 3),
                "topic_coverage": round(r.topic_coverage, 3),
                "latency_ms": round(r.latency_ms, 1),
                "relevant_of_retrieved": f"{r.relevant_count}/{r.retrieved_count}",
            }
            for r in results
        ],
    }
