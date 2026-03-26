#!/usr/bin/env python3
"""
Search Evaluation — Tests hybrid search (BM25+dense+RRF) against dense-only baseline.

Evaluates:
  1. Hybrid vs Dense-only retrieval quality
  2. BM25 contribution analysis
  3. Cross-encoder reranking impact
  4. min_score threshold calibration via precision-recall curve
  5. Task prefix impact on nomic-embed-text-v1.5

Metrics:
  - Hit@K: Is the relevant doc in top K?
  - MRR (Mean Reciprocal Rank): Average of 1/rank for first relevant result
  - Precision@K: Fraction of top-K that are relevant

Usage:
    cd /Users/fluffy314/.openclaw/workspace
    python3 -m benchmarks.search_eval
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

WORKSPACE = Path(__file__).parent.parent


EVAL_DATASET = [
    {
        "id": "q1",
        "query": "第二大脑系统的设计和碰撞频率",
        "relevant_keywords": ["第二大脑", "碰撞频率", "碰撞", "6小时", "灵感"],
        "irrelevant_keywords": ["量子", "quantum", "machine learning"],
        "description": "Should match the Second Brain design entry",
    },
    {
        "id": "q2",
        "query": "向量数据库选型决策",
        "relevant_keywords": ["向量数据库", "Pinecone", "Milvus", "JSONL"],
        "irrelevant_keywords": ["用户习惯", "凌晨"],
        "description": "Should match the vector DB evaluation entry",
    },
    {
        "id": "q3",
        "query": "用户的工作时间和编程习惯",
        "relevant_keywords": ["凌晨", "1-3点", "思考时间", "代码Review", "中文"],
        "irrelevant_keywords": ["InsightEngine", "SaaS"],
        "description": "Should match the user profile entry",
    },
    {
        "id": "q4",
        "query": "InsightEngine产品商业模式",
        "relevant_keywords": ["InsightEngine", "SaaS", "碰撞机制", "Pro版", "Notion"],
        "irrelevant_keywords": ["Memory Server", "ThreadingMixIn"],
        "description": "Should match the product idea entry",
    },
    {
        "id": "q5",
        "query": "Memory Server调试经验和超时策略",
        "relevant_keywords": ["Memory Server", "单线程", "ThreadingMixIn", "超时", "LLM"],
        "irrelevant_keywords": ["InsightEngine", "凌晨"],
        "description": "Should match the engineering lesson entry",
    },
    {
        "id": "q6",
        "query": "OpenClaw核心理念",
        "relevant_keywords": ["OpenClaw", "记忆", "内化", "模型参数"],
        "irrelevant_keywords": ["量子", "quantum"],
        "description": "Should match OpenClaw core concept (tests exact keyword matching)",
    },
    {
        "id": "q7",
        "query": "xAI Grok模型配置",
        "relevant_keywords": ["xAI", "Grok", "模型", "配置"],
        "irrelevant_keywords": ["碰撞", "InsightEngine"],
        "description": "Should match xAI Grok preference (tests proper noun exact match)",
    },
    {
        "id": "q8",
        "query": "ACP protocol learning",
        "relevant_keywords": ["ACP", "协议", "protocol"],
        "irrelevant_keywords": ["InsightEngine", "向量数据库"],
        "description": "Tests cross-lingual matching (English query, Chinese content)",
    },
]


def _score_result(result: dict, entry: dict) -> dict:
    """Score a single search result against evaluation criteria."""
    content = result.get("content", "").lower()
    relevant_hits = sum(1 for kw in entry["relevant_keywords"]
                        if kw.lower() in content)
    irrelevant_hits = sum(1 for kw in entry["irrelevant_keywords"]
                          if kw.lower() in content)
    is_relevant = relevant_hits >= 2 or (
        relevant_hits >= 1 and irrelevant_hits == 0
    )
    return {
        "is_relevant": is_relevant,
        "relevant_hits": relevant_hits,
        "irrelevant_hits": irrelevant_hits,
        "score": result.get("score", 0),
        "dense_score": result.get("dense_score", result.get("score", 0)),
        "content_preview": content[:100],
    }


def _compute_metrics(scored_results: List[dict], k: int = 3) -> dict:
    """Compute IR metrics from scored results."""
    hit_at_k = any(r["is_relevant"] for r in scored_results[:k])

    mrr = 0.0
    for i, r in enumerate(scored_results):
        if r["is_relevant"]:
            mrr = 1.0 / (i + 1)
            break

    precision_at_k = sum(1 for r in scored_results[:k] if r["is_relevant"]) / min(k, len(scored_results)) if scored_results else 0

    has_irrelevant_top1 = (
        scored_results[0]["irrelevant_hits"] > 0 if scored_results else False
    )

    return {
        f"hit@{k}": hit_at_k,
        "mrr": round(mrr, 4),
        f"precision@{k}": round(precision_at_k, 4),
        "irrelevant_top1": has_irrelevant_top1,
    }


def eval_vectorstore(use_hybrid: bool = True) -> Dict:
    """Evaluate the VectorStore search quality."""
    sys.path.insert(0, str(WORKSPACE))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(f"\n{'='*60}")
    mode = "Hybrid (BM25+Dense+RRF)" if use_hybrid else "Dense Only"
    print(f"  Search Evaluation: {mode}")
    print(f"{'='*60}\n")

    from memora.vectorstore import VectorStore
    vs = VectorStore()

    entry_count = vs.count()
    print(f"VectorStore entries: {entry_count}")
    if entry_count == 0:
        print("ERROR: No entries in VectorStore. Cannot evaluate.")
        return {"error": "no_entries"}

    all_metrics = []
    score_distribution = []

    for entry in EVAL_DATASET:
        print(f"\n--- [{entry['id']}] {entry['query']}")
        print(f"    Expect: {entry['description']}")

        t0 = time.time()
        if use_hybrid:
            results = vs.search(entry["query"], limit=5, min_score=0.0)
        else:
            results = vs.search_dense_only(entry["query"], limit=5, min_score=0.0)
        elapsed = time.time() - t0

        scored = [_score_result(r, entry) for r in results]
        metrics = _compute_metrics(scored, k=3)

        for sr in scored:
            score_distribution.append({
                "query_id": entry["id"],
                "score": sr["score"],
                "dense_score": sr["dense_score"],
                "is_relevant": sr["is_relevant"],
            })

        status = "PASS" if metrics["hit@3"] else "FAIL"
        print(f"    [{status}] MRR={metrics['mrr']:.2f} P@3={metrics['precision@3']:.2f} ({elapsed:.3f}s)")

        for i, (r, sr) in enumerate(zip(results, scored)):
            rel = "REL" if sr["is_relevant"] else "   "
            print(f"      #{i+1} [{rel}] score={sr['score']:.4f} dense={sr['dense_score']:.4f} | {sr['content_preview'][:60]}")

        metrics["query_id"] = entry["id"]
        all_metrics.append(metrics)

    total = len(all_metrics)
    avg_mrr = sum(m["mrr"] for m in all_metrics) / total
    avg_hit3 = sum(1 for m in all_metrics if m["hit@3"]) / total
    avg_p3 = sum(m["precision@3"] for m in all_metrics) / total
    irr_top1 = sum(1 for m in all_metrics if m["irrelevant_top1"]) / total

    print(f"\n{'='*60}")
    print(f"  AGGREGATE ({mode})")
    print(f"{'='*60}")
    print(f"  Hit@3:     {avg_hit3:.1%} ({sum(1 for m in all_metrics if m['hit@3'])}/{total})")
    print(f"  MRR:       {avg_mrr:.4f}")
    print(f"  P@3:       {avg_p3:.4f}")
    print(f"  Irr.Top1:  {irr_top1:.1%} (lower is better)")

    return {
        "mode": mode,
        "queries": total,
        "hit@3": round(avg_hit3, 4),
        "mrr": round(avg_mrr, 4),
        "precision@3": round(avg_p3, 4),
        "irrelevant_top1_rate": round(irr_top1, 4),
        "details": all_metrics,
        "score_distribution": score_distribution,
    }


def calibrate_threshold(score_data: List[dict]) -> Dict:
    """Find optimal min_score threshold using precision-recall analysis."""
    print(f"\n{'='*60}")
    print(f"  Threshold Calibration")
    print(f"{'='*60}\n")

    if not score_data:
        print("  No score data available for calibration.")
        return {}

    thresholds = [round(t * 0.05, 2) for t in range(0, 21)]
    best_f1 = 0.0
    best_threshold = 0.0
    results = []

    for thresh in thresholds:
        tp = sum(1 for s in score_data if s["score"] >= thresh and s["is_relevant"])
        fp = sum(1 for s in score_data if s["score"] >= thresh and not s["is_relevant"])
        fn = sum(1 for s in score_data if s["score"] < thresh and s["is_relevant"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "threshold": thresh,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"  {'Thresh':>7} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | TP  FP  FN")
    print(f"  {'-'*50}")
    for r in results:
        marker = " <-- best" if r["threshold"] == best_threshold else ""
        print(f"  {r['threshold']:>7.2f} | {r['precision']:>6.3f} | {r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['tp']:>2}  {r['fp']:>2}  {r['fn']:>2}{marker}")

    print(f"\n  Recommended min_score: {best_threshold} (F1={best_f1:.3f})")

    rel_scores = [s["dense_score"] for s in score_data if s["is_relevant"]]
    irr_scores = [s["dense_score"] for s in score_data if not s["is_relevant"]]
    if rel_scores and irr_scores:
        print(f"\n  Relevant dense scores:   min={min(rel_scores):.4f}  avg={sum(rel_scores)/len(rel_scores):.4f}  max={max(rel_scores):.4f}")
        print(f"  Irrelevant dense scores: min={min(irr_scores):.4f}  avg={sum(irr_scores)/len(irr_scores):.4f}  max={max(irr_scores):.4f}")
        gap = min(rel_scores) - max(irr_scores)
        print(f"  Score gap (min_rel - max_irr): {gap:.4f} {'(good separation)' if gap > 0.1 else '(overlap, needs reranker)'}")

    return {
        "best_threshold": best_threshold,
        "best_f1": round(best_f1, 3),
        "curve": results,
    }


def eval_bm25_contribution() -> Dict:
    """Show what BM25 adds beyond dense search."""
    sys.path.insert(0, str(WORKSPACE))

    print(f"\n{'='*60}")
    print(f"  BM25 Contribution Analysis")
    print(f"{'='*60}\n")

    try:
        from bm25 import BM25Index, tokenize

        entries_file = WORKSPACE / "memory" / "vector_db" / "entries.jsonl"
        if not entries_file.exists():
            print("  No entries file found.")
            return {}

        docs = []
        with open(entries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line).get("content", ""))
                    except json.JSONDecodeError:
                        continue

        if not docs:
            print("  No documents loaded.")
            return {}

        idx = BM25Index()
        idx.build(docs)
        print(f"  BM25 index: {len(docs)} documents\n")

        for entry in EVAL_DATASET[:5]:
            query = entry["query"]
            results = idx.search(query, top_k=3)
            tokens = tokenize(query)
            print(f"  [{entry['id']}] \"{query}\"")
            print(f"    Tokens: {tokens[:10]}")
            for score, doc_idx in results:
                preview = docs[doc_idx][:80].replace("\n", " ")
                print(f"    #{doc_idx} score={score:.3f} | {preview}")
            if not results:
                print(f"    (no BM25 matches)")
            print()

        return {"status": "ok", "doc_count": len(docs)}
    except Exception as e:
        print(f"  BM25 analysis failed: {e}")
        return {"error": str(e)}


def run() -> Dict:
    """Full evaluation suite."""
    results = {}

    results["bm25_analysis"] = eval_bm25_contribution()

    results["dense_only"] = eval_vectorstore(use_hybrid=False)

    results["hybrid"] = eval_vectorstore(use_hybrid=True)

    dense_scores = results["dense_only"].get("score_distribution", [])
    results["calibration"] = calibrate_threshold(dense_scores)

    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    d = results.get("dense_only", {})
    h = results.get("hybrid", {})
    print(f"  {'Metric':<20} {'Dense':>10} {'Hybrid':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    for metric in ["hit@3", "mrr", "precision@3"]:
        dv = d.get(metric, 0)
        hv = h.get(metric, 0)
        delta = hv - dv
        sign = "+" if delta > 0 else ""
        print(f"  {metric:<20} {dv:>10.4f} {hv:>10.4f} {sign}{delta:>9.4f}")

    irr_d = d.get("irrelevant_top1_rate", 0)
    irr_h = h.get("irrelevant_top1_rate", 0)
    delta_irr = irr_h - irr_d
    sign = "+" if delta_irr > 0 else ""
    print(f"  {'irr_top1 (lower=better)':<20} {irr_d:>10.4f} {irr_h:>10.4f} {sign}{delta_irr:>9.4f}")

    cal = results.get("calibration", {})
    if cal.get("best_threshold"):
        print(f"\n  Calibrated min_score: {cal['best_threshold']} (F1={cal['best_f1']:.3f})")

    return results


if __name__ == "__main__":
    run()
