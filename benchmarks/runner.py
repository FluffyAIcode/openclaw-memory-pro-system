#!/usr/bin/env python3
"""
Memory Pro System — Benchmark Runner

Usage:
    python3 -m benchmarks.runner                  # run all phases
    python3 -m benchmarks.runner --phase 1        # run specific phase
    python3 -m benchmarks.runner --phase 1 2 4    # run multiple phases
    python3 -m benchmarks.runner --output report.json  # custom output path
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent


def check_server():
    import requests
    try:
        r = requests.get("http://127.0.0.1:18790/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def print_header(text: str):
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {text}")
    print(f"{'=' * w}")


def print_metric(label: str, value, good_threshold=None, bad_threshold=None):
    indicator = ""
    if good_threshold is not None and bad_threshold is not None:
        if isinstance(value, (int, float)):
            if value >= good_threshold:
                indicator = " ✓"
            elif value <= bad_threshold:
                indicator = " ✗"
            else:
                indicator = " ~"
    print(f"  {label:<40} {value}{indicator}")


def print_phase1(result: dict):
    print_header("Phase 1: Recall Quality")
    s = result["summary"]
    print()
    print_metric("Precision@K", f"{s['avg_precision_at_k']:.1%}", 0.6, 0.2)
    print_metric("MRR (Mean Reciprocal Rank)", f"{s['avg_mrr']:.3f}", 0.7, 0.3)
    print_metric("Keyword Hit Rate", f"{s['avg_keyword_hit_rate']:.1%}", 0.6, 0.2)
    print_metric("Topic Coverage", f"{s['avg_topic_coverage']:.1%}", 0.5, 0.1)
    print_metric("Avg Latency", f"{s['avg_latency_ms']:.0f}ms", None, None)
    print_metric("Assembled Recall KW Rate",
                 f"{result['assembled_recall']['avg_keyword_hit_rate']:.1%}", 0.6, 0.2)

    print("\n  By Difficulty:")
    for diff, vals in result.get("by_difficulty", {}).items():
        p = vals.get("precision", 0)
        m = vals.get("mrr", 0)
        k = vals.get("keyword", 0)
        print(f"    {diff:<12} P={p:.1%}  MRR={m:.3f}  KW={k:.1%}")

    print("\n  Per-Query:")
    for d in result["details"]:
        status = "✓" if d["mrr"] > 0 else "✗"
        print(f"    {status} {d['id']} P={d['precision']:.0%} "
              f"MRR={d['mrr']:.2f} KW={d['keyword_hit_rate']:.0%} "
              f"({d['relevant_of_retrieved']}) {d['latency_ms']:.0f}ms")


def print_phase2(result: dict):
    print_header("Phase 2: Faithfulness")
    s = result["summary"]
    print()
    print_metric("Grounding Score", f"{s['avg_grounding_score']:.1%}", 0.7, 0.3)
    print_metric("Hallucination Rate", f"{s['avg_hallucination_rate']:.1%}", None, None)
    print_metric("Trap Avoidance Rate", f"{s['avg_trap_avoidance_rate']:.1%}", 0.8, 0.5)
    print_metric("Claims Verified", s["total_claims_verified"])
    print_metric("Claims Supported", s["total_supported"])
    print_metric("Traps Triggered",
                 f"{s['total_traps_triggered']}/{s['total_traps_checked']}")
    print_metric("LLM Available", f"{result['llm_available']}/{result['dataset_size']}")

    print("\n  Per-Query:")
    for d in result["details"]:
        gs = d["grounding_score"]
        status = "✓" if gs >= 0.7 else ("~" if gs >= 0.4 else "✗")
        llm_tag = "" if d["llm_available"] else " [no-llm]"
        print(f"    {status} {d['id']} G={gs:.0%} H={d['hallucination_rate']:.0%} "
              f"Trap={d['trap_avoidance']:.0%} Claims={d['claims']}{llm_tag}")


def print_phase3(result: dict):
    print_header("Phase 3: A/B Test (No Memory vs With Memory)")
    s = result["summary"]
    print()
    print_metric("With Memory Wins", s["wins_with_memory"])
    print_metric("No Memory Wins", s["wins_no_memory"])
    print_metric("Ties", s["ties"])
    print_metric("Avg Score (no memory)", f"{s['avg_score_no_memory']:.2f}/5")
    print_metric("Avg Score (with memory)", f"{s['avg_score_with_memory']:.2f}/5")
    print_metric("Memory Delta", f"{s['memory_delta']:+.2f}", 0.3, -0.1)
    print_metric("Memory Win Rate", f"{s['memory_win_rate']:.0%}", 0.6, 0.3)

    print("\n  Per-Query:")
    for d in result["details"]:
        w = d["winner"]
        icon = "B✓" if w == "B" else ("A✓" if w == "A" else "==")
        judge_tag = "" if d["llm_judge"] else " [heuristic]"
        print(f"    [{icon}] {d['id']} A={d['avg_no_memory']:.1f} "
              f"B={d['avg_with_memory']:.1f} Δ={d['delta']:+.1f}{judge_tag}")


def print_phase4(result: dict):
    print_header("Phase 4: Context Efficiency")
    s = result["summary"]
    print()
    print_metric("Raw Memory Dump", f"{s['raw_memory_tokens']} tokens")
    print_metric("Session Context", f"{s['session_context_tokens']} tokens")
    print_metric("Avg Recall (per query)", f"{s['avg_recall_tokens']} tokens")
    print_metric("Avg Search (per query)", f"{s['avg_search_tokens']} tokens")
    print_metric("Briefing", f"{s['briefing_tokens']} tokens")
    print_metric("KG Total", f"{s['kg_total_tokens']} tokens")
    print_metric("Skills Total", f"{s['skills_tokens']} tokens")
    print_metric("Compression (session)", f"{s['compression_ratio_session']:.0%}")
    print_metric("Compression (recall)", f"{s['compression_ratio_recall']:.0%}")
    print_metric("Recall Latency", f"{s['avg_recall_latency_ms']:.0f}ms")
    print_metric("Search Latency", f"{s['avg_search_latency_ms']:.0f}ms")

    budget = result["token_budget_comparison"]
    print("\n  Token Budget Comparison (for same information need):")
    print(f"    Raw dump all:       {budget['raw_dump_all']:>6} tokens")
    print(f"    Session context:    {budget['session_context']:>6} tokens")
    print(f"    Assembled recall:   {budget['assembled_recall_per_query']:>6} tokens/query")
    print(f"    Simple search:      {budget['search_per_query']:>6} tokens/query")
    print(f"    Briefing:           {budget['briefing']:>6} tokens")


def print_summary(all_results: dict):
    print_header("OVERALL BENCHMARK SUMMARY")
    print()

    p1 = all_results.get("phase_1")
    p2 = all_results.get("phase_2")
    p3 = all_results.get("phase_3")
    p4 = all_results.get("phase_4")

    grades = []

    if p1:
        mrr = p1["summary"]["avg_mrr"]
        grade = "A" if mrr >= 0.8 else ("B" if mrr >= 0.6 else ("C" if mrr >= 0.4 else "D"))
        grades.append(("Recall Quality", grade, f"MRR={mrr:.3f}"))

    if p2:
        gs = p2["summary"]["avg_grounding_score"]
        grade = "A" if gs >= 0.8 else ("B" if gs >= 0.6 else ("C" if gs >= 0.4 else "D"))
        grades.append(("Faithfulness", grade, f"Grounding={gs:.0%}"))

    if p3:
        wr = p3["summary"]["memory_win_rate"]
        grade = "A" if wr >= 0.8 else ("B" if wr >= 0.6 else ("C" if wr >= 0.4 else "D"))
        grades.append(("Memory Impact", grade, f"WinRate={wr:.0%}"))

    if p4:
        comp = p4["summary"]["compression_ratio_recall"]
        grade = "A" if comp >= 0.9 else ("B" if comp >= 0.7 else ("C" if comp >= 0.5 else "D"))
        grades.append(("Context Efficiency", grade, f"Compression={comp:.0%}"))

    for label, grade, detail in grades:
        print(f"  [{grade}] {label:<25} {detail}")

    print()
    overall_grades = [g[1] for g in grades]
    if overall_grades:
        score = sum({"A": 4, "B": 3, "C": 2, "D": 1}.get(g, 0) for g in overall_grades)
        avg = score / len(overall_grades)
        overall = "A" if avg >= 3.5 else ("B" if avg >= 2.5 else ("C" if avg >= 1.5 else "D"))
        print(f"  Overall Grade: [{overall}]")


def main():
    parser = argparse.ArgumentParser(description="Memory Pro System Benchmark Runner")
    parser.add_argument("--phase", nargs="*", type=int, default=None,
                        help="Phase(s) to run (1-4). Default: all.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON report path.")
    args = parser.parse_args()

    phases = args.phase or [1, 2, 3, 4]

    health = check_server()
    if not health:
        print("ERROR: Memory server is offline.")
        print("Start it with: memory-cli server-start")
        sys.exit(1)

    print(f"Memory Pro System Benchmark Suite")
    print(f"Server: OK (uptime {health['uptime_seconds']}s, "
          f"embedder={health['embedder']})")
    print(f"Phases: {phases}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "server": health,
    }

    total_t0 = time.time()

    if 1 in phases:
        print("\nRunning Phase 1: Recall Quality...")
        from benchmarks.phase1_recall import run as run_p1
        r1 = run_p1()
        all_results["phase_1"] = r1
        print_phase1(r1)

    if 2 in phases:
        print("\nRunning Phase 2: Faithfulness...")
        from benchmarks.phase2_faithfulness import run as run_p2
        r2 = run_p2()
        all_results["phase_2"] = r2
        print_phase2(r2)

    if 3 in phases:
        print("\nRunning Phase 3: A/B Test...")
        from benchmarks.phase3_ab_test import run as run_p3
        r3 = run_p3()
        all_results["phase_3"] = r3
        print_phase3(r3)

    if 4 in phases:
        print("\nRunning Phase 4: Context Efficiency...")
        from benchmarks.phase4_context_efficiency import run as run_p4
        r4 = run_p4()
        all_results["phase_4"] = r4
        print_phase4(r4)

    total_elapsed = time.time() - total_t0
    all_results["total_elapsed_seconds"] = round(total_elapsed, 1)

    print_summary(all_results)
    print(f"\n  Total time: {total_elapsed:.1f}s")

    output_path = args.output or str(
        WORKSPACE / "benchmarks" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Report saved: {output_path}")


if __name__ == "__main__":
    main()
