"""
LongMemEval Benchmark — tests Memory Pro against the ICLR 2025 long-term memory benchmark.

Uses the Oracle dataset (evidence-only sessions) to evaluate retrieval + QA quality
across 5 core abilities: Information Extraction, Multi-Session Reasoning,
Knowledge Updates, Temporal Reasoning, and Abstention.

Strategy: direct Python vector-store writes for ingestion (bypasses KG hooks),
HTTP /recall for retrieval (tests the real pipeline), LLM for answer + judge.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).resolve().parent.parent
_SERVER = "http://127.0.0.1:18790"
_AUTH_TOKEN: Optional[str] = None
_SOURCE_TAG = "longmemeval-bench"
_JUDGE_MODEL = "deepseek/deepseek-chat"
_ANSWER_MODEL = "deepseek/deepseek-chat"


def _load_auth_token() -> Optional[str]:
    global _AUTH_TOKEN
    if _AUTH_TOKEN:
        return _AUTH_TOKEN
    token_path = _WORKSPACE / "memory" / "security" / ".auth_token"
    if token_path.exists():
        _AUTH_TOKEN = token_path.read_text().strip()
    return _AUTH_TOKEN


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    token = _load_auth_token()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _post(path: str, body: dict, timeout: int = 30) -> Optional[dict]:
    req = Request(f"{_SERVER}{path}", data=json.dumps(body).encode(),
                  headers=_headers(), method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning("POST %s failed: %s", path, e)
        return None


def _get(path: str, timeout: int = 10) -> Optional[dict]:
    req = Request(f"{_SERVER}{path}", headers=_headers())
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM judge prompt templates (adapted from LongMemEval evaluate_qa.py)
# ---------------------------------------------------------------------------

_JUDGE_STANDARD = """Question: {question}
Gold answer: {gold}
Model answer: {hypothesis}
Is the model's answer correct? Answer only "yes" or "no"."""

_JUDGE_TEMPORAL = """Question: {question}
Gold answer: {gold}
Model answer: {hypothesis}
For temporal reasoning, off-by-one errors are acceptable.
Is the model's answer correct? Answer only "yes" or "no"."""

_JUDGE_UPDATE = """Question: {question}
Gold answer (UPDATED value): {gold}
Model answer: {hypothesis}
The system must return the latest/updated value.
Is the model's answer correct? Answer only "yes" or "no"."""

_JUDGE_PREFERENCE = """Question: {question}
Rubric: {gold}
Model answer: {hypothesis}
Does the answer match the user's preference/info in the rubric?
Answer only "yes" or "no"."""

_JUDGE_ABSTENTION = """Question: {question}
Note: This was NEVER discussed with the user.
Model answer: {hypothesis}
Did the model correctly refuse or say it doesn't know?
Answer only "yes" or "no"."""


def _get_judge_prompt(q_type: str, q_id: str, question: str, gold: str, hypothesis: str) -> str:
    is_abs = q_id.endswith("_abs")
    if is_abs:
        return _JUDGE_ABSTENTION.format(question=question, hypothesis=hypothesis)
    if q_type == "temporal-reasoning":
        return _JUDGE_TEMPORAL.format(question=question, gold=gold, hypothesis=hypothesis)
    if q_type == "knowledge-update":
        return _JUDGE_UPDATE.format(question=question, gold=gold, hypothesis=hypothesis)
    if q_type == "single-session-preference":
        return _JUDGE_PREFERENCE.format(question=question, gold=gold, hypothesis=hypothesis)
    return _JUDGE_STANDARD.format(question=question, gold=gold, hypothesis=hypothesis)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _llm_call(prompt: str, system: str = "", max_tokens: int = 256,
              model: str = "") -> Optional[str]:
    sys.path.insert(0, str(_WORKSPACE))
    from llm_client import generate
    return generate(prompt, system=system, model=model or _ANSWER_MODEL,
                    max_tokens=max_tokens, temperature=0.0, timeout=45)


def generate_answer(question: str, context: str, q_date: str) -> Optional[str]:
    system_prompt = (
        "You are a personal assistant with access to the user's long-term memory.\n"
        "IMPORTANT RULES:\n"
        "- For \"how many\" questions: count ALL distinct instances across ALL memories\n"
        "- For questions about current state: use the MOST RECENT memory (check dates)\n"
        "- For preference questions: look for expressions of like/dislike/preference\n"
        "- If memories conflict, state the most recent one and note the change\n"
        "- Always ground your answer in the provided memories\n"
        "- If the context lacks relevant info, say you don't know"
    )
    prompt = (
        f"Today's date: {q_date}\n\n"
        f"Memory Context (sorted by date, newest first):\n{context[:2000]}\n\n"
        f"Question: {question}\n"
        f"Answer (be specific and cite memory dates when relevant):"
    )
    return _llm_call(prompt, system=system_prompt, max_tokens=200)


def judge_answer(q_type: str, q_id: str, question: str, gold: str, hypothesis: str) -> bool:
    prompt = _get_judge_prompt(q_type, q_id, question, gold, hypothesis)
    result = _llm_call(prompt, system="Answer only 'yes' or 'no'.",
                       max_tokens=5, model=_JUDGE_MODEL)
    if result is None:
        return False
    return "yes" in result.lower()


# ---------------------------------------------------------------------------
# Ingestion via HTTP /add (uses force_systems=["memora"] — skips MSA/KG writes)
# ---------------------------------------------------------------------------

def ingest_sessions(sessions: list, dates: list) -> int:
    """Ingest evidence sessions via HTTP /add (vector store only, no KG overhead)."""
    count = 0
    for i, session in enumerate(sessions):
        date_str = dates[i] if i < len(dates) else ""
        parts = []
        for turn in session:
            content = turn.get("content", "").strip()
            if not content:
                continue
            role = turn.get("role", "user")
            parts.append(f"[{role}]: {content}")
        if not parts:
            continue
        block = f"[{date_str}]\n" + "\n".join(parts) if date_str else "\n".join(parts)
        resp = _post("/add", {
            "content": block[:2000],
            "source": _SOURCE_TAG,
            "importance": 0.6,
            "skip_hooks": True,
        }, timeout=15)
        if resp:
            count += 1
    return count


def clean_test_entries() -> int:
    """Remove tagged entries from disk, then ask server to reload vectorstore."""
    entries_path = _WORKSPACE / "memory" / "vector_db" / "entries.jsonl"
    if not entries_path.exists():
        return 0
    lines = entries_path.read_text().splitlines()
    keep = []
    removed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("metadata", {}).get("source") == _SOURCE_TAG:
                removed += 1
                continue
        except json.JSONDecodeError:
            pass
        keep.append(line)
    entries_path.write_text("\n".join(keep) + ("\n" if keep else ""))
    if removed:
        _get("/vectorstore/reload", timeout=10)
    return removed


# ---------------------------------------------------------------------------
# Recall (through HTTP to test the real pipeline)
# ---------------------------------------------------------------------------

def recall_context(question: str) -> dict:
    return _post("/recall", {"query": question, "top_k": 10, "max_tokens": 4000}, timeout=30) or {}


def build_context_text(recall_resp: dict) -> str:
    parts = []
    for skill in recall_resp.get("skills", []):
        parts.append(f"[Skill] {skill.get('content', skill.get('name', ''))}")
    for rel in recall_resp.get("kg_relations", []):
        parts.append(f"[KG] {rel}")
    for item in recall_resp.get("merged", []):
        text = item if isinstance(item, str) else item.get("content", item.get("text", str(item)))
        parts.append(text)
    if not parts:
        for item in recall_resp.get("evidence", recall_resp.get("memora", [])):
            text = item if isinstance(item, str) else item.get("content", item.get("text", str(item)))
            parts.append(text)
    return "\n\n".join(parts) if parts else "(no relevant memory found)"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(data_path: str, max_questions: int = 0, resume_from: int = 0) -> dict:
    health = _get("/health")
    if not health:
        logger.error("Memory server not reachable at %s", _SERVER)
        sys.exit(1)
    logger.info("Server healthy: %s", health)

    sys.path.insert(0, str(_WORKSPACE))
    from llm_client import is_available, get_provider_info
    if not is_available():
        logger.error("No LLM API key available")
        sys.exit(1)
    provider_info = get_provider_info()
    logger.info("LLM provider: %s", provider_info)

    with open(data_path) as f:
        dataset = json.load(f)

    if max_questions > 0:
        dataset = dataset[:max_questions]

    total = len(dataset)
    logger.info("Running LongMemEval benchmark: %d questions (resume_from=%d)", total, resume_from)

    # Load partial results if resuming
    partial_path = _WORKSPACE / "benchmarks" / "_partial_results.json"
    results = []
    if resume_from > 0 and partial_path.exists():
        results = json.loads(partial_path.read_text())
        logger.info("Loaded %d partial results", len(results))

    type_correct: Dict[str, List[bool]] = {}
    abs_correct: List[bool] = []
    total_recall_ms = 0.0
    total_ingest_turns = 0
    layer_counts = {"skills": 0, "kg_relations": 0, "merged": 0}
    errors = 0

    # Recompute stats from loaded results
    for r in results:
        type_correct.setdefault(r["question_type"], []).append(r["correct"])
        if r.get("is_abstention"):
            abs_correct.append(r["correct"])
        total_recall_ms += r.get("recall_ms", 0)
        total_ingest_turns += r.get("ingested_turns", 0)
        if not r.get("correct") and r.get("hypothesis", "").startswith("("):
            errors += 1

    start_time = time.time()

    for idx in range(resume_from, total):
        q = dataset[idx]
        qid = q["question_id"]
        qtype = q["question_type"]
        question = q["question"]
        gold = q["answer"]
        q_date = q.get("question_date", "")
        is_abs = qid.endswith("_abs")

        logger.info("[%d/%d] %s (%s%s)", idx + 1, total, qid, qtype,
                    " ABS" if is_abs else "")

        # 1. Ingest evidence sessions (direct vector store write)
        n_ingested = ingest_sessions(q["haystack_sessions"], q.get("haystack_dates", []))
        total_ingest_turns += n_ingested

        # 2. Recall via HTTP
        t0 = time.time()
        recall_resp = recall_context(question)
        recall_ms = (time.time() - t0) * 1000
        total_recall_ms += recall_ms

        context_text = build_context_text(recall_resp)
        layer_counts["skills"] += len(recall_resp.get("skills", []))
        layer_counts["kg_relations"] += len(recall_resp.get("kg_relations", []))
        layer_counts["merged"] += len(recall_resp.get("merged", []))

        # 3. Generate answer
        hypothesis = generate_answer(question, context_text, q_date)
        if hypothesis is None:
            hypothesis = "(LLM generation failed)"
            errors += 1

        # 4. Judge
        correct = judge_answer(qtype, qid, question, gold, hypothesis)

        type_correct.setdefault(qtype, []).append(correct)
        if is_abs:
            abs_correct.append(correct)

        results.append({
            "question_id": qid,
            "question_type": qtype,
            "is_abstention": is_abs,
            "question": question,
            "gold_answer": gold,
            "hypothesis": hypothesis,
            "correct": correct,
            "recall_ms": round(recall_ms, 1),
            "context_length": len(context_text),
            "ingested_turns": n_ingested,
        })

        # 5. Clean up this question's data
        clean_test_entries()

        # Save partial results every 25 questions
        if (idx + 1) % 25 == 0:
            partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
            elapsed = time.time() - start_time
            done = idx + 1 - resume_from
            pct = sum(r["correct"] for r in results) / len(results) * 100
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / speed if speed > 0 else 0
            logger.info("Progress: %d/%d done, %.1f%% correct, %.0fs elapsed, ETA %.0fs",
                        idx + 1, total, pct, elapsed, eta)

    elapsed = time.time() - start_time

    # Compute metrics
    n_results = len(results)
    overall_acc = sum(r["correct"] for r in results) / max(n_results, 1)
    type_acc = {}
    for t, bools in type_correct.items():
        type_acc[t] = sum(bools) / max(len(bools), 1)
    abs_acc = sum(abs_correct) / max(len(abs_correct), 1) if abs_correct else None

    ability_map = {
        "Information Extraction": ["single-session-user", "single-session-assistant",
                                   "single-session-preference"],
        "Multi-Session Reasoning": ["multi-session"],
        "Knowledge Update": ["knowledge-update"],
        "Temporal Reasoning": ["temporal-reasoning"],
    }
    ability_acc = {}
    for ability, types in ability_map.items():
        bools = []
        for t in types:
            bools.extend(type_correct.get(t, []))
        if bools:
            ability_acc[ability] = sum(bools) / len(bools)

    report = {
        "benchmark": "LongMemEval (Oracle)",
        "timestamp": datetime.now().isoformat(),
        "dataset_size": total,
        "questions_evaluated": n_results,
        "elapsed_seconds": round(elapsed, 1),
        "llm_answer_model": _ANSWER_MODEL,
        "llm_judge_model": _JUDGE_MODEL,
        "llm_provider": provider_info,
        "errors": errors,
        "metrics": {
            "overall_accuracy": round(overall_acc, 4),
            "ability_accuracy": {k: round(v, 4) for k, v in ability_acc.items()},
            "type_accuracy": {k: round(v, 4) for k, v in type_acc.items()},
            "abstention_accuracy": round(abs_acc, 4) if abs_acc is not None else None,
            "avg_recall_latency_ms": round(total_recall_ms / max(n_results, 1), 1),
            "avg_ingested_sessions": round(total_ingest_turns / max(n_results, 1), 1),
            "layer_totals": layer_counts,
        },
        "results": results,
    }

    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()

    return report


def print_report(report: dict):
    m = report["metrics"]
    print("\n" + "=" * 70)
    print("  LongMemEval Benchmark Report — Memory Pro System v0.0.10")
    print("=" * 70)
    print(f"  Dataset:    {report['benchmark']} ({report['dataset_size']} questions)")
    print(f"  Evaluated:  {report['questions_evaluated']} questions")
    print(f"  Duration:   {report['elapsed_seconds']}s")
    print(f"  Answer LLM: {report.get('llm_answer_model', '?')}")
    print(f"  Judge LLM:  {report.get('llm_judge_model', '?')}")
    print(f"  Errors:     {report['errors']}")
    print()
    print(f"  Overall Accuracy:       {m['overall_accuracy']:.1%}")
    print()
    print("  Ability Breakdown:")
    for ability, acc in m.get("ability_accuracy", {}).items():
        print(f"    {ability:30s} {acc:.1%}")
    if m.get("abstention_accuracy") is not None:
        print(f"    {'Abstention':30s} {m['abstention_accuracy']:.1%}")
    print()
    print("  Per-Type Breakdown:")
    for t, acc in m.get("type_accuracy", {}).items():
        print(f"    {t:35s} {acc:.1%}")
    print()
    print("  Memory Pro Metrics:")
    print(f"    Avg Recall Latency:    {m['avg_recall_latency_ms']:.0f} ms")
    print(f"    Avg Ingested Sess/Q:   {m['avg_ingested_sessions']:.1f}")
    lt = m.get("layer_totals", {})
    print(f"    Total Skills hits:     {lt.get('skills', 0)}")
    print(f"    Total KG Relations:    {lt.get('kg_relations', 0)}")
    print(f"    Total Merged items:    {lt.get('merged', 0)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for Memory Pro")
    parser.add_argument("data", nargs="?",
                        default="/tmp/longmemeval/longmemeval_oracle.json",
                        help="Path to longmemeval_oracle.json")
    parser.add_argument("--max", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--resume", type=int, default=0, help="Resume from question index")
    parser.add_argument("--output", type=str, default="", help="Output JSON report path")
    args = parser.parse_args()

    report = run_benchmark(args.data, args.max, args.resume)

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(_WORKSPACE / "benchmarks" / f"longmemeval_report_{ts}.json")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\n  Report saved: {args.output}\n")


if __name__ == "__main__":
    main()
