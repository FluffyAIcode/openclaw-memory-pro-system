#!/usr/bin/env python3
"""
memory-cli — Thin HTTP client for the Memory Server.

No heavy dependencies (no SentenceTransformer, no numpy at import time).
Just sends JSON requests to localhost:18790 and prints results.

Usage:
    memory-cli remember "content" [-i 0.8] [-s source]
    memory-cli recall "query" [-k 8]
    memory-cli deep-recall "query" [-r 3]
    memory-cli search "query" [-k 8]
    memory-cli add "content" [-i 0.8] [-s source]
    memory-cli digest [--days 7]
    memory-cli status
    memory-cli health
    memory-cli server-start
    memory-cli server-stop
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_BASE_URL = "http://127.0.0.1:18790"
_WORKSPACE = Path(__file__).parent


def _post(path: str, data: dict, timeout: int = 120) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(
        f"{_BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"Error: Memory Server not reachable ({e})", file=sys.stderr)
        print("Start it with: memory-cli server-start", file=sys.stderr)
        sys.exit(1)


def _get(path: str, timeout: int = 30) -> dict:
    req = Request(f"{_BASE_URL}{path}", method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"Error: Memory Server not reachable ({e})", file=sys.stderr)
        print("Start it with: memory-cli server-start", file=sys.stderr)
        sys.exit(1)


def _post_async(path: str, data: dict, label: str = "") -> dict:
    """Submit a slow request asynchronously and poll until done."""
    import time as _time

    data["async"] = True
    resp = _post(path, data, timeout=10)

    task_id = resp.get("task_id")
    if not task_id:
        return resp

    tag = label or path
    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0
    poll_interval = 2.0

    while True:
        _time.sleep(poll_interval)
        task = _get(f"/task/{task_id}", timeout=10)
        status = task.get("status", "unknown")
        elapsed = task.get("elapsed", 0)

        frame = _SPINNER[idx % len(_SPINNER)]
        sys.stdout.write(f"\r{frame} {tag} ... {elapsed:.0f}s [{status}]   ")
        sys.stdout.flush()
        idx += 1

        if status == "done":
            sys.stdout.write(f"\r✓ {tag} 完成 ({elapsed:.0f}s)            \n")
            sys.stdout.flush()
            return task.get("result", {})
        elif status == "error":
            sys.stdout.write(f"\r✗ {tag} 失败 ({elapsed:.0f}s)            \n")
            sys.stdout.flush()
            print(f"  Error: {task.get('error', 'unknown')}", file=sys.stderr)
            sys.exit(1)

        if poll_interval < 5.0:
            poll_interval = min(poll_interval + 0.5, 5.0)


def cmd_remember(args):
    result = _post("/remember", {
        "content": args.content,
        "source": args.source,
        "importance": args.importance,
        "doc_id": args.doc_id,
        "title": args.title,
    })
    systems = result.get("systems_used", [])
    print(f"Remembered ({result.get('word_count', '?')} words) via: {', '.join(systems)}")


def cmd_recall(args):
    result = _post("/recall", {"query": args.query, "top_k": args.top_k})
    merged = result.get("merged", [])
    if not merged:
        print("No results found.")
        return
    for i, r in enumerate(merged, 1):
        sys_tag = r.get("system", r.get("metadata", {}).get("source", "?"))
        score = r.get("score", 0)
        content = r.get("content", "")[:200]
        print(f"  [{i}] (score={score:.4f}, system={sys_tag})")
        print(f"      {content}")
        print()


def cmd_deep_recall(args):
    result = _post_async("/deep-recall", {"query": args.query, "max_rounds": args.max_rounds},
                         label="深度召回")
    interleave = result.get("interleave")
    if interleave:
        print(f"Multi-hop reasoning: {interleave.get('rounds', '?')} rounds, "
              f"{interleave.get('total_docs_used', '?')} documents used")
        answer = interleave.get("final_answer", "")
        if answer:
            print(f"\n{answer[:2000]}")
    else:
        print("MSA interleave not available.")

    ctx = result.get("memora_context", [])
    if ctx:
        print(f"\n--- Memora context ({len(ctx)} snippets) ---")
        for r in ctx[:3]:
            print(f"  [{r.get('score', 0):.4f}] {r.get('content', '')[:150]}")


def cmd_search(args):
    result = _post("/search", {"query": args.query, "limit": args.top_k})
    entries = result.get("results", [])
    if not entries:
        print("No results found.")
        return
    for i, r in enumerate(entries, 1):
        print(f"  [{i}] (score={r.get('score', 0):.4f}) {r.get('content', '')[:200]}")


def cmd_add(args):
    result = _post("/add", {
        "content": args.content,
        "source": args.source,
        "importance": args.importance,
    })
    print(f"Added: {result.get('timestamp', '?')} [{result.get('source', '?')}]")


def cmd_digest(args):
    result = _post_async("/digest", {"days": args.days}, label="记忆提炼")
    if result.get("success"):
        print(f"Digest complete (last {args.days} days)")
    else:
        print("Digest failed")


def cmd_status(args):
    result = _get("/status")
    server = result.get("server", {})
    print(f"Memory Server: uptime={server.get('uptime_seconds', '?')}s, "
          f"embedder={server.get('embedder', '?')}")
    print()
    for name, info in result.get("systems", {}).items():
        if isinstance(info, dict) and "error" not in info:
            print(f"  {name}: {json.dumps(info, ensure_ascii=False)}")
        elif isinstance(info, dict):
            print(f"  {name}: ERROR — {info.get('error', '?')}")
        else:
            print(f"  {name}: {info}")


def cmd_health(args):
    result = _get("/health")
    status = result.get("status", "unknown")
    uptime = result.get("uptime_seconds", "?")
    embedder = result.get("embedder", "?")
    pid = result.get("pid", "?")
    print(f"Status: {status} | PID: {pid} | Uptime: {uptime}s | Embedder: {embedder}")


def cmd_server_start(args):
    pid_file = _WORKSPACE / "memory" / "server.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        try:
            os.kill(int(pid), 0)
            print(f"Memory Server already running (PID {pid})")
            return
        except (OSError, ValueError):
            pid_file.unlink()

    server_path = _WORKSPACE / "memory_server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path), "--daemon"],
        cwd=str(_WORKSPACE),
    )
    proc.wait()
    print("Memory Server starting... use 'memory-cli health' to check.")


def cmd_collide(args):
    result = _post_async("/second-brain/collide", {}, label="灵感碰撞")
    insights = result.get("insights", [])
    msg = result.get("message", "")
    print(msg)
    for i, ins in enumerate(insights, 1):
        print(f"\n  [{i}] [{ins.get('strategy', '?')}] 新颖度={ins.get('novelty', 0)}")
        print(f"      联系: {ins.get('connection', '')[:150]}")
        print(f"      灵感: {ins.get('ideas', '')[:150]}")


def cmd_deep_collide(args):
    result = _post_async("/second-brain/deep-collide",
                         {"topic": args.topic if hasattr(args, 'topic') else ""},
                         label="深度碰撞")
    answer = result.get("answer")
    if answer:
        print(f"深度碰撞 ({result.get('docs_used', 0)} 篇文档, {result.get('rounds', 0)} 轮推理):")
        print(f"\n{answer[:2000]}")
    else:
        print(result.get("message", "深度碰撞未产生结果"))


def cmd_sb_report(args):
    result = _get("/second-brain/report")
    print(f"追踪统计: 总访问 {result.get('tracker_stats', {}).get('total_accesses', 0)} 次, "
          f"独立记忆 {result.get('tracker_stats', {}).get('unique_memories', 0)} 条")
    print(f"平均活力值: {result.get('avg_vitality', 0):.4f} (样本 {result.get('vitality_sample_size', 0)})")
    print(f"沉睡记忆: {result.get('dormant_count', 0)} 条")
    dormant = result.get("dormant_top3", [])
    for d in dormant:
        print(f"  - [{d.get('dormant_days', '?')}天未访问] (重要性={d.get('importance', 0):.2f}) "
              f"{d.get('content', '')[:80]}")
    trends = result.get("trending", [])
    if trends:
        print(f"近期趋势 (最近{3}天):")
        for t in trends:
            print(f"  - 命中 {t.get('hits', 0)} 次: {', '.join(t.get('queries', [])[:3])}")
    rins = result.get("recent_insights", [])
    if rins:
        print(f"最近灵感: {result.get('recent_insights_count', 0)} 条")
        for r in rins:
            print(f"  - {r.get('date', '?')}: {r.get('insight_count', 0)} 条碰撞")


def cmd_sb_status(args):
    result = _get("/second-brain/status")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_tasks(args):
    result = _get("/tasks")
    tasks = result.get("tasks", [])
    if not tasks:
        print("No active tasks.")
        return
    for t in tasks:
        status = t.get("status", "?")
        name = t.get("name", "?")
        elapsed = t.get("elapsed", 0)
        tid = t.get("id", "?")
        icon = {"running": "⟳", "done": "✓", "error": "✗"}.get(status, "?")
        print(f"  {icon} [{tid}] {name} — {status} ({elapsed:.0f}s)")
        if status == "error":
            print(f"    Error: {t.get('error', '')[:100]}")


def cmd_briefing(args):
    result = _get("/briefing")
    print(result.get("text", "无法生成简报"))
    dist = result.get("vitality_distribution", {})
    if dist:
        print(f"\n活力分布: 高={dist.get('high', 0)} 中={dist.get('medium', 0)} 低={dist.get('low', 0)}")


def cmd_vitality(args):
    result = _get("/vitality")
    total = result.get("total", 0)
    dist = result.get("distribution", {})
    high = dist.get("high", 0)
    medium = dist.get("medium", 0)
    low = dist.get("low", 0)

    bar_h = "█" * min(high, 40)
    bar_m = "█" * min(medium, 40)
    bar_l = "█" * min(low, 40)

    print(f"记忆活力分布 (共 {total} 条):")
    print(f"  {bar_h}  高活力 (>0.7): {high} 条")
    print(f"  {bar_m}  中活力 (0.4-0.7): {medium} 条")
    print(f"  {bar_l}  低活力 (<0.4): {low} 条")

    top = result.get("top_active", [])
    if top:
        print("\n最活跃:")
        for t in top[:5]:
            print(f"  [{t['vitality']:.2f}] {t['content'][:80]}")

    near = result.get("nearly_dormant", [])
    if near:
        print("\n即将沉睡:")
        for n in near[:5]:
            print(f"  [{n['vitality']:.2f}] ({n.get('timestamp', '?')}) {n['content'][:80]}")


def cmd_inspect(args):
    result = _post("/inspect", {"query": args.query})
    matches = result.get("matches", [])
    if not matches:
        print("未找到匹配的记忆。")
        return
    for i, m in enumerate(matches, 1):
        print(f"\n  [{i}] (相似度={m.get('score', 0):.4f})")
        print(f"      内容: {m.get('content', '')[:200]}")
        print(f"      来源: {m.get('source', '?')} | 时间: {m.get('timestamp', '?')}")
        print(f"      重要性: {m.get('importance', 0):.2f} | 活力: {m.get('vitality', 0):.2f}")
        print(f"      被召回: {m.get('access_count', 0)} 次 | 最后访问: {m.get('last_accessed', '从未')}")
        insights = m.get("related_insights", [])
        if insights:
            print(f"      关联灵感: {', '.join(i.get('date', '') for i in insights)}")


def cmd_review_dormant(args):
    result = _get("/dormant")
    memories = result.get("memories", [])
    count = result.get("count", 0)
    never_count = result.get("never_accessed_count", 0)
    threshold = result.get("threshold_days", 14)

    if not memories and not never_count:
        print(f"没有沉睡记忆（阈值: {threshold} 天未访问 + 重要性 >= {result.get('threshold_importance', 0.7)}）")
        return

    if memories:
        print(f"🔇 {count} 条真正沉睡的记忆（曾被访问，但超过 {threshold} 天未再访问）:\n")
        for i, m in enumerate(memories, 1):
            days = m.get("dormant_days", 0)
            imp = m.get("importance", 0)
            layer = m.get("layer", "?")
            content = m.get("content", "")[:120]
            print(f"  [{i}] {days}天未访问 | 重要性={imp:.2f} | 来源={layer}")
            print(f"      {content}")
            print()
    else:
        print("没有真正沉睡的记忆（所有曾访问过的记忆都在活跃状态）。")

    never = result.get("never_accessed", [])
    if never:
        print(f"\n📭 另有 {never_count} 条重要记忆从未被召回（最早的 {len(never)} 条）:\n")
        for i, m in enumerate(never, 1):
            days = m.get("dormant_days", 0)
            imp = m.get("importance", 0)
            content = m.get("content", "")[:100]
            print(f"  [{i}] 创建 {days} 天 | 重要性={imp:.2f}")
            print(f"      {content}")
            print()

    print("提示: 用 memory-cli recall \"关键词\" 可唤醒相关记忆")


def cmd_contradictions(args):
    result = _get("/contradictions")
    count = result.get("count", 0)
    reports = result.get("reports", [])
    if not reports:
        print("✅ 知识图谱中没有发现矛盾决策")
        return
    print(f"⚠️ 发现 {count} 个存在矛盾证据的决策:\n")
    for i, r in enumerate(reports, 1):
        risk = r.get("risk_score", 0)
        decision = r.get("decision_content", "")[:120]
        sc = r.get("supporting_count", 0)
        cc = r.get("contradicting_count", 0)
        print(f"  [{i}] 风险={risk:.1%} | 支撑={sc} | 矛盾={cc}")
        print(f"      决策: {decision}")
        for c in r.get("contradicting", [])[:2]:
            print(f"      ❌ {c.get('content', '')[:100]}")
            if c.get("evidence"):
                print(f"         理由: {c['evidence']}")
        print()


def cmd_blindspots(args):
    result = _get("/blindspots")
    count = result.get("count", 0)
    reports = result.get("reports", [])
    if not reports:
        print("✅ 没有发现明显的认知盲区")
        return
    print(f"🔍 发现 {count} 个决策存在未考虑的维度:\n")
    for i, r in enumerate(reports, 1):
        decision = r.get("decision_content", "")[:120]
        missing = r.get("missing", [])
        coverage = r.get("coverage_ratio", 0)
        print(f"  [{i}] 覆盖率={coverage:.0%}")
        print(f"      决策: {decision}")
        print(f"      ⚠️ 未考虑: {', '.join(missing)}")
        print()


def cmd_threads(args):
    result = _get("/threads")
    count = result.get("count", 0)
    threads = result.get("threads", [])
    if not threads:
        print("知识图谱中暂无思维线索（需要更多记忆来形成线索）")
        return
    print(f"🧵 发现 {count} 条思维线索:\n")
    for i, t in enumerate(threads, 1):
        title = t.get("title", "未命名")
        nc = t.get("node_count", 0)
        status = t.get("status", "?")
        dtype = t.get("dominant_type", "?")
        status_icon = {"exploring": "🔎", "decided": "✅", "nascent": "🌱",
                       "developing": "📈"}.get(status, "❓")
        print(f"  [{i}] {status_icon} {title} ({nc} 个节点, 状态={status}, 主类型={dtype})")


def cmd_kg_status(args):
    result = _get("/kg/status")
    nodes = result.get("total_nodes", 0)
    edges = result.get("total_edges", 0)
    components = result.get("connected_components", 0)
    print(f"📊 知识图谱状态:")
    print(f"   节点: {nodes}")
    print(f"   边:   {edges}")
    print(f"   连通分量: {components}")
    nt = result.get("node_types", {})
    if nt:
        print(f"   节点类型: {', '.join(f'{k}={v}' for k, v in nt.items())}")
    et = result.get("edge_types", {})
    if et:
        print(f"   边类型:   {', '.join(f'{k}={v}' for k, v in et.items())}")


def cmd_rate(args):
    result = _post("/insight/rate", {
        "insight_id": args.insight_id,
        "strategy": args.strategy,
        "rating": args.rating,
        "comment": args.comment,
    })
    strategy = result.get("strategy", "?")
    new_weight = result.get("new_weight", 0)
    rating = result.get("rating", 0)
    print(f"✅ 评分已记录: 策略={strategy}, 评分={rating}, 新权重={new_weight:.2f}")


def cmd_insight_stats(args):
    result = _get("/insight/stats")
    weights = result.get("weights", {})
    total = result.get("total_ratings", 0)
    per_strat = result.get("per_strategy", {})
    print(f"📊 灵感策略统计 (共 {total} 次评分):\n")
    for name, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        info = per_strat.get(name, {})
        avg = info.get("avg_rating", "-")
        count = info.get("count", 0)
        bar = "█" * int(w * 5)
        print(f"  {name:25s} 权重={w:.2f} {bar}  (评{count}次, 均={avg})")


def cmd_server_stop(args):
    pid_file = _WORKSPACE / "memory" / "server.pid"
    if not pid_file.exists():
        print("Memory Server not running (no PID file)")
        return
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Memory Server stopped (PID {pid})")
    except OSError:
        print(f"Process {pid} not found, cleaning up PID file")
    if pid_file.exists():
        pid_file.unlink()


def main():
    parser = argparse.ArgumentParser(
        prog="memory-cli",
        description="Thin client for OpenClaw Memory Server",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("remember", help="Smart ingestion via Memory Hub")
    p.add_argument("content", help="Text to remember")
    p.add_argument("-i", "--importance", type=float, default=0.7)
    p.add_argument("-s", "--source", default="openclaw")
    p.add_argument("-d", "--doc-id", default=None)
    p.add_argument("-t", "--title", default=None)
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("recall", help="Merged search across all systems")
    p.add_argument("query", help="Search query")
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("deep-recall", help="MSA multi-hop reasoning")
    p.add_argument("query", help="Complex question")
    p.add_argument("-r", "--max-rounds", type=int, default=3)
    p.set_defaults(func=cmd_deep_recall)

    p = sub.add_parser("search", help="Memora vector search only")
    p.add_argument("query", help="Search query")
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("add", help="Add to Memora vector store")
    p.add_argument("content", help="Text to add")
    p.add_argument("-i", "--importance", type=float, default=0.7)
    p.add_argument("-s", "--source", default="cli")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("digest", help="Run memory digest")
    p.add_argument("--days", type=int, default=7)
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser("status", help="All systems status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("health", help="Server health check")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("collide", help="Run one round of inspiration collisions")
    p.set_defaults(func=cmd_collide)

    p = sub.add_parser("deep-collide", help="MSA multi-hop deep inspiration")
    p.add_argument("topic", nargs="?", default="", help="Optional focus topic")
    p.set_defaults(func=cmd_deep_collide)

    p = sub.add_parser("sb-report", help="Second Brain comprehensive report")
    p.set_defaults(func=cmd_sb_report)

    p = sub.add_parser("sb-status", help="Second Brain quick status")
    p.set_defaults(func=cmd_sb_status)

    p = sub.add_parser("briefing", help="Daily memory briefing")
    p.set_defaults(func=cmd_briefing)

    p = sub.add_parser("vitality", help="Memory vitality distribution")
    p.set_defaults(func=cmd_vitality)

    p = sub.add_parser("inspect", help="Inspect memory lifecycle")
    p.add_argument("query", help="Topic or keyword to inspect")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("review-dormant", help="List all dormant memories")
    p.set_defaults(func=cmd_review_dormant)

    p = sub.add_parser("tasks", help="List async tasks")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("contradictions", help="Scan KG for contradicted decisions")
    p.set_defaults(func=cmd_contradictions)

    p = sub.add_parser("blindspots", help="Detect blind spots in decisions")
    p.set_defaults(func=cmd_blindspots)

    p = sub.add_parser("threads", help="Discover thought threads from KG")
    p.set_defaults(func=cmd_threads)

    p = sub.add_parser("graph-status", help="Knowledge graph statistics")
    p.set_defaults(func=cmd_kg_status)

    p = sub.add_parser("rate", help="Rate an insight (1-5)")
    p.add_argument("insight_id", help="Insight ID or filename")
    p.add_argument("rating", type=int, help="Rating 1-5 (5=best)")
    p.add_argument("-s", "--strategy", default="", help="Strategy name")
    p.add_argument("-c", "--comment", default="", help="Optional comment")
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("insight-stats", help="Insight strategy statistics")
    p.set_defaults(func=cmd_insight_stats)

    p = sub.add_parser("server-start", help="Start the Memory Server")
    p.set_defaults(func=cmd_server_start)

    p = sub.add_parser("server-stop", help="Stop the Memory Server")
    p.set_defaults(func=cmd_server_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
