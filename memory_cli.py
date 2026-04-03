#!/usr/bin/env python3
"""
memory-cli — Your AI memory assistant.

Quick start:
    memory-cli server-start          # Start the memory server
    memory-cli remember "something"  # Save a memory
    memory-cli recall "keyword"      # Search your memories
    memory-cli briefing              # Get today's summary
    memory-cli status                # Check system status
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

_BASE_URL = "http://127.0.0.1:18790"
_WORKSPACE = Path(__file__).resolve().parent
_auth_token: Optional[str] = None


def _load_auth_token() -> str:
    global _auth_token
    if _auth_token is not None:
        return _auth_token
    token_path = _WORKSPACE / "memory" / "security" / ".auth_token"
    try:
        if token_path.exists():
            _auth_token = token_path.read_text().strip()
            return _auth_token
    except OSError:
        pass
    _auth_token = ""
    return _auth_token


def _auth_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    token = _load_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _post(path: str, data: dict, timeout: int = 120) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(
        f"{_BASE_URL}{path}",
        data=body,
        headers=_auth_headers(),
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except URLError as e:
        if hasattr(e, "code") and e.code == 401:
            print("\n  认证失败: auth token 无效或缺失", file=sys.stderr)
            print(f"  检查: {_WORKSPACE / 'memory/security/.auth_token'}\n",
                  file=sys.stderr)
        else:
            print("\n  连接失败: Memory Server 未响应", file=sys.stderr)
            print("  请先启动: memory-cli server-start", file=sys.stderr)
            print("  然后等待约 3 分钟加载模型，再运行: memory-cli health\n",
                  file=sys.stderr)
        sys.exit(1)


def _get(path: str, timeout: int = 30) -> dict:
    hdrs = _auth_headers()
    hdrs.pop("Content-Type", None)
    req = Request(f"{_BASE_URL}{path}", headers=hdrs, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except URLError as e:
        if hasattr(e, "code") and e.code == 401:
            print("\n  认证失败: auth token 无效或缺失", file=sys.stderr)
            print(f"  检查: {_WORKSPACE / 'memory/security/.auth_token'}\n",
                  file=sys.stderr)
        else:
            print("\n  连接失败: Memory Server 未响应", file=sys.stderr)
            print("  请先启动: memory-cli server-start", file=sys.stderr)
            print("  然后等待约 3 分钟加载模型，再运行: memory-cli health\n",
                  file=sys.stderr)
        sys.exit(1)


def _post_async(path: str, data: dict, label: str = "") -> dict:
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


# ── Daily commands ────────────────────────────────────────

def cmd_remember(args):
    payload = {
        "content": args.content,
        "source": args.source,
        "importance": args.importance,
        "doc_id": args.doc_id,
        "title": args.title,
    }
    if args.tag:
        payload["tag"] = args.tag
    result = _post("/remember", payload)
    tag_label = f" #{args.tag}" if args.tag else ""
    print(f"  ✓ 已记住{tag_label}")


def cmd_recall(args):
    result = _post("/recall", {"query": args.query, "top_k": args.top_k})

    skills = result.get("skills", [])
    kg_rels = result.get("kg_relations", [])
    evidence = result.get("evidence", [])
    total = len(skills) + len(kg_rels) + len(evidence)

    if total == 0:
        print("  没有找到相关记忆。试试换个关键词？")
        return

    if skills:
        print(f"\n  ▎技能 ({len(skills)} 条匹配)  ──────────")
        for s in skills:
            print(f"  ★ {s.get('name', '?')}")
            tags = ", ".join(s.get("tags", []))
            if tags:
                print(f"    标签: {tags}")
            print(f"    {s.get('content', '')[:250]}")
            print()

    if kg_rels:
        print(f"  ▎知识关系 ({len(kg_rels)} 条)  ──────────")
        for rel in kg_rels:
            prefix = "⚠" if rel.get("is_critical") else "→"
            print(f"  {prefix} {rel.get('description', '')[:200]}")
        print()

    if evidence:
        print(f"  ▎原始记忆 ({len(evidence)} 条)  ──────────")
        for i, r in enumerate(evidence, 1):
            score = r.get("score", 0)
            content = r.get("content", "")[:200]
            ts = r.get("metadata", {}).get("timestamp", r.get("timestamp", ""))
            date = ts[:10] if ts else ""
            bar = "█" * max(1, int(score * 10))
            date_label = f" ({date})" if date else ""
            print(f"  [{i}] {bar} {score:.0%}{date_label}")
            print(f"      {content}")
            print()


def cmd_deep_recall(args):
    result = _post_async("/deep-recall", {"query": args.query, "max_rounds": args.max_rounds},
                         label="深度搜索")

    skills = result.get("skills", [])
    if skills:
        print(f"\n  ▎相关技能 ({len(skills)} 条)  ──────────")
        for s in skills:
            print(f"  ★ {s.get('name', '?')}: {s.get('content', '')[:150]}")
        print()

    interleave = result.get("interleave")
    if interleave:
        answer = interleave.get("final_answer", "")
        if answer:
            print(f"\n{answer[:2000]}")
    else:
        print("  深度搜索暂无结果（需要更多长文档记忆）")

    ctx = result.get("memora_context", [])
    if ctx:
        print(f"\n  --- 相关片段 ({len(ctx)} 条) ---")
        for r in ctx[:3]:
            print(f"  • {r.get('content', '')[:150]}")


def cmd_search(args):
    result = _post("/search", {"query": args.query, "limit": args.top_k})
    entries = result.get("results", [])
    if not entries:
        print("  没有找到相关记忆。")
        return
    for i, r in enumerate(entries, 1):
        print(f"  [{i}] ({r.get('score', 0):.0%}) {r.get('content', '')[:200]}")


def cmd_add(args):
    result = _post("/add", {
        "content": args.content,
        "source": args.source,
        "importance": args.importance,
    })
    print(f"  ✓ 已添加")


def cmd_digest(args):
    result = _post_async("/digest", {"days": args.days}, label="记忆总结")
    if result.get("success"):
        print(f"  ✓ 已生成最近 {args.days} 天的记忆总结")
    else:
        print("  总结生成失败")


def cmd_status(args):
    result = _get("/status")
    server = result.get("server", {})
    uptime = server.get("uptime_seconds", 0)
    hrs = uptime // 3600
    mins = (uptime % 3600) // 60

    print(f"  服务状态: 运行中 ({hrs}小时{mins}分)")
    print(f"  嵌入模型: {server.get('embedder', '?')}")
    print()
    for name, info in result.get("systems", {}).items():
        if isinstance(info, dict) and "error" not in info:
            if name == "memora":
                print(f"  记忆库: {info.get('entries', 0)} 条记忆")
            elif name == "chronos":
                print(f"  训练缓冲: {info.get('buffer_size', 0)} 条候选")
                nb = info.get("nebius", {})
                if nb:
                    print(f"  Nebius微调: {'已配置' if nb.get('configured') else '未配置'}")
            elif name == "msa":
                print(f"  长文档库: {info.get('document_count', 0)} 篇文档")
        elif isinstance(info, dict):
            print(f"  {name}: 异常 — {info.get('error', '?')}")


def cmd_health(args):
    result = _get("/health")
    status = result.get("status", "unknown")
    uptime = result.get("uptime_seconds", "?")
    pid = result.get("pid", "?")
    icon = "✓" if status == "ok" else "✗"
    print(f"  {icon} 服务正常 | PID {pid} | 已运行 {uptime}s")


def cmd_briefing(args):
    result = _get("/briefing")
    print(result.get("text", "无法生成简报"))


# ── Skills commands ───────────────────────────────────────

def cmd_skills(args):
    result = _get("/skills")
    skills = result.get("skills", [])
    if not skills:
        print("  还没有注册任何技能。用 memory-cli skill-add 创建第一个。")
        return
    print(f"  共 {len(skills)} 个技能:\n")
    status_icon = {"draft": "📝", "active": "✅", "deprecated": "🚫"}
    for s in skills:
        icon = status_icon.get(s.get("status", ""), "?")
        uses = s.get("total_uses", 0)
        utility = s.get("utility_rate", 0)
        use_label = f" | utility {utility:.0%} ({uses} uses)" if uses > 0 else ""
        print(f"  {icon} {s['name']} (v{s.get('version', 1)}) [{s['id']}]{use_label}")
        if s.get("tags"):
            print(f"     标签: {', '.join(s['tags'])}")


def cmd_skill_add(args):
    result = _post("/skills/add", {
        "name": args.name,
        "content": args.content,
        "tags": args.tags.split(",") if args.tags else [],
    })
    if "error" in result:
        print(f"  ✗ {result['error']}")
    else:
        print(f"  ✓ 技能已创建: {result.get('name', '')} [{result.get('id', '')}] (草稿)")


def cmd_skill_promote(args):
    result = _post("/skills/promote", {"skill_id": args.skill_id})
    if "error" in result:
        print(f"  ✗ {result['error']}")
    else:
        print(f"  ✓ 技能已激活: {result.get('name', '')}")


def cmd_skill_deprecate(args):
    result = _post("/skills/deprecate", {"skill_id": args.skill_id})
    if "error" in result:
        print(f"  ✗ {result['error']}")
    else:
        print(f"  ✓ 技能已废弃: {result.get('name', '')}")


def cmd_training_export(args):
    result = _post("/training/export", {})
    path = result.get("dataset_path", "")
    print(f"  ✓ 训练数据已导出: {path}")


def cmd_skill_propose(args):
    result = _post("/skills/propose", {"days": args.days})
    proposals = result.get("proposals", [])
    if not proposals:
        scores = result.get("scores", {})
        print("  暂无技能提名（需要任意两个分数达标）:")
        print(f"    KG structural_gain:      {scores.get('kg', 0):.2f} (阈值 0.6)")
        print(f"    Digest compression_value: {scores.get('digest', 0):.2f} (阈值 0.7)")
        print(f"    Collision novelty:        {scores.get('collision', 0)} (阈值 4)")
        return
    for p in proposals:
        print(f"  ✓ 已提名 draft 技能: {p.get('title', '?')}")
        for k, v in p.get("sources", {}).items():
            print(f"    {k} = {v}")


def cmd_skill_feedback(args):
    result = _post("/skills/feedback", {
        "skill_id": args.skill_id,
        "query": args.query,
        "outcome": args.outcome,
        "context": args.context or "",
    })
    if "error" in result:
        print(f"  ✗ {result['error']}")
    else:
        print(f"  ✓ 反馈已记录: {result.get('name', '')} → {args.outcome}")
        print(f"    utility: {result.get('utility_rate', 0):.0%} ({result.get('total_uses', 0)} uses, v{result.get('version', 0)})")


def cmd_skill_usage(args):
    result = _get("/skills/usage")
    usage = result.get("usage", [])
    if not usage:
        print("  还没有技能使用记录。")
        return
    print(f"  共 {len(usage)} 个技能有使用记录:\n")
    usage.sort(key=lambda x: x.get("total_uses", 0), reverse=True)
    for u in usage:
        bar = "█" * max(1, int(u.get("utility_rate", 0) * 10))
        print(f"  {bar} {u['utility_rate']:.0%} | {u['name']} "
              f"({u['successes']}✓ {u['failures']}✗ v{u['version']}) [{u['id']}]")


# ── Second Brain commands ─────────────────────────────────

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
        print(f"近期趋势 (最近3天):")
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
        print("  没有正在运行的任务。")
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
        print(f"没有沉睡记忆 :)")
        return

    if memories:
        print(f"🔇 {count} 条记忆已沉睡超过 {threshold} 天:\n")
        for i, m in enumerate(memories, 1):
            days = m.get("dormant_days", 0)
            content = m.get("content", "")[:120]
            print(f"  [{i}] {days}天未访问")
            print(f"      {content}")
            print()

    never = result.get("never_accessed", [])
    if never:
        print(f"\n📭 另有 {never_count} 条记忆从未被召回:\n")
        for i, m in enumerate(never, 1):
            days = m.get("dormant_days", 0)
            content = m.get("content", "")[:100]
            print(f"  [{i}] 已存在 {days} 天")
            print(f"      {content}")
            print()

    print("  提示: 用 memory-cli recall \"关键词\" 可唤醒相关记忆")


def cmd_contradictions(args):
    result = _get("/contradictions")
    count = result.get("count", 0)
    reports = result.get("reports", [])
    if not reports:
        print("✅ 没有发现矛盾决策")
        return
    print(f"⚠️ 发现 {count} 个矛盾:\n")
    for i, r in enumerate(reports, 1):
        decision = r.get("decision_content", "")[:120]
        print(f"  [{i}] {decision}")
        for c in r.get("contradicting", [])[:2]:
            print(f"      ❌ {c.get('content', '')[:100]}")
        print()


def cmd_blindspots(args):
    result = _get("/blindspots")
    reports = result.get("reports", [])
    if not reports:
        print("✅ 没有发现认知盲区")
        return
    print(f"🔍 发现 {len(reports)} 个待补充的决策:\n")
    for i, r in enumerate(reports, 1):
        decision = r.get("decision_content", "")[:120]
        missing = r.get("missing", [])
        print(f"  [{i}] {decision}")
        print(f"      未考虑: {', '.join(missing)}")
        print()


def cmd_threads(args):
    result = _get("/threads")
    threads = result.get("threads", [])
    if not threads:
        print("  还没有形成思维线索（需要更多记忆）")
        return
    print(f"🧵 发现 {len(threads)} 条思维线索:\n")
    for i, t in enumerate(threads, 1):
        title = t.get("title", "未命名")
        nc = t.get("node_count", 0)
        status = t.get("status", "?")
        icon = {"exploring": "🔎", "decided": "✅", "nascent": "🌱",
                "developing": "📈"}.get(status, "❓")
        print(f"  [{i}] {icon} {title} ({nc} 个知识点)")


def cmd_kg_status(args):
    result = _get("/kg/status")
    nodes = result.get("total_nodes", 0)
    edges = result.get("total_edges", 0)
    print(f"📊 知识图谱: {nodes} 个知识点, {edges} 条关联")
    nt = result.get("node_types", {})
    if nt:
        print(f"   类型: {', '.join(f'{k}={v}' for k, v in nt.items())}")


def cmd_rate(args):
    result = _post("/insight/rate", {
        "insight_id": args.insight_id,
        "strategy": args.strategy,
        "rating": args.rating,
        "comment": args.comment,
    })
    print(f"  ✓ 评分已记录 ({args.rating}/5)")


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


def cmd_session_context(args):
    result = _get("/session-context")

    summary = result.get("last_conversation_summary", "")
    if summary:
        date = result.get("last_conversation_date", "")
        print(f"📝 上次对话 ({date}): {summary[:200]}")
        topics = result.get("last_conversation_topics", [])
        if topics:
            print(f"   话题: {', '.join(topics)}")
        print()

    threads = result.get("active_threads", [])
    if threads:
        print("🧵 活跃思维线索:")
        for t in threads:
            status_icon = {"exploring": "🔎", "decided": "✅", "nascent": "🌱",
                           "goal": "🎯", "question": "❓", "decision": "✅"}.get(
                t.get("status", t.get("type", "")), "📌")
            title = t.get("title") or t.get("type", "?")
            count = t.get("node_count", t.get("count", 0))
            print(f"   {status_icon} {title} ({count} 节点)")
        print()

    focus = result.get("recent_focus", [])
    if focus:
        print(f"📈 近期关注: {', '.join(focus)}")
        print()

    personality = result.get("personality_traits", "")
    if personality:
        print(f"🎭 人格特质: {personality[:200]}")
        print()

    milestones = result.get("milestones", {})
    days = milestones.get("days_since_first_memory")
    if days is not None:
        print(f"🏆 记忆系统已运行 {days} 天")


def cmd_bookmark(args):
    result = _post("/bookmark", {
        "summary": args.summary,
        "topics": args.topics.split(",") if args.topics else [],
    })
    if result.get("ok"):
        print(f"  ✓ 书签已保存")
    else:
        print("  ✗ 书签保存失败")


# ── Server management ─────────────────────────────────────

def cmd_server_start(args):
    pid_file = _WORKSPACE / "memory" / "server.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        try:
            os.kill(int(pid), 0)
            print(f"  服务已在运行 (PID {pid})")
            return
        except (OSError, ValueError):
            pid_file.unlink()

    server_path = _WORKSPACE / "memory_server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_path), "--daemon"],
        cwd=str(_WORKSPACE),
    )
    proc.wait()
    print("  服务启动中... 首次启动需约 3 分钟加载模型")
    print("  运行 memory-cli health 确认就绪")


def cmd_server_stop(args):
    pid_file = _WORKSPACE / "memory" / "server.pid"
    if not pid_file.exists():
        print("  服务未运行")
        return
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  ✓ 服务已停止 (PID {pid})")
    except OSError:
        print(f"  进程 {pid} 已不存在，已清理 PID 文件")
    if pid_file.exists():
        pid_file.unlink()


# ── CLI argument parser ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="memory-cli",
        description="AI 记忆助手 — 帮你记住、回忆、整理碎片化知识",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
常用命令:
  remember    记住一段内容        recall      搜索记忆
  briefing    今日简报            status      系统状态

技能管理:
  skills      查看所有技能        skill-add   创建技能
  skill-on    激活技能            skill-off   废弃技能
  skill-feedback 记录使用反馈     skill-usage 使用统计

更多: collide, threads, contradictions, blindspots, inspect, vitality
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── Daily (top 5) ────
    p = sub.add_parser("remember", help="记住一段内容")
    p.add_argument("content", help="要记住的文本")
    p.add_argument("-i", "--importance", type=float, default=0.7, help="重要性 0-1 (默认 0.7)")
    p.add_argument("-s", "--source", default="cli", help="来源标记")
    p.add_argument("--tag", choices=["thought", "share", "reference", "to_verify"],
                   default=None, help="内容标签: thought=思考 | share=分享 | reference=参考 | to_verify=待验证")
    p.add_argument("-d", "--doc-id", default=None, help="文档ID (长文自动路由到MSA)")
    p.add_argument("-t", "--title", default=None, help="文档标题")
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("recall", help="搜索记忆")
    p.add_argument("query", help="搜索关键词")
    p.add_argument("-k", "--top-k", type=int, default=8, help="返回条数 (默认 8)")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("briefing", help="今日记忆简报")
    p.set_defaults(func=cmd_briefing)

    p = sub.add_parser("status", help="系统状态")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("health", help="服务健康检查")
    p.set_defaults(func=cmd_health)

    # ── Ingest extras ────
    p = sub.add_parser("deep-recall", help="深度搜索 (多轮推理)")
    p.add_argument("query", help="复杂问题")
    p.add_argument("-r", "--max-rounds", type=int, default=3)
    p.set_defaults(func=cmd_deep_recall)

    p = sub.add_parser("search", help="向量搜索 (仅 Memora)")
    p.add_argument("query", help="搜索关键词")
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("add", help="直接添加到向量库")
    p.add_argument("content", help="要添加的文本")
    p.add_argument("-i", "--importance", type=float, default=0.7)
    p.add_argument("-s", "--source", default="cli")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("digest", help="生成记忆总结")
    p.add_argument("--days", type=int, default=7, help="总结最近几天 (默认 7)")
    p.set_defaults(func=cmd_digest)

    # ── Skills ────
    p = sub.add_parser("skills", help="查看所有技能")
    p.set_defaults(func=cmd_skills)

    p = sub.add_parser("skill-add", help="创建新技能")
    p.add_argument("name", help="技能名称")
    p.add_argument("content", help="技能内容/描述")
    p.add_argument("--tags", default="", help="标签，逗号分隔")
    p.set_defaults(func=cmd_skill_add)

    p = sub.add_parser("skill-on", help="激活技能")
    p.add_argument("skill_id", help="技能 ID")
    p.set_defaults(func=cmd_skill_promote)

    p = sub.add_parser("skill-off", help="废弃技能")
    p.add_argument("skill_id", help="技能 ID")
    p.set_defaults(func=cmd_skill_deprecate)

    p = sub.add_parser("training-export", help="导出训练数据 (JSONL)")
    p.set_defaults(func=cmd_training_export)

    p = sub.add_parser("skill-propose", help="扫描 Second Brain 产出，自动提名技能")
    p.add_argument("--days", type=int, default=7, help="扫描最近 N 天")
    p.set_defaults(func=cmd_skill_propose)

    p = sub.add_parser("skill-feedback", help="记录技能使用反馈")
    p.add_argument("skill_id", help="技能 ID")
    p.add_argument("outcome", choices=["success", "failure"], help="使用结果")
    p.add_argument("--query", default="", help="触发查询")
    p.add_argument("--context", default="", help="反馈详情")
    p.set_defaults(func=cmd_skill_feedback)

    p = sub.add_parser("skill-usage", help="查看技能使用统计")
    p.set_defaults(func=cmd_skill_usage)

    # ── Second Brain ────
    p = sub.add_parser("collide", help="灵感碰撞 (从记忆中发现联系)")
    p.set_defaults(func=cmd_collide)

    p = sub.add_parser("deep-collide", help="深度碰撞 (多轮推理)")
    p.add_argument("topic", nargs="?", default="", help="聚焦话题")
    p.set_defaults(func=cmd_deep_collide)

    p = sub.add_parser("sb-report", help="第二大脑详细报告")
    p.set_defaults(func=cmd_sb_report)

    p = sub.add_parser("sb-status", help="第二大脑状态")
    p.set_defaults(func=cmd_sb_status)

    p = sub.add_parser("vitality", help="记忆活力分布")
    p.set_defaults(func=cmd_vitality)

    p = sub.add_parser("inspect", help="查看记忆生命周期")
    p.add_argument("query", help="要查看的记忆关键词")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("review-dormant", help="查看沉睡记忆")
    p.set_defaults(func=cmd_review_dormant)

    p = sub.add_parser("contradictions", help="扫描知识矛盾")
    p.set_defaults(func=cmd_contradictions)

    p = sub.add_parser("blindspots", help="检测认知盲区")
    p.set_defaults(func=cmd_blindspots)

    p = sub.add_parser("threads", help="发现思维线索")
    p.set_defaults(func=cmd_threads)

    p = sub.add_parser("graph-status", help="知识图谱统计")
    p.set_defaults(func=cmd_kg_status)

    p = sub.add_parser("rate", help="给灵感评分 (1-5)")
    p.add_argument("insight_id", help="灵感 ID")
    p.add_argument("rating", type=int, help="评分 1-5")
    p.add_argument("-s", "--strategy", default="")
    p.add_argument("-c", "--comment", default="")
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("insight-stats", help="灵感策略统计")
    p.set_defaults(func=cmd_insight_stats)

    p = sub.add_parser("session-context", help="会话上下文 (用于 Agent 启动)")
    p.set_defaults(func=cmd_session_context)

    p = sub.add_parser("bookmark", help="保存会话书签")
    p.add_argument("summary", help="对话摘要")
    p.add_argument("-t", "--topics", default="", help="话题 (逗号分隔)")
    p.set_defaults(func=cmd_bookmark)

    p = sub.add_parser("tasks", help="查看异步任务")
    p.set_defaults(func=cmd_tasks)

    # ── Server ────
    p = sub.add_parser("server-start", help="启动服务")
    p.set_defaults(func=cmd_server_start)

    p = sub.add_parser("server-stop", help="停止服务")
    p.set_defaults(func=cmd_server_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
