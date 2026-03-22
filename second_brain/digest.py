"""
记忆蒸馏/总结 — 从 daily 文件和 MSA 文档库中提取素材，生成长期摘要。

数据来源:
  1. memory/YYYY-MM-DD.md — 段落级日志（原始记录）
  2. MSA 文档库 — 文档级跨天上下文（通过 interleave 获取深层关联）

Moved from memora/ to second_brain/ as part of the architecture refactor:
distillation is a "value extraction" step, not a storage step.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def _collect_msa_context(days: int) -> str:
    """Query MSA for cross-day themes to enrich the digest."""
    try:
        from msa.bridge import bridge as msa_bridge
        status = msa_bridge.status()
        if status.get("document_count", 0) == 0:
            return ""

        result = msa_bridge.interleave_query(
            f"总结最近{days}天的核心主题、关键决策和跨天关联",
            max_rounds=2,
        )
        answer = result.get("final_answer", "")
        if answer and len(answer) > 30:
            docs_used = result.get("total_docs_used", 0)
            logger.info("Digest: MSA interleave 提供跨天上下文 (%d docs, %d chars)",
                        docs_used, len(answer))
            return answer
    except Exception as e:
        logger.warning("Digest: MSA context fetch failed: %s", e)
    return ""


def digest_memories(days: int = None) -> bool:
    if days is None:
        days = config.digest_interval_days

    logger.info("正在提炼最近 %d 天的记忆...", days)

    daily_dir = config.daily_dir
    if not daily_dir.exists():
        logger.warning("daily 目录不存在: %s", daily_dir)
        return False

    cutoff = datetime.now() - timedelta(days=days)
    collected = []

    for md_file in sorted(daily_dir.glob("*.md")):
        try:
            file_date = datetime.strptime(md_file.stem, "%Y-%m-%d")
        except ValueError:
            continue
        if file_date >= cutoff:
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                collected.append((md_file.stem, content))

    if not collected:
        logger.info("没有需要提炼的记忆")
        return True

    logger.info("收集了 %d 天的记忆数据", len(collected))

    msa_context = _collect_msa_context(days)

    long_term_dir = config.long_term_dir
    long_term_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    digest_file = long_term_dir / f"digest_{now}.md"

    combined_text = "\n\n".join(f"[{d}]\n{c}" for d, c in collected)

    summary = None
    try:
        import llm_client
        if llm_client.is_available():
            logger.info("调用 LLM 生成记忆摘要...")

            msa_section = ""
            if msa_context:
                msa_section = (
                    "\n\n## MSA 跨天关联分析（来自文档级多跳推理）\n"
                    f"{msa_context[:3000]}\n"
                )

            summary = llm_client.generate(
                prompt=(
                    f"以下是最近 {days} 天的记忆日志（共 {len(collected)} 天）。"
                    f"{msa_section}\n"
                    "请提炼出：\n"
                    "1. 关键决策和结论\n"
                    "2. 重要发现和新知识\n"
                    "3. 跨天的主题演变和深层关联\n"
                    "4. 值得长期保留的用户偏好和模式\n"
                    "5. 未完成的任务或待跟进事项\n\n"
                    f"记忆日志：\n{combined_text[:8000]}"
                ),
                system="你是记忆提炼助手。生成简洁的结构化摘要，用中文。特别注意跨天的知识关联和主题演变。",
                max_tokens=2000,
            )
    except ImportError:
        pass

    with open(digest_file, "w", encoding="utf-8") as f:
        f.write(f"# 记忆提炼 — {now}\n\n")
        f.write(f"来源: 最近 {days} 天 ({len(collected)} 个文件)")
        if msa_context:
            f.write(f" + MSA 跨天关联")
        f.write("\n\n")

        if summary:
            f.write(f"## AI 摘要\n\n{summary}\n\n---\n\n")

        if msa_context:
            f.write(f"## MSA 跨天上下文\n\n{msa_context[:2000]}\n\n---\n\n")

        f.write("## 原始记录\n\n")
        for date_str, content in collected:
            f.write(f"### {date_str}\n\n{content}\n\n---\n\n")

    cv = _calc_compression_value(combined_text, summary, msa_context,
                                   len(collected), days)
    logger.info("提炼完成 → %s (AI摘要: %s, MSA上下文: %s, compression_value=%.2f)",
                digest_file, "有" if summary else "无",
                "有" if msa_context else "无", cv)

    _save_digest_score(digest_file, cv)
    return True


def _calc_compression_value(raw_text: str, summary: str, msa_context: str,
                            days_collected: int, days_requested: int) -> float:
    """Score how valuable this digest is (0.0–1.0).

    Components:
      - compression_ratio: how much text was reduced
      - decision_density: how many actionable items in the summary
      - day_coverage: what fraction of the requested time range was covered
      - novelty_vs_last: how different from the most recent prior digest
    """
    if not summary:
        return 0.1

    input_len = max(len(raw_text), 1)
    output_len = len(summary)
    compression_ratio = 1.0 - min(output_len / input_len, 1.0)

    decision_keywords = ["决策", "决定", "选择", "发现", "确认", "偏好",
                         "结论", "方案", "目标", "关键", "重要"]
    keyword_hits = sum(1 for kw in decision_keywords if kw in summary)
    decision_density = min(keyword_hits / 5.0, 1.0)

    day_coverage = min(days_collected / max(days_requested, 1), 1.0)

    novelty_vs_last = _compare_with_last_digest(summary)

    value = (
        0.2 * compression_ratio
        + 0.3 * decision_density
        + 0.2 * day_coverage
        + 0.3 * novelty_vs_last
    )
    return round(value, 4)


def _compare_with_last_digest(current_summary: str) -> float:
    """Estimate novelty by comparing with the most recent prior digest."""
    long_term_dir = config.long_term_dir
    if not long_term_dir or not long_term_dir.exists():
        return 0.8

    digests = sorted(long_term_dir.glob("digest_*.md"), reverse=True)
    if len(digests) < 2:
        return 0.8

    try:
        prev_text = digests[1].read_text(encoding="utf-8")[:3000]
    except OSError:
        return 0.8

    current_chars = set(current_summary)
    prev_chars = set(prev_text)
    if not current_chars:
        return 0.5
    overlap = len(current_chars & prev_chars) / len(current_chars | prev_chars)
    return round(1.0 - overlap, 4)


def _save_digest_score(digest_file: Path, score: float):
    """Persist the score alongside the digest for the Skill Proposer to read."""
    score_file = digest_file.with_suffix(".score.json")
    try:
        import json
        score_file.write_text(json.dumps({
            "digest_file": digest_file.name,
            "compression_value": score,
            "timestamp": datetime.now().isoformat(),
        }, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save digest score: %s", e)
