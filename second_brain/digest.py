"""
记忆蒸馏/总结 — 读取近期 daily 记录并生成长期摘要。

Moved from memora/ to second_brain/ as part of the architecture refactor:
distillation is a "value extraction" step, not a storage step.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()


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
            summary = llm_client.generate(
                prompt=(
                    f"以下是最近 {days} 天的记忆日志（共 {len(collected)} 天）。"
                    "请提炼出：\n"
                    "1. 关键决策和结论\n"
                    "2. 重要发现和新知识\n"
                    "3. 值得长期保留的用户偏好和模式\n"
                    "4. 未完成的任务或待跟进事项\n\n"
                    f"记忆日志：\n{combined_text[:8000]}"
                ),
                system="你是记忆提炼助手。生成简洁的结构化摘要，用中文。",
                max_tokens=1500,
            )
    except ImportError:
        pass

    with open(digest_file, "w", encoding="utf-8") as f:
        f.write(f"# 记忆提炼 — {now}\n\n")
        f.write(f"来源: 最近 {days} 天 ({len(collected)} 个文件)\n\n")

        if summary:
            f.write(f"## AI 摘要\n\n{summary}\n\n---\n\n")

        f.write("## 原始记录\n\n")
        for date_str, content in collected:
            f.write(f"### {date_str}\n\n{content}\n\n---\n\n")

    logger.info("提炼完成 → %s (AI摘要: %s)", digest_file, "有" if summary else "无")
    return True
