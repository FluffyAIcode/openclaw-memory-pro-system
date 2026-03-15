"""
记忆提炼 — 读取近期 daily 记录并生成长期摘要。
当前实现：合并近 N 天的 daily 文件内容；
vLLM 摘要生成需配置 config.vllm_url 后启用。
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def digest_memories(days: int = 7) -> bool:
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

    with open(digest_file, "w", encoding="utf-8") as f:
        f.write(f"# 记忆提炼 — {now}\n\n")
        f.write(f"来源: 最近 {days} 天 ({len(collected)} 个文件)\n\n")
        for date_str, content in collected:
            f.write(f"## {date_str}\n\n{content}\n\n---\n\n")
        # TODO: 接入 vLLM 生成摘要
        # summary = call_vllm(config.vllm_url, combined_text)
        # f.write(f"## AI 摘要\n\n{summary}\n")

    logger.info("提炼完成 → %s", digest_file)
    return True
