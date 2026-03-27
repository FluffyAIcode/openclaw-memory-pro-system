#!/usr/bin/env python3
"""
Memora CLI - 命令行工具
"""

import argparse
import logging
from rich.console import Console

from .config import load_config

console = Console()
logger = logging.getLogger(__name__)
config = load_config()


def main():
    parser = argparse.ArgumentParser(description="Memora 记忆管理系统 CLI")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    add_parser = subparsers.add_parser("add", help="添加一条记忆")
    add_parser.add_argument("content", nargs="+", help="记忆内容")
    add_parser.add_argument("-i", "--importance", type=float, default=0.7, help="重要性 (0.0-1.0)")
    add_parser.add_argument("-s", "--source", default="cli", help="来源")

    search_parser = subparsers.add_parser("search", help="搜索记忆")
    search_parser.add_argument("query", nargs="+", help="搜索关键词")

    digest_parser = subparsers.add_parser("digest", help="提炼记忆")
    digest_parser.add_argument("--days", type=int, default=config.digest_interval_days, help="提炼最近多少天")

    subparsers.add_parser("status", help="查看系统状态")
    subparsers.add_parser("init", help="初始化目录结构")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "add":
        from memory_hub import hub
        content = " ".join(args.content)
        hub.remember(
            content=content,
            source=args.source,
            importance=args.importance,
        )
        console.print("[green]✓ 记忆已通过 CLI 添加[/green]")

    elif args.command == "search":
        from .vectorstore import vector_store
        query = " ".join(args.query)
        results = vector_store.search(query, limit=8)
        if not results:
            console.print("[yellow]未找到相关记忆[/yellow]")
            return
        console.print(f"[bold]找到 {len(results)} 条相关记忆:[/bold]")
        for r in results:
            score = r.get("score", "?")
            text = r.get("content", str(r))[:100]
            console.print(f"  [{score}] {text}")

    elif args.command == "digest":
        from second_brain.digest import digest_memories
        digest_memories(days=config.digest_interval_days)
        console.print("[green]✓ 记忆提炼完成[/green]")

    elif args.command == "status":
        from .vectorstore import vector_store
        console.print("[bold cyan]Memora 系统状态[/bold cyan]")
        console.print(f"  记忆存储: {config.base_dir.resolve()}")
        console.print(f"  向量条目: {vector_store.count()}")
        console.print(f"  嵌入模型: {config.embedding_model}")
        console.print(f"  ZFS snapshot: {config.use_zfs_snapshot}")

    elif args.command == "init":
        config.ensure_dirs()
        console.print("[green]✓ 所有目录已创建[/green]")


if __name__ == "__main__":
    main()
