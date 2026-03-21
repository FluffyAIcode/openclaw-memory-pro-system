#!/usr/bin/env python3
"""Chronos CLI — 持续学习记忆系统命令行工具"""

import argparse
import json
import logging

from rich.console import Console

from .bridge import bridge
from .config import load_config

console = Console()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Chronos 持续学习记忆系统 CLI")
    sub = parser.add_subparsers(dest="command", help="可用命令")

    p_learn = sub.add_parser("learn", help="学习一条新记忆")
    p_learn.add_argument("content", nargs="+", help="记忆内容")
    p_learn.add_argument("-i", "--importance", type=float, default=0.75)
    p_learn.add_argument("-s", "--source", default="cli")

    sub.add_parser("consolidate", help="强制执行记忆巩固")
    sub.add_parser("status", help="查看系统状态")
    sub.add_parser("report", help="详细系统报告")
    sub.add_parser("init", help="初始化目录结构")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "learn":
        content = " ".join(args.content)
        enc = bridge.learn_and_save(
            content, source=args.source, importance=args.importance)
        console.print(
            f"[green]✓ 记忆已学习并内化[/green]  重要性={enc.importance:.2f}")

    elif args.command == "consolidate":
        result = bridge.consolidate()
        console.print("[green]✓ 记忆巩固完成[/green]")
        console.print(f"  巩固记忆数: {result.get('consolidated', 0)}")

    elif args.command == "status":
        st = bridge.status()
        console.print("[bold cyan]Chronos 系统状态[/bold cyan]")
        console.print(f"  缓冲区: {st['buffer_size']} 条记忆")
        console.print(f"  学习次数: {st['learn_count']}")
        console.print(f"  EWC 模式: {st['ewc_mode']}")
        console.print(f"  LoRA 适配器: {st['lora_adapters']}")
        console.print(f"  记忆类型: 参数级记忆（以算代存）")

    elif args.command == "report":
        rpt = bridge.report()
        console.print("[bold cyan]Chronos 详细报告[/bold cyan]")
        console.print(json.dumps(rpt, indent=2, ensure_ascii=False, default=str))

    elif args.command == "init":
        cfg = load_config()
        cfg.ensure_dirs()
        console.print("[green]✓ Chronos 目录已初始化[/green]")


if __name__ == "__main__":
    main()
