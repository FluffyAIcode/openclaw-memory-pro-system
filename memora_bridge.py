#!/usr/bin/env python3
"""
向后兼容入口 — 所有逻辑已迁入 memora.bridge
"""

from memora.bridge import MemoraBridge, bridge

__all__ = ["MemoraBridge", "bridge"]

if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    console.print("[bold cyan]Memora-OpenClaw Bridge v1.0 已加载[/bold cyan]")
    console.print("可用方法: bridge.save_to_both(), bridge.auto_digest()")
