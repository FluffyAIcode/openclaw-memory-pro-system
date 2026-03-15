#!/usr/bin/env python3

from rich.console import Console
from memora.config import load_config

console = Console()


def main():
    console.print("[bold cyan]=== Memora 记忆管理系统 v1.0 ===[/bold cyan]")

    config = load_config()
    console.print(f"  记忆存储: {config.base_dir.resolve()}")

    console.print("\n可用命令：")
    console.print('  python3 -m memora add "内容"')
    console.print('  python3 -m memora search "关键词"')
    console.print("  python3 -m memora status")
    console.print("  streamlit run streamlit_app.py")
    console.print("\n项目已就绪。")


if __name__ == "__main__":
    main()
