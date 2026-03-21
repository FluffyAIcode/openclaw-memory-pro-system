#!/usr/bin/env python3
"""Memory Hub CLI — Unified interface to all OpenClaw memory systems."""

import argparse
import json
import logging
import sys

from rich.console import Console

from memory_hub import hub

console = Console()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Memory Hub — Unified OpenClaw memory interface")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_remember = sub.add_parser("remember", help="Smart ingestion to the right system(s)")
    p_remember.add_argument("content", nargs="*", help="Content to remember")
    p_remember.add_argument("-f", "--file", help="Read content from file")
    p_remember.add_argument("-i", "--importance", type=float, default=0.7)
    p_remember.add_argument("-s", "--source", default="cli")
    p_remember.add_argument("-t", "--title", help="Document title (for MSA)")
    p_remember.add_argument("-d", "--doc-id", help="Document ID (for MSA)")
    p_remember.add_argument("--systems", help="Force specific systems (comma-separated: memora,msa,chronos)")

    p_recall = sub.add_parser("recall", help="Merged search across all systems")
    p_recall.add_argument("query", nargs="+", help="Search query")
    p_recall.add_argument("-k", "--top-k", type=int, default=8)

    p_deep = sub.add_parser("deep-recall", help="Multi-hop reasoning via MSA interleave + Memora")
    p_deep.add_argument("query", nargs="+", help="Complex question")
    p_deep.add_argument("-r", "--rounds", type=int, default=3)

    sub.add_parser("status", help="Combined status from all systems")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "remember":
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        elif args.content:
            content = " ".join(args.content)
        else:
            console.print("[red]Error: provide text or --file[/red]")
            sys.exit(1)

        force = args.systems.split(",") if args.systems else None
        result = hub.remember(
            content, source=args.source, importance=args.importance,
            doc_id=args.doc_id, title=args.title, force_systems=force)

        console.print(f"[green]✓ Remembered ({result['word_count']} words)[/green]")
        console.print(f"  Systems: {', '.join(result['systems_used'])}")
        if "msa" in result:
            console.print(f"  MSA doc: {result['msa']['doc_id']} ({result['msa']['chunks']} chunks)")

    elif args.command == "recall":
        query = " ".join(args.query)
        result = hub.recall(query, top_k=args.top_k)
        console.print(f"[bold cyan]Memory Recall[/bold cyan] "
                      f"(Memora: {len(result['memora'])}, MSA: {len(result['msa'])})")
        for r in result["merged"][:args.top_k]:
            sys_tag = r.get("system", "?")
            score = r.get("score", 0)
            meta = r.get("metadata", {})
            title = meta.get("title", meta.get("doc_id", ""))
            content_preview = r.get("content", "")[:120].replace("\n", " ")
            console.print(f"  [{sys_tag}] score={score:.4f} {title}")
            console.print(f"    {content_preview}...")

    elif args.command == "deep-recall":
        query = " ".join(args.query)
        result = hub.deep_recall(query, max_rounds=args.rounds)
        interleave = result.get("interleave")
        if interleave:
            console.print(f"[bold cyan]Deep Recall[/bold cyan] "
                          f"({interleave['rounds']} rounds, {interleave['total_docs_used']} docs)")
            console.print(f"  Documents: {', '.join(interleave['doc_ids_used'])}")
            console.print(f"\n{interleave['final_answer'][:1000]}")
        else:
            console.print("[yellow]MSA interleave not available[/yellow]")

        snippets = result.get("memora_context", [])
        if snippets:
            console.print(f"\n[dim]+ {len(snippets)} Memora snippets for context[/dim]")

    elif args.command == "status":
        st = hub.status()
        console.print("[bold cyan]Memory Hub Status[/bold cyan]")
        for name, info in st["systems"].items():
            if "error" in info:
                console.print(f"  {name}: [red]{info['error']}[/red]")
            else:
                console.print(f"  {name}: {json.dumps(info, default=str)}")


if __name__ == "__main__":
    main()
