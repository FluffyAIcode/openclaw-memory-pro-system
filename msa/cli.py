#!/usr/bin/env python3
"""MSA CLI — Memory Sparse Attention memory system command-line tool"""

import argparse
import json
import logging
import sys

from rich.console import Console

from .bridge import bridge
from .config import load_config

console = Console()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="MSA (Memory Sparse Attention) CLI")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_ingest = sub.add_parser("ingest", help="Ingest text or file into memory bank")
    p_ingest.add_argument("content", nargs="*", help="Text content to ingest")
    p_ingest.add_argument("-f", "--file", help="File path to ingest")
    p_ingest.add_argument("-d", "--doc-id", help="Document ID (auto-generated if omitted)")
    p_ingest.add_argument("-s", "--source", default="cli")
    p_ingest.add_argument("-t", "--title", help="Document title")

    p_query = sub.add_parser("query", help="Single-round sparse retrieval")
    p_query.add_argument("question", nargs="+", help="Query text")
    p_query.add_argument("-k", "--top-k", type=int, help="Number of documents to retrieve")

    p_interleave = sub.add_parser("interleave",
                                  help="Multi-hop retrieval with Memory Interleave")
    p_interleave.add_argument("question", nargs="+", help="Query text")
    p_interleave.add_argument("-r", "--rounds", type=int, help="Max interleave rounds")

    p_remove = sub.add_parser("remove", help="Remove a document from memory bank")
    p_remove.add_argument("doc_id", help="Document ID to remove")

    sub.add_parser("status", help="Show memory bank stats")
    sub.add_parser("report", help="Detailed system report")
    sub.add_parser("init", help="Initialize directories")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "ingest":
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        elif args.content:
            content = " ".join(args.content)
        else:
            console.print("[red]Error: provide text or --file[/red]")
            sys.exit(1)

        metadata = {}
        if args.title:
            metadata["title"] = args.title

        result = bridge.ingest_and_save(
            content, source=args.source,
            doc_id=args.doc_id, metadata=metadata)
        console.print(f"[green]✓ Document ingested[/green]  "
                      f"doc_id={result['doc_id']}  chunks={result['chunks']}")

    elif args.command == "query":
        question = " ".join(args.question)
        result = bridge.query_memory(question, top_k=args.top_k)
        console.print(f"[bold cyan]Query Results[/bold cyan] "
                      f"({result['total_results']}/{result['total_documents']} docs)")
        for doc in result["results"]:
            console.print(f"  [{doc['rank']}] {doc['title']}  "
                          f"score={doc['score']:.4f}  chunks={doc['chunk_count']}")
            for i, chunk in enumerate(doc["chunks"][:2]):
                preview = chunk[:120].replace("\n", " ")
                console.print(f"      chunk {i}: {preview}...")

    elif args.command == "interleave":
        question = " ".join(args.question)
        result = bridge.interleave_query(question, max_rounds=args.rounds)
        console.print(f"[bold cyan]Memory Interleave[/bold cyan] "
                      f"({result['rounds']} rounds, {result['total_docs_used']} docs)")
        console.print(f"  Documents used: {', '.join(result['doc_ids_used'])}")
        console.print(f"\n[bold]Result:[/bold]")
        console.print(result["final_answer"][:1000])

    elif args.command == "remove":
        if bridge.remove(args.doc_id):
            console.print(f"[green]✓ Document '{args.doc_id}' removed[/green]")
        else:
            console.print(f"[red]Document '{args.doc_id}' not found[/red]")

    elif args.command == "status":
        st = bridge.status()
        console.print("[bold cyan]MSA System Status[/bold cyan]")
        console.print(f"  Documents: {st['document_count']}")
        console.print(f"  Total chunks: {st['total_chunks']}")
        console.print(f"  Ingested: {st['ingest_count']} times")
        console.print(f"  Top-k: {st['top_k']}")
        console.print(f"  Chunk size: {st['chunk_size']}")
        console.print(f"  Similarity threshold: {st['similarity_threshold']}")
        console.print(f"  Max interleave rounds: {st['max_interleave_rounds']}")
        console.print(f"  Memory type: Sparse routing + document-level retrieval")

    elif args.command == "report":
        rpt = bridge.report()
        console.print("[bold cyan]MSA Detailed Report[/bold cyan]")
        console.print(json.dumps(rpt, indent=2, ensure_ascii=False, default=str))

    elif args.command == "init":
        cfg = load_config()
        cfg.ensure_dirs()
        console.print("[green]✓ MSA directories initialized[/green]")


if __name__ == "__main__":
    main()
