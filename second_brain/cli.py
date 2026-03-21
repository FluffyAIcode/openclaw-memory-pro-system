"""CLI for Second Brain — collide, report, status."""

import argparse
import json
import sys

from .bridge import bridge


def cmd_collide(args):
    result = bridge.collide()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_report(args):
    result = bridge.report()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_status(args):
    result = bridge.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="second-brain",
        description="Second Brain — long-term memory tracking & inspiration collision",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("collide", help="Run one round of inspiration collisions")
    sub.add_parser("report", help="Generate comprehensive report")
    sub.add_parser("status", help="Quick status check")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "collide": cmd_collide,
        "report": cmd_report,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
