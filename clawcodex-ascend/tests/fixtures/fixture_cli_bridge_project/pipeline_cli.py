"""Minimal target-app CLI for CLI Bridge tests."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fixture-cli-bridge")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("execute-stage")
    run.add_argument("--stage-id", type=int, required=True)
    run.add_argument("--project-dir", default=".")
    run.add_argument("--stage-name", default="")
    args = parser.parse_args(argv)

    if args.command == "execute-stage":
        print(
            json.dumps(
                {
                    "ok": True,
                    "stage_id": args.stage_id,
                    "stage_name": args.stage_name,
                    "project_dir": args.project_dir,
                }
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
