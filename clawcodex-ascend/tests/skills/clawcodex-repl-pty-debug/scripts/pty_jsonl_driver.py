#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Drive the ClawCodex REPL PTY JSONL controller from an ops file.

This is a convenience wrapper for hosts that cannot keep an interactive PTY
controller session open directly. It executes the controller as a child process,
writes one JSON operation per line, and prints controller JSON responses.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _read_ops(path: str) -> list[dict[str, Any]]:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).read_text(encoding="utf-8")

    ops: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"operation line {line_no} is not a JSON object")
        ops.append(value)
    return ops


def _default_controller_command(artifact_root: Path) -> list[str]:
    uv_cache_dir = Path(tempfile.gettempdir()) / "clawcodex-uv-cache"
    return [
        "uv",
        "--cache-dir",
        str(uv_cache_dir),
        "run",
        "--extra",
        "dev",
        "--frozen",
        "python",
        "scripts/debug/repl_pty_session.py",
        "interactive",
        "--artifact-root",
        str(artifact_root),
    ]


def _drain_available(stdout) -> None:
    while True:
        ready, _, _ = select.select([stdout], [], [], 0.05)
        if not ready:
            return
        line = stdout.readline()
        if not line:
            return
        sys.stdout.write(line)
        sys.stdout.flush()


def _send_op(proc: subprocess.Popen[str], op: dict[str, Any]) -> None:
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("controller pipes are unavailable")
    proc.stdin.write(json.dumps(op, ensure_ascii=False) + "\n")
    proc.stdin.flush()

    line = proc.stdout.readline()
    if not line:
        detail = ""
        if proc.poll() is not None and proc.stderr is not None:
            stderr = proc.stderr.read().strip()
            if stderr:
                detail = f": {stderr}"
        raise RuntimeError(f"controller exited before writing a response{detail}")
    sys.stdout.write(line)
    sys.stdout.flush()
    _drain_available(proc.stdout)


def _close_stdin(proc: subprocess.Popen[str]) -> None:
    if proc.stdin is None or proc.stdin.closed:
        return
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", required=True, help="JSONL operations file, or '-' for stdin")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "clawcodex-pty-jsonl-driver",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--no-auto-exit",
        action="store_true",
        help="Do not append an exit op when the ops file omits exit/quit.",
    )
    args = parser.parse_args(argv)

    ops = _read_ops(args.ops)
    if not args.no_auto_exit:
        if not ops or ops[-1].get("op") not in {"exit", "quit"}:
            ops.append({"op": "exit", "label": "driver auto exit"})

    env = os.environ.copy()
    env.setdefault("UV_SKIP_WHEEL_FILENAME_CHECK", "1")
    command = _default_controller_command(args.artifact_root)

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(args.repo_root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        sys.stderr.write(f"pty_jsonl_driver: failed to start controller: {exc}\n")
        return 1

    try:
        for op in ops:
            _send_op(proc, op)
    except RuntimeError as exc:
        _close_stdin(proc)
        if proc.poll() is None:
            proc.terminate()
        sys.stderr.write(f"pty_jsonl_driver: {exc}\n")
        return 1
    finally:
        _close_stdin(proc)

    stderr = proc.stderr.read() if proc.stderr is not None else ""
    rc = proc.wait()
    if stderr:
        sys.stderr.write(stderr)
        sys.stderr.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
