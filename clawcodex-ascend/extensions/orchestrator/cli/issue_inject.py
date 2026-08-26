#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .issue_control import (
    _resolve_issue_workspace_path,
    _resolve_sock_path,
    _send_and_wait,
    _try_socket_inject,
)


def _run_inject(args: argparse.Namespace) -> int:
    """Inject operator hints. Idempotent — listing/removal are safe."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    ws_path = _resolve_issue_workspace_path(issue_id)
    hints_file = ws_path / ".operator_hints.md" if ws_path else None
    if hints_file is None:
        print(
            f"Could not find workspace for issue {issue_id}.\n"
            "Hints are stored in the issue's workspace directory.\n"
            "Set CLAWCODEX_WORKSPACE_ROOT or run the orchestrator with --workflow.",
            file=sys.stderr,
        )
        return 1

    hint = getattr(args, "hint", None)
    list_hints = getattr(args, "list_hints", False)
    remove_hint = getattr(args, "remove_hint", None)

    if list_hints or (not hint and remove_hint is None):
        # List hints
        return _list_hints(issue_id, hints_file)
    elif remove_hint is not None:
        return _remove_hint(issue_id, hints_file, remove_hint)
    elif hint:
        no_wait = getattr(args, "no_wait", False)
        sock_path = _resolve_sock_path(issue_id)
        if sock_path is not None and not no_wait:

            async def _do_inject() -> int:
                t0 = asyncio.get_event_loop().time()

                # 1. Pause the agent so the message can be safely
                #    written to the transcript at a clean boundary.
                try:
                    pause_data = await _send_and_wait(
                        sock_path,
                        "pause",
                        "",
                        "Paused",
                        timeout=30.0,
                    )
                    if pause_data is None:
                        print(
                            "warning: pause not confirmed within 30s — "
                            "agent may be in a long operation or already "
                            "paused. Injecting anyway.",
                            file=sys.stderr,
                        )
                except (ConnectionRefusedError, FileNotFoundError, OSError):
                    # Socket gone — fall back to file.
                    return _inject_hint(issue_id, hints_file, hint)

                # 2. Inject the message (writes UserMessage to transcript
                #    + queues for in-memory Conversation).
                try:
                    data = await _send_and_wait(
                        sock_path,
                        "inject",
                        hint,
                        "InjectDelivered",
                        timeout=30.0,
                    )
                except (ConnectionRefusedError, FileNotFoundError, OSError):
                    return _inject_hint(issue_id, hints_file, hint)

                # 3. Auto-resume so the agent processes the message.
                try:
                    await _send_and_wait(
                        sock_path,
                        "resume",
                        "",
                        "Resumed",
                        timeout=30.0,
                    )
                except (ConnectionRefusedError, FileNotFoundError, OSError):
                    pass  # Best-effort resume

                elapsed = asyncio.get_event_loop().time() - t0
                if data is not None:
                    snippet = data.get("hint_snippet", "")
                    print(
                        f"Message injected and agent resumed ({elapsed:.1f}s). Agent will see it in its next response."
                    )
                    if snippet:
                        print(f"  hint: {snippet}{'...' if len(hint) > 80 else ''}")
                    return 0
                else:
                    print(f"Hint queued ({elapsed:.1f}s). Will be delivered at next tool result boundary.")
                    return 0

            return asyncio.run(_do_inject())
        elif sock_path is not None and no_wait:
            if _try_socket_inject(issue_id, hint):
                print(
                    f"\u2713 hint injected for issue {issue_id}"
                    f" \u00b7 agent will receive it at the next tool result boundary"
                )
                return 0
            return _inject_hint(issue_id, hints_file, hint)
        else:
            return _inject_hint(issue_id, hints_file, hint)
    else:
        return _list_hints(issue_id, hints_file)


def _parse_hints_file(hints_file: Path) -> list[tuple[float, str]]:
    """Parse hints file into list of (timestamp, hint) tuples."""
    import time

    if not hints_file.exists():
        return []

    hints: list[tuple[float, str]] = []
    content = hints_file.read_text(encoding="utf-8")
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("--- Operator Hint #"):
            ts_str = ""
            try:
                parts = line.split("(injected at ")
                if len(parts) > 1:
                    ts_str = parts[1].rstrip(") ---")
                    from datetime import datetime

                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    ts = dt.timestamp()
                else:
                    ts = time.time()
            except Exception:
                ts = time.time()

            hint_lines: list[str] = []
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("-" * 45):
                    break
                hint_lines.append(lines[i])
                i += 1
            hint = "\n".join(hint_lines).strip()
            if hint:
                hints.append((ts, hint))
        i += 1
    return hints


def _inject_hint(issue_id: str, hints_file: Path, hint: str) -> int:
    """Append a hint to the .operator_hints.md file.

    Idempotent: if the hint text already exists in the file, it is
    not duplicated.
    """
    import time

    hints = _parse_hints_file(hints_file)
    # Idempotency — skip if the exact hint text
    # already exists.
    for _ts, existing_hint in hints:
        if existing_hint.strip() == hint.strip():
            print(f"Hint already exists for issue {issue_id} — no action taken.")
            return 0
    next_num = len(hints) + 1
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    header = f"--- Operator Hint #{next_num} (injected at {timestamp}) ---\n"
    separator = "-" * 50 + "\n"
    try:
        with open(hints_file, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(hint + "\n")
            f.write(separator)
        print(
            f"\u2713 hint injected for issue {issue_id} \u00b7 agent will pick it up at the next tool result boundary"
        )
        return 0
    except Exception as exc:
        print(f"Failed to inject hint: {exc}", file=sys.stderr)
        return 1


def _list_hints(issue_id: str, hints_file: Path) -> int:
    """List all hints for an issue."""
    hints = _parse_hints_file(hints_file)
    if not hints:
        print(f"No hints for issue {issue_id}.")
        return 0
    print(f"Hints for issue {issue_id}:")
    for i, (ts, hint) in enumerate(hints, 1):
        import time

        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        preview = hint[:60].replace("\n", " ")
        print(f"  #{i}: [{ts_str}] {preview}")
    return 0


def _remove_hint(issue_id: str, hints_file: Path, hint_num: int) -> int:
    """Remove a hint by number."""
    hints = _parse_hints_file(hints_file)
    if hint_num < 1 or hint_num > len(hints):
        print(f"Hint #{hint_num} not found (have {len(hints)} hints).", file=sys.stderr)
        return 1

    hints.pop(hint_num - 1)
    # Rebuild file
    import time

    content = ""
    for i, (ts, hint) in enumerate(hints, 1):
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        header = f"--- Operator Hint #{i} (injected at {ts_str}) ---\n"
        separator = "-" * 50 + "\n"
        content += header + hint + "\n" + separator
    try:
        hints_file.write_text(content, encoding="utf-8")
    except Exception as exc:
        print(f"Failed to remove hint: {exc}", file=sys.stderr)
        return 1
    print(f"Removed hint #{hint_num} for issue {issue_id}.")
    return 0


# ---------------------------------------------------------------------------
# issue workspace
# ---------------------------------------------------------------------------
