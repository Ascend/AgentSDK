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
import json
import sys
from pathlib import Path

from .issue_control import _resolve_sock_path, _resolve_tail_run_id, _send_and_wait, _write_control


def _format_ts(timestamp_str: str | None) -> str:
    """Format an ISO-8601 timestamp string to ``HH:MM:SS``.

    Falls back to the current local time when the transcript entry
    has no timestamp (legacy records, session_snapshot lines, etc.).
    """
    if timestamp_str:
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(timestamp_str)
            return dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            pass
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def _summarize_tool_args(name: str, inp: dict) -> str:
    """Return a one-line argument summary for a tool call.

    Examples::

        Read → ``src/services/lock.py``
        Grep → ``"asyncio.Lock"``
        Edit → ``src/services/lock.py``
        Bash → ``pytest tests/test_lock.py``
        Git  → ``commit -m "fix: …"``
    """
    if not inp:
        return ""
    if name == "Read":
        return inp.get("file_path", inp.get("path", "")).strip()
    if name in ("Grep", "grep"):
        pat = inp.get("pattern", "")
        return f'"{pat}"' if pat else ""
    if name in ("Edit", "Write", "create", "Create"):
        return inp.get("file_path", inp.get("path", "")).strip()
    if name in ("Bash", "bash", "Git", "git"):
        cmd = inp.get("command", "")
        return cmd.strip()[:90]
    # Fallback: join first 3 non-empty string values
    parts = [str(v)[:60] for v in inp.values() if isinstance(v, str) and v.strip()]
    return " ".join(parts[:3])


def _summarize_tool_result(name: str, content: str | list | None) -> str:
    """Return a brief one-line result summary for a tool result.

    Returns empty string when no meaningful summary can be inferred.
    """
    if not content:
        return ""
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                break

    if not text.strip():
        return ""

    # Line count for Read
    if name in ("Read", "read"):
        n = text.count("\n") + 1
        return f"{n} lines"

    # Hit count for Grep
    if name in ("Grep", "grep"):
        lines = text.strip().splitlines()
        # Count actual match lines (omit "X results" footer / header lines)
        match_lines = [line for line in lines if line.strip() and not line.startswith("─")]
        return f"{len(match_lines)} hits" if match_lines else "0 hits"

    # Diff stat for Edit
    if name in ("Edit", "edit", "Write", "write"):
        n = text.count("\n") + 1 if text.strip() else 0
        # If the result is just "No changes" or similar, say so
        text_lower = text.strip().lower()
        if "no change" in text_lower or "nothing" in text_lower:
            return "no changes"
        # Show first line of diff patch as preview
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if first_line.startswith("diff --git"):
            return f"+{n} lines" if n > 0 else "0 changes"
        return f"+{n} lines" if n > 0 else ""

    # Exit code / summary for Bash
    if name in ("Bash", "bash"):
        first = text.strip().splitlines()[0] if text.strip() else ""
        # Look for common test result patterns
        passed = ""
        import re

        m = re.search(r"(\d+)\s+passed", text)
        if m:
            passed = m.group(0)
        failed = ""
        m = re.search(r"(\d+)\s+failed", text)
        if m:
            failed = m.group(0)
        if passed or failed:
            parts = [p for p in (passed, failed) if p]
            return " · ".join(parts) if parts else "done"
        # Return first meaningful output line
        first = first.rstrip("\n")[:60]
        return first if first else "done"

    return ""


def _render_message(msg: dict, turn_counter: int, pending_calls: dict) -> None:
    """Render one Message dict from transcript.jsonl as a tail line.

    Produces output matching the README Demo format::

        14:02:11  ◐ Read src/services/lock.py · 132 lines
        14:02:13  ◐ Grep "asyncio.Lock" · 3 hits
        14:02:18  ◐ Edit src/services/lock.py · +18 -4
        14:02:24  ◐ Bash pytest tests/test_lock.py · 4 passed
        14:02:24  ✓ Verification gate OK (pytest -x)
        14:02:25  ◐ Git commit -m "fix: per-key lock granularity in flush_batch"
        14:02:26  ◐ Git push origin clawcodex/AGENTSDK-15
        14:02:31  ✓ PR opened · auto-review-loop subscribed

    tool_use + tool_result pairs are merged into a single line by
    buffering the tool_use in ``pending_calls`` (keyed by tool_use_id)
    and rendering when the matching tool_result arrives.
    """
    role = msg.get("role", "?")
    content = msg.get("content")
    if not isinstance(content, list):
        return

    ts = _format_ts(msg.get("timestamp"))

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        # -- tool_use: buffer the call, render when result arrives --
        if btype == "tool_use" and role == "assistant":
            name = block.get("name", "?")
            tid = block.get("id") or block.get("tool_use_id", "")
            inp = block.get("input", {})
            pending_calls[tid] = {
                "name": name,
                "input": inp,
                "timestamp": ts,
            }

        # -- tool_result: pair with buffered tool_use and render --
        elif btype == "tool_result" and role == "user":
            tid = block.get("tool_use_id", "?")
            err = block.get("is_error", False)
            result_content = block.get("content", "")

            call = pending_calls.pop(tid, None)
            if call:
                name = call["name"]
                inp = call["input"]
                call_ts = call["timestamp"]
                args_str = _summarize_tool_args(name, inp)
                result_str = _summarize_tool_result(name, result_content)

                icon = "✗" if err else "◐"
                line = f"{call_ts}  {icon} {name}"
                if args_str:
                    line += f" {args_str}"
                if result_str:
                    line += f" · {result_str}"
                print(line)
            else:
                icon = "✗" if err else "◐"
                print(f"{ts}  {icon} [result {tid}]")

        # -- assistant text: special-cased detection --
        elif btype == "text" and role == "assistant":
            text = (block.get("text") or "").strip()
            if not text:
                continue

            lower = text.lower()

            # Verification gate passed
            if "pytest" in lower and ("passed" in lower or "ok" in lower):
                preview = text[:80].replace("\n", " ")
                # Strip to a single line
                preview = preview.strip()
                print(f"{ts}  ✓ Verification gate OK ({preview})")
            # PR opened
            elif "pr opened" in lower or "pull request" in lower or "opened pr" in lower:
                preview = text[:80].replace("\n", " ")
                print(f"{ts}  ✓ PR opened · {preview.strip()}")
            # Git operations
            elif lower.startswith("git") or "git commit" in lower or "committed" in lower:
                preview = text[:80].replace("\n", " ")
                print(f"{ts}  ◐ {preview.strip()}")
            elif "push" in lower and ("git" in lower or "origin" in lower):
                preview = text[:80].replace("\n", " ")
                print(f"{ts}  ◐ {preview.strip()}")
            # Generic assistant text
            else:
                preview = text[:80].replace("\n", " ")
                print(f"{ts}  ◐ {preview.strip()}")


def _msg_references_tool(msg: dict, tool_use_id: str) -> bool:
    """Whether a Message dict contains any block referring to tool_use_id.

    Matches both ``tool_use.id`` and ``tool_result.tool_use_id`` so the
    filter surfaces the full tool_use + tool_result pair, not just one
    half of it.
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "tool_use" and block.get("id") == tool_use_id:
            return True
        if btype == "tool_result" and block.get("tool_use_id") == tool_use_id:
            return True
    return False


def _print_message(
    msg: dict,
    tool_use_id_filter: str | None = None,
) -> None:
    """Print one Message dict from transcript.jsonl in human-readable form.

    Designed for `issue transcript` (snapshot mode) — full content
    instead of the one-line preview used by `issue tail`.

    When ``tool_use_id_filter`` is set, only blocks that reference that
    tool_use id are printed: ``tool_use.id == filter`` or
    ``tool_result.tool_use_id == filter``. Text blocks in the same
    message are suppressed under filter, so a single multi-tool
    assistant message prints only the relevant tool_use (not the
    unrelated ones that share the same message).
    """
    role = msg.get("role", "?")
    origin = msg.get("origin", "")
    origin_suffix = f" (origin={origin})" if origin else ""
    print(f"## {role}{origin_suffix}")
    content = msg.get("content")
    if isinstance(content, str):
        if tool_use_id_filter is None:
            for line in content.splitlines():
                print(f"  Text: {line}")
        print()
        return
    if not isinstance(content, list):
        return
    printed_any_block = False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if tool_use_id_filter is not None:
            if btype == "tool_use" and block.get("id") != tool_use_id_filter:
                continue
            if btype == "tool_result" and block.get("tool_use_id") != tool_use_id_filter:
                continue
            if btype == "text":
                continue
        if btype == "text":
            text = (block.get("text") or "").rstrip()
            if text:
                for line in text.splitlines():
                    print(f"  Text: {line}")
                printed_any_block = True
        elif btype == "tool_use":
            tid = block.get("id", "?")
            name = block.get("name", "?")
            print(f"  Tool Use: {name} (id={tid})")
            inp = block.get("input", {})
            if isinstance(inp, dict):
                for k, v in inp.items():
                    preview = str(v).replace("\n", " ")
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    print(f"    {k}: {preview}")
            printed_any_block = True
        elif btype == "tool_result":
            tid = block.get("tool_use_id", "?")
            err = " [ERROR]" if block.get("is_error") else ""
            print(f"  Tool Result: {tid}{err}")
            result_content = block.get("content", "")
            if isinstance(result_content, str):
                lines = result_content.splitlines()
                for line in lines[:50]:
                    print(f"    {line}")
                if len(lines) > 50:
                    print(
                        f"    ... ({len(lines) - 50} more lines)",
                    )
            else:
                print(f"    {result_content!r}")
            printed_any_block = True
    if tool_use_id_filter is not None and not printed_any_block:
        # Header was already printed; emit a blank line for visual
        # separation but otherwise stay quiet (the matching blocks
        # live in another message that will be printed separately).
        pass
    print()


def _run_transcript(registry_path: Path | None, args: argparse.Namespace) -> int:
    """Print the full session transcript for an issue or run. Idempotent.

    Read-only access to the unified
    ``~/.clawcodex/sessions/{run_id}/transcript.jsonl`` so operators
    can review a completed (or in-progress) orchestrator run without
    entering an interactive REPL.  Suitable for piping.
    """
    issue_id = getattr(args, "id", None)
    run_id = getattr(args, "run", None) or getattr(args, "run_id", None)
    if not issue_id and not run_id:
        print(
            "error: --id <issue_id> or --run <run_id> is required",
            file=sys.stderr,
        )
        return 2

    from clawcodex_ext.services.session_storage import SESSIONS_DIR

    run_id = _resolve_tail_run_id(registry_path, issue_id, run_id)
    if not run_id:
        print(
            f"No session run found for issue {issue_id or '?'} (registry has no run_id recorded).",
            file=sys.stderr,
        )
        return 1

    transcript_path = SESSIONS_DIR / run_id / "transcript.jsonl"
    if not transcript_path.exists():
        print(
            f"No transcript found at {transcript_path} for run_id {run_id}.",
            file=sys.stderr,
        )
        return 1

    role_filter = getattr(args, "role", None)
    tool_use_id_filter = getattr(args, "tool_use_id", None)
    limit = getattr(args, "limit", None)

    print(f"# Transcript for run {run_id}")
    if issue_id:
        print(f"# (issue {issue_id})")
    print(f"# Source: {transcript_path}")
    print()

    count = 0
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[transcript] warning: malformed entry: {exc}",
                    file=sys.stderr,
                )
                continue

            if role_filter and msg.get("role") != role_filter:
                continue

            if tool_use_id_filter and not _msg_references_tool(
                msg,
                tool_use_id_filter,
            ):
                continue

            _print_message(msg, tool_use_id_filter=tool_use_id_filter)
            count += 1
            if limit is not None and count >= limit:
                break

    print(f"# {count} message(s) shown")
    return 0


# ---------------------------------------------------------------------------
# issue stop
# ---------------------------------------------------------------------------


def _run_stop(
    args: argparse.Namespace,
    registry_path: Path | None = None,
    workspace_root: str | Path | None = None,
) -> int:
    """Stop a running issue agent. Idempotent — already-stopped → success."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    skip_confirm = getattr(args, "yes", False)

    # Check registry for current status (best-effort)
    current_status = None
    if registry_path and registry_path.exists():
        try:
            from extensions.orchestrator.issue_registry import IssueRegistry

            registry = IssueRegistry(registry_path)
            record = registry.get(issue_id) or registry.get_by_identifier(issue_id)
            if record is not None:
                current_status = record.status.value
        except Exception as exc:
            print(f"Warning: could not read registry: {exc}", file=sys.stderr)

    if current_status is not None:
        # RUNNING is the only status where the stop command will be effective.
        # For all other statuses the orchestrator cannot find the issue in
        # _state.running and the control file will be silently ignored.
        if current_status != "running":
            print(
                f"Warning: issue {issue_id} is not currently running (status: {current_status}).",
                file=sys.stderr,
            )
            print(
                "  The stop command will not take effect — no agent session to stop.",
                file=sys.stderr,
            )
            if not skip_confirm:
                try:
                    raw = input("  Write control file anyway? [y/N]: ")
                    if raw.strip().lower() not in ("y", "yes"):
                        print("Stop cancelled.")
                        return 0
                except (EOFError, KeyboardInterrupt):
                    print("\nStop cancelled.")
                    return 0
            else:
                print("  (use --id to target a running issue)")
    else:
        print(
            f"Warning: issue {issue_id} not found in registry — cannot verify current status.",
            file=sys.stderr,
        )
        if not skip_confirm:
            try:
                raw = input("  Write stop control file anyway? [y/N]: ")
                if raw.strip().lower() not in ("y", "yes"):
                    print("Stop cancelled.")
                    return 0
            except (EOFError, KeyboardInterrupt):
                print("\nStop cancelled.")
                return 0

    # Confirmation prompt (unless --yes is set)
    if not skip_confirm:
        try:
            raw = input(f"Stop agent for issue {issue_id}? [y/N]: ")
            if raw.strip().lower() not in ("y", "yes"):
                print("Stop cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nStop cancelled.")
            return 0

    print(f"Issue stop: sending stop command for {issue_id}")
    no_wait = getattr(args, "no_wait", False)

    sock_path = _resolve_sock_path(issue_id, workspace_root)
    if sock_path is not None and not no_wait:

        async def _do_stop() -> int:
            t0 = asyncio.get_event_loop().time()
            data = await _send_and_wait(sock_path, "stop", "", "SessionComplete", timeout=10.0)
            elapsed = asyncio.get_event_loop().time() - t0
            if data is not None:
                print(f"Agent stopped ({elapsed:.1f}s).")
                return 0
            else:
                print("Stop sent. Agent is unwinding (may take a few seconds for long-running tools).")
                return 0

        return asyncio.run(_do_stop())
    else:
        return _write_control("stop", issue_id, workspace_root=workspace_root)


# ---------------------------------------------------------------------------
# issue pause
# ---------------------------------------------------------------------------
