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
import os
import sys
import time
from pathlib import Path


def _get_status_str(status) -> str:
    """Normalize enum-backed and plain status values without importing issue_ops."""
    if hasattr(status, "value"):
        return status.value
    return str(status)


def _control_path(workspace_root: str | Path | None = None) -> Path:
    """Path to the orchestrator control directory.

    Uses the workspace root when provided (preferred), otherwise falls
    back to the CLAWCODEX_WORKSPACE_ROOT env var or ~/.clawcodex.
    """
    if workspace_root is not None:
        return Path(workspace_root) / ".orchestrator_control"
    base = Path(os.environ.get("CLAWCODEX_WORKSPACE_ROOT", Path.home() / ".clawcodex"))
    return base / ".orchestrator_control"


def _resolve_sock_path(
    issue_id: str,
    workspace_root: str | Path | None = None,
) -> Path | None:
    """Resolve the control socket path for an issue via the registry."""
    try:
        ws = Path(workspace_root) if workspace_root else None
        if ws is None:
            from extensions.orchestrator.workspace_locator import get_registry_path

            registry_path = get_registry_path()
        else:
            registry_path = ws / ".clawcodex_issue_registry.json"
        if registry_path is None or not registry_path.exists():
            return None
        from extensions.orchestrator.issue_registry import IssueRegistry

        registry = IssueRegistry(registry_path)
        record = registry.get(issue_id) or registry.get_by_identifier(issue_id)
        if record is None or not record.run_id or not record.workspace_path:
            return None
        sock_path = Path(record.workspace_path) / ".run_control" / f"{record.run_id}.sock"
        return sock_path if sock_path.exists() else None
    except Exception:
        return None


async def _send_and_wait(
    sock_path: Path,
    cmd: str,
    payload: str,
    expected_type: str,
    timeout: float = 30.0,
) -> dict | None:
    """Send a control command via socket and wait for a confirmation event.

    Opens a Unix socket connection, sends the command, then keeps the
    connection open reading event lines until one matching
    ``expected_type`` arrives. Returns the event's ``data`` dict, or
    ``None`` on timeout.
    """

    reader, writer = await asyncio.open_unix_connection(str(sock_path))
    started = asyncio.get_event_loop().time()
    try:
        # Send the command.
        writer.write(
            (json.dumps({"cmd": cmd, "payload": payload}) + "\n").encode("utf-8"),
        )
        await writer.drain()

        # Listen for the confirmation event.
        while True:
            remaining = timeout - (asyncio.get_event_loop().time() - started)
            if remaining <= 0:
                return None
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if not line:
                return None  # socket closed
            try:
                event = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if event.get("type") == expected_type:
                return event.get("data", {})
            # Ignore other event types (TextDelta, ToolCallEvent, etc.)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # nosec B110
            pass


def _write_control(cmd: str, issue_id: str, extra: str = "", workspace_root: str | Path | None = None) -> int:
    """Send a control command, preferring the Unix socket for near-real-time
    delivery. Falls back to the control-file mechanism (picked up on the
    orchestrator's next poll cycle) when the socket is unavailable.

    Socket-first delivery eliminates the 30s poll-cycle
    latency for ``pause`` / ``resume`` / ``stop`` when the agent is
    running and the control socket is alive.
    """

    # Try to resolve run_id + workspace_path from the
    # registry so we can attempt a direct socket connection.
    # Only agent-level commands (pause/resume/stop) go through the
    # socket; orchestrator-level commands (retry/rebase/etc.) always
    # go through the control file.
    _SOCKET_CMDS = {"pause", "resume", "stop"}
    if cmd in _SOCKET_CMDS and workspace_root is not None:
        try:
            registry_path = Path(workspace_root) / ".clawcodex_issue_registry.json"
            if registry_path.exists():
                from extensions.orchestrator.issue_registry import IssueRegistry

                registry = IssueRegistry(registry_path)
                record = registry.get(issue_id) or registry.get_by_identifier(issue_id)
                if record is not None and record.run_id and record.workspace_path:
                    sock_path = Path(record.workspace_path) / ".run_control" / f"{record.run_id}.sock"
                    if sock_path.exists():

                        async def _send_via_socket() -> None:
                            _reader, writer = await asyncio.open_unix_connection(
                                str(sock_path),
                            )
                            try:
                                payload = {"cmd": cmd, "payload": extra}
                                writer.write(
                                    (json.dumps(payload) + "\n").encode("utf-8"),
                                )
                                await writer.drain()
                            finally:
                                writer.close()
                                try:
                                    await writer.wait_closed()
                                except Exception:  # nosec B110
                                    pass

                        asyncio.run(_send_via_socket())
                        print(f"Control command '{cmd}' sent for issue {issue_id} (via socket)")
                        print("  The agent will process this at the next tool-result boundary.")
                        return 0
        except Exception:  # nosec B110
            pass  # Fall through to control-file path.

    # Fallback: write a control file for the orchestrator's next poll.
    control_dir = _control_path(workspace_root=workspace_root)
    control_dir.mkdir(parents=True, exist_ok=True)

    control_file = control_dir / f"{cmd}_{issue_id}.control"
    payload = f"{cmd}\n{issue_id}\n{extra}\n"
    try:
        control_file.write_text(payload, encoding="utf-8")
        print(f"Control command '{cmd}' sent for issue {issue_id} (via control file)")
        print("  The orchestrator will pick this up on its next poll cycle.")
        return 0
    except Exception as exc:
        print(f"Failed to send '{cmd}' for issue {issue_id}: {exc}", file=sys.stderr)
        return 1


def _try_socket_inject(issue_id: str, hint: str) -> bool:
    """Try to send an inject command via the control socket.

    Returns ``True`` if the hint was queued via the socket (which
    routes to ``queue_pending_message`` for real-time delivery at
    the next ToolResult boundary). Returns ``False`` if the socket
    is unavailable — the caller should fall back to file-based inject.

    CLI ``issue inject`` prefers socket delivery for
    near-real-time inject, matching the socket ``inject`` command.
    """
    try:
        from extensions.orchestrator.workspace_locator import get_registry_path

        registry_path = get_registry_path()
        if registry_path is None or not registry_path.exists():
            return False
        from extensions.orchestrator.issue_registry import IssueRegistry

        registry = IssueRegistry(registry_path)
        record = registry.get(issue_id) or registry.get_by_identifier(issue_id)
        if record is None or not record.run_id or not record.workspace_path:
            return False
        sock_path = Path(record.workspace_path) / ".run_control" / f"{record.run_id}.sock"
        if not sock_path.exists():
            return False

        async def _send() -> None:
            _reader, writer = await asyncio.open_unix_connection(str(sock_path))
            try:
                writer.write(
                    (json.dumps({"cmd": "inject", "payload": hint}) + "\n").encode("utf-8"),
                )
                await writer.drain()
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:  # nosec B110
                    pass

        asyncio.run(_send())
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# issue list
# ---------------------------------------------------------------------------


def _run_list(registry_path: Path | None, args: argparse.Namespace) -> int:
    """List all issues with status. Idempotent — pure read."""
    if not registry_path or not registry_path.exists():
        ws = getattr(args, "workspace", None)
        wf = getattr(args, "workflow", None)

        # When neither --workspace nor --workflow is passed, check for multiple live orch entries
        if not ws and not wf:
            from extensions.orchestrator.workspace_locator import (
                get_live_projects,
                print_multi_project_hint,
            )

            live = get_live_projects()
            if len(live) > 1:
                print_multi_project_hint(live, "orchestrator issue list")
                return 0

        from extensions.orchestrator.workspace_locator import (
            get_workspace_root,
            list_orchestrator_projects,
        )

        workspace_root = get_workspace_root(workspace_arg=ws, workflow_path=wf)
        projects = list_orchestrator_projects()

        if workspace_root and projects:
            p = projects[0]
            pid = p.get("pid", "?")
            print(f"Orchestrator is running (PID {pid}, {p.get('project_slug', '?')})")
            print(f"Workspace: {workspace_root}")
            print("No issues processed yet.")
        else:
            print("No orchestrator registry found. No issues to list.")
            print("Hint: Start with 'clawcodex orchestrator server start --workflow WORKFLOW.md'")
        return 0  # idempotent: no-issues is a valid state

    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    counts: dict[str, int] = {
        "PENDING": 0,
        "RUNNING": 0,
        "SYNCED": 0,
        "COMPLETED": 0,
        "FAILED": 0,
        "ABANDONED": 0,
    }
    records = list(registry._records.values())

    # Filter by status
    status_filter = getattr(args, "status", None)
    if status_filter:
        records = [r for r in records if _get_status_str(r.status) == status_filter]

    if not records:
        print("No issues found.")
        if status_filter:
            print(f"  (filtered by status: {status_filter})")
        return 0

    # Status display mapping matching README Demo format
    _STATUS_DISPLAY = {
        "completed": "done",
        "pending_review": "paused",
        "running": "running",
        "pending": "pending",
        "synced": "synced",
        "failed": "failed",
        "abandoned": "abandoned",
        "verification_failed": "vfailed",
    }

    print(f"{'ID':<20} {'STATUS':<10} {'BRANCH':<25} {'ATTEMPTS':<9} PR")
    for r in records:
        raw_status = _get_status_str(r.status)
        display_status = _STATUS_DISPLAY.get(raw_status, raw_status)
        branch = r.branch_name or "-"
        attempts = str(r.attempt_count) if r.attempt_count else "-"
        pr = r.pr_url or "-"
        print(f"{r.issue_id:<20} {display_status:<10} {branch:<25} {attempts:<9} {pr}")

    print()
    for r in records:
        s = _get_status_str(r.status)
        counts[s.upper()] = counts.get(s.upper(), 0) + 1
    print(f"  PENDING  : {counts.get('PENDING', 0)}")
    print(f"  RUNNING  : {counts.get('RUNNING', 0)}")
    print(f"  SYNCED   : {counts.get('SYNCED', 0)}")
    print(f"  COMPLETED: {counts.get('COMPLETED', 0)}")
    print(f"  FAILED   : {counts.get('FAILED', 0)}")
    print(f"  ABANDONED: {counts.get('ABANDONED', 0)}")
    return 0


# ---------------------------------------------------------------------------
# issue show
# ---------------------------------------------------------------------------


def _run_show(registry_path: Path | None, args: argparse.Namespace) -> int:
    """Show details for a specific issue. Idempotent — pure read."""
    issue_id = getattr(args, "id", None) or getattr(args, "issue_id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    if not registry_path or not registry_path.exists():
        print(f"No registry found. Cannot show issue {issue_id}.", file=sys.stderr)
        return 1

    from extensions.orchestrator.issue_registry import IssueRegistry

    registry = IssueRegistry(registry_path)
    record = registry.get_by_issue_ref(issue_id)
    if record is None:
        print(f"Issue {issue_id} not found in registry.", file=sys.stderr)
        return 1

    created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created_at))
    updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.updated_at))

    print(f"Issue: {record.issue_id}")
    print(f"  Identifier     : {record.issue_identifier}")
    print(f"  Status         : {record.status.value}")
    print(f"  Branch         : {record.branch_name or '-'}")
    print(f"  Base Branch    : {record.base_branch or 'main'}")
    print(f"  Commit SHA     : {record.commit_sha or '-'}")
    print(f"  PR Number      : {record.pr_number or '-'}")
    print(f"  PR URL         : {record.pr_url or '-'}")
    pr_created = getattr(record, "pr_created_at", None)
    if pr_created:
        pr_created_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pr_created))
        # Time from issue claim to first PR creation — the orchestrator's
        # key "issue → PR" latency metric for leadership reporting.
        latency_s = pr_created - record.created_at
        print(f"  PR Created     : {pr_created_text}")
        print(f"  Issue→PR Time  : {latency_s:.0f}s")
    else:
        print("  PR Created     : -")
    print(f"  Attempts       : {record.attempt_count}")
    print(f"  Run ID         : {getattr(record, 'run_id', None) or '-'}")
    print(f"  Turns / Tools  : {getattr(record, 'run_turn_count', 0)} / {getattr(record, 'run_tool_count', 0)}")
    print(f"  Last Event     : {getattr(record, 'run_last_event', None) or '-'}")
    print(f"  Last Tool      : {getattr(record, 'run_last_tool', None) or '-'}")
    print(f"  Output Chars   : {getattr(record, 'run_output_len', 0)}")
    deadline = getattr(record, "run_timeout_deadline_at", None)
    if deadline:
        deadline_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline))
    else:
        deadline_text = "-"
    print(f"  Timeout By     : {deadline_text}")
    workspace_dirty = getattr(record, "run_workspace_dirty", None)
    dirty_text = "-" if workspace_dirty is None else str(workspace_dirty).lower()
    print(f"  Workspace Dirty: {dirty_text}")
    print(f"  Workspace Path : {record.workspace_path or '-'}")
    print(f"  Debug Log      : {getattr(record, 'debug_log_path', None) or '-'}")
    print(f"  Created        : {created}")
    print(f"  Updated        : {updated}")
    if record.clarification_status:
        print(f"  Clarification  : {record.clarification_status}")
    return 0


def _resolve_issue_workspace_path(issue_id: str) -> Path | None:
    """Resolve an issue workspace, including sequential registry layouts."""
    from extensions.orchestrator.workspace_locator import get_registry_path, get_workspace_root

    workspace_root = get_workspace_root(workspace_arg=os.environ.get("CLAWCODEX_WORKSPACE_ROOT"))
    registry_path = get_registry_path(workspace_arg=str(workspace_root)) if workspace_root else None
    if registry_path and registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            record = registry.get(issue_id)
            if record:
                root = Path(record.get("workspace_path") or workspace_root)
                candidates = []
                identifier = record.get("issue_identifier")
                if identifier:
                    candidates.append(root / identifier)
                candidates.append(root)
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
        except Exception:  # nosec B110
            pass

    base = workspace_root or Path.home() / ".clawcodex" / "workspace"
    if not base.exists():
        return None
    for wd in base.iterdir():
        if not wd.is_dir():
            continue
        metadata_file = wd / ".metadata"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                if metadata.get("issue_id") == issue_id:
                    return wd
            except Exception:  # nosec B110
                pass
        if wd.name == issue_id or issue_id in wd.name:
            return wd
    return None


# ---------------------------------------------------------------------------
# issue tail
# ---------------------------------------------------------------------------


def _resolve_tail_run_id(
    registry_path: Path | None,
    issue_id: str | None,
    run_id: str | None,
) -> str | None:
    """Resolve which session run to tail.

    Priority: explicit ``--run <run_id>`` wins; otherwise look up
    the most recent ``run_id`` for ``--id <issue_id>`` via the
    issue registry.  Returns ``None`` if no run can be determined.
    """
    if run_id:
        return run_id
    if not issue_id or not registry_path or not registry_path.exists():
        return None
    try:
        from extensions.orchestrator.issue_registry import IssueRegistry

        registry = IssueRegistry(registry_path)
        record = registry.get(issue_id)
        if record is None:
            record = registry.get_by_identifier(issue_id)
        if record is None:
            return None
        return record.run_id
    except Exception:
        return None


def _run_tail(registry_path: Path | None, args: argparse.Namespace) -> int:
    """Tail a session transcript for an issue or run. Idempotent — pure read.

    Unified storage: headless agent and REPL sessions both
    write to ``~/.clawcodex/sessions/{run_id}/transcript.jsonl``
    via :class:`SessionStorage`.  This command tails that file and
    renders tool calls / tool results / assistant text the same
    way the legacy ``.event_logs/{issue_id}.ndjson`` reader did.
    """
    issue_id = getattr(args, "id", None) or getattr(args, "issue_id", None)
    run_id = getattr(args, "run", None) or getattr(args, "run_id", None)
    if not issue_id and not run_id:
        print("error: --id <issue_id> or --run <run_id> is required", file=sys.stderr)
        return 2

    run_id = _resolve_tail_run_id(registry_path, issue_id, run_id)
    if not run_id:
        print(
            f"No session run found for issue {issue_id or '?'} (registry has no run_id recorded).",
            file=sys.stderr,
        )
        return 1

    from clawcodex_ext.services.session_storage import SESSIONS_DIR

    transcript_path = SESSIONS_DIR / run_id / "transcript.jsonl"
    if not transcript_path.exists():
        print(
            f"No transcript found at {transcript_path} for run_id {run_id}.",
            file=sys.stderr,
        )
        return 1

    label = f"run {run_id}" if not issue_id else f"issue {issue_id} (run {run_id})"
    print(f"Tailing transcript for {label} (Ctrl+C to stop)...")
    from .issue_transcript import _render_message

    try:
        last_size = transcript_path.stat().st_size
        pending = ""
        turn_counter = 0
        pending_calls: dict[str, dict] = {}
        while True:
            current_size = transcript_path.stat().st_size
            if current_size <= last_size:
                # Flush stale pending calls every 5 seconds
                time.sleep(0.5)
                continue

            with open(transcript_path, "r", encoding="utf-8") as f:
                f.seek(last_size)
                chunk = f.read()

            lines = (pending + chunk).splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                pending = lines.pop()
            else:
                pending = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"[tail] warning: malformed entry in {transcript_path}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                _render_message(msg, turn_counter, pending_calls)
            last_size = current_size
    except KeyboardInterrupt:
        print("\n[tail] stopped")
    except Exception as exc:
        print(f"[tail] error: {exc}", file=sys.stderr)
        return 1
    return 0
