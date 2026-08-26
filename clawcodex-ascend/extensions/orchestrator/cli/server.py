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

# pylint: disable=too-many-nested-blocks

"""orchestrator server — manage the orchestrator daemon process.

Usage (noun-verb):
  clawcodex orchestrator server status                                   Show orchestrator daemon status
  clawcodex orchestrator server stop                                     Stop the orchestrator daemon gracefully
  clawcodex orchestrator server start [--workflow PATH]                  Start the orchestrator daemon
  clawcodex orchestrator server start [--workflow PATH]                  Start with declarative workflow engine
                                       [--workflow-yaml PATH]

All commands are idempotent:
  - status: pure read, always safe
  - stop: stopping an already-stopped daemon succeeds silently
  - start: starting an already-running daemon shows its status and exits 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_server_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``server`` sub-subcommands (status | stop | start)."""
    server_parser = subparsers.add_parser(
        "server",
        help="Manage the orchestrator daemon process",
        description="Start, stop, or check the status of the orchestrator daemon. "
        "All commands are idempotent — running them multiple times "
        "has no ill effect.",
    )
    server_sub = server_parser.add_subparsers(
        dest="server_subcommand",
        required=True,
    )

    # --- server status ---
    status_parser = server_sub.add_parser(
        "status",
        help="Show orchestrator daemon status",
        description="Display whether the orchestrator daemon is running, its PID, "
        "uptime, workspace root, and project slug. Idempotent (pure read).",
    )
    status_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path (optional auto-detection override)",
    )
    status_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (helps resolve workspace when metadata is missing)",
    )

    # --- server stop ---
    stop_parser = server_sub.add_parser(
        "stop",
        help="Stop the orchestrator daemon gracefully",
        description="Send SIGTERM to the orchestrator process and clean up metadata. "
        "Idempotent: if the daemon is already stopped, exits 0 silently.",
    )
    stop_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path (optional auto-detection override)",
    )
    stop_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (helps resolve workspace when metadata is missing)",
    )
    stop_parser.add_argument(
        "--force",
        action="store_true",
        help="Use SIGKILL instead of SIGTERM (force immediate termination)",
    )
    stop_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds to wait after SIGTERM before SIGKILL (default: 5.0)",
    )
    stop_parser.add_argument(
        "--all",
        action="store_true",
        help="Stop all running orchestrator daemons and clean up all stale metadata. "
        "Useful after test suites or when multiple workflows were started.",
    )

    # --- server start ---
    start_parser = server_sub.add_parser(
        "start",
        help="Start the orchestrator daemon",
        description="Launch the orchestrator with a workflow file. "
        "Optionally enable the declarative workflow engine via --workflow-yaml "
        "for multi-stage DAG execution with quality gates and decision branches.",
        epilog="Examples:\n"
        "  clawcodex orchestrator server start --workflow ./workflow.md\n"
        "  clawcodex orchestrator server start --workflow ./workflow.md --workflow-yaml ./workflow.yaml\n"
        "  clawcodex orchestrator server start --workflow ./workflow.md --workflow-yaml ./workflow.yaml --dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    start_parser.add_argument(
        "--workflow",
        type=str,
        required=False,
        metavar="PATH",
        help="Path to WORKFLOW.md file",
    )
    start_parser.add_argument(
        "--workflow-yaml",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to workflow.yaml for the declarative workflow engine",
    )
    start_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show embedded status dashboard",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="LiveView dashboard port",
    )
    start_parser.add_argument(
        "--gateway",
        dest="gateway",
        action="store_true",
        help="Opt into all supported direct/private messages via the IM gateway",
    )
    start_parser.add_argument(
        "--im-gateway",
        dest="gateway",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    start_parser.add_argument(
        "--gateway-origin",
        dest="gateway_origin",
        type=str,
        default=None,
        metavar="ORIGIN",
        help=("Advanced: opt into the IM gateway for a specific origin, e.g. wechat:direct:default:user_id"),
    )
    start_parser.add_argument(
        "--im-gateway-origin",
        dest="gateway_origin",
        type=str,
        default=None,
        metavar="ORIGIN",
        help=argparse.SUPPRESS,
    )
    start_parser.add_argument(
        "--gateway-sock",
        dest="gateway_sock",
        type=str,
        default=None,
        metavar="PATH",
        help=("Gateway daemon Unix socket for --gateway-origin (default: ~/.clawcodex/gateway/gateway.sock)"),
    )
    start_parser.add_argument(
        "--im-gateway-sock",
        dest="gateway_sock",
        type=str,
        default=None,
        metavar="PATH",
        help=argparse.SUPPRESS,
    )

    # --- server connect-gateway ---
    connect_parser = server_sub.add_parser(
        "connect-gateway",
        help="Ask a running daemon to connect to the IM gateway",
        description=(
            "Submit an IM gateway connect request to an already-running orchestrator daemon. "
            "The daemon handles the request on its next control-file poll."
        ),
    )
    connect_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path (optional auto-detection override)",
    )
    connect_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (helps resolve workspace when metadata is missing)",
    )
    connect_parser.add_argument(
        "--gateway",
        dest="gateway",
        type=str,
        default=None,
        metavar="ORIGIN",
        help=(
            "Optional specific origin to bind, e.g. wechat:direct:default:user_id. "
            "Omit for all supported direct/private IM messages."
        ),
    )
    connect_parser.add_argument(
        "--im-gateway",
        dest="gateway",
        type=str,
        default=None,
        metavar="ORIGIN",
        help=argparse.SUPPRESS,
    )
    connect_parser.add_argument(
        "--gateway-sock",
        dest="gateway_sock",
        type=str,
        default=None,
        metavar="PATH",
        help=("Gateway daemon Unix socket for connect-gateway (default: ~/.clawcodex/gateway/gateway.sock)"),
    )
    connect_parser.add_argument(
        "--im-gateway-sock",
        dest="gateway_sock",
        type=str,
        default=None,
        metavar="PATH",
        help=argparse.SUPPRESS,
    )

    # --- server disconnect-gateway ---
    disconnect_parser = server_sub.add_parser(
        "disconnect-gateway",
        help="Ask a running daemon to disconnect from the IM gateway",
        description=(
            "Submit an IM gateway disconnect request to an already-running orchestrator daemon. "
            "The daemon handles the request on its next control-file poll."
        ),
    )
    disconnect_parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit workspace root path (optional auto-detection override)",
    )
    disconnect_parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to WORKFLOW.md (helps resolve workspace when metadata is missing)",
    )


# ---------------------------------------------------------------------------
# Run dispatch
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate server subcommand."""
    # Late import: server_gateway ships in a parallel PR; defer so this
    # module loads even if that file is not merged yet.
    from .server_gateway import _run_connect_gateway, _run_disconnect_gateway, _run_start

    cmd = args.server_subcommand
    if cmd == "status":
        return _run_status(args)
    elif cmd == "stop":
        return _run_stop(args)
    elif cmd == "start":
        return _run_start(args)
    elif cmd == "connect-gateway":
        return _run_connect_gateway(args)
    elif cmd == "disconnect-gateway":
        return _run_disconnect_gateway(args)
    print(f"error: unknown server subcommand '{cmd}'", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _find_metadata(args: argparse.Namespace) -> tuple[Path | None, dict | None]:  # pylint: disable=too-many-nested-blocks
    """Resolve orchestrator metadata.

    Returns (metadata_path, metadata_dict) or (None, None) if not found.
    """
    from extensions.orchestrator.workspace_locator import (
        _find_latest_metadata,
        get_workspace_root,
    )

    # 0. Multi-project ambiguity detection: prompt when no explicit args and multiple live projects
    if not getattr(args, "workspace", None) and not getattr(args, "workflow", None):
        from extensions.orchestrator.workspace_locator import (
            get_live_projects,
            print_multi_project_hint,
        )

        live = get_live_projects()
        if len(live) > 1:
            subcmd = getattr(args, "server_subcommand", "server")
            print_multi_project_hint(live, f"orchestrator server {subcmd}")
            return None, None

    # Priority: explicit --workspace > --workflow > env var > latest metadata
    workspace_root = get_workspace_root(
        workspace_arg=getattr(args, "workspace", None),
        workflow_path=getattr(args, "workflow", None),
    )
    if workspace_root:
        slug = _slug_from_workspace(str(workspace_root))
        metadata_path = Path.home() / ".clawcodex" / "orchestrator" / slug / "metadata.json"
        if metadata_path.exists():
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                return metadata_path, data
            except Exception:  # nosec B110
                pass
        # Fallback: search by workspace_root matching
        metadata_dir = Path.home() / ".clawcodex" / "orchestrator"
        if metadata_dir.exists():
            for md_dir in metadata_dir.iterdir():
                mf = md_dir / "metadata.json"
                if mf.exists():
                    try:
                        data = json.loads(mf.read_text(encoding="utf-8"))
                        if data.get("workspace_root") == str(workspace_root):
                            return mf, data
                    except Exception:  # nosec B110
                        pass

    # Fallback: latest metadata (only when no explicit --workspace/--workflow)
    has_explicit = getattr(args, "workspace", None) or getattr(args, "workflow", None)
    if not has_explicit:
        latest = _find_latest_metadata()
        if latest and latest.exists():
            try:
                data = json.loads(latest.read_text(encoding="utf-8"))
                return latest, data
            except Exception:  # nosec B110
                pass

    return None, None


def _slug_from_workspace(ws_str: str) -> str:
    """Generate a deterministic slug from a workspace path string."""
    parts = [
        p
        for p in ws_str.strip().replace("/", "-").replace("\\", "-").split("-")
        if p and p not in ("tmp", ".clawcodex", "~")
    ]
    return "-".join(parts[-3:]) if parts else "default"


def _is_pid_alive(pid: int) -> bool:
    """Check whether a PID is still alive (no-side-effect signal 0 test)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user; treat as alive
        return True
    except OSError:
        # Other OSError: be conservative and treat as alive
        return True


def _format_uptime(started_at: float) -> str:
    """Format uptime as human-readable string."""
    elapsed = time.time() - started_at
    if elapsed < 60:
        return f"{int(elapsed)}s"
    elif elapsed < 3600:
        return f"{int(elapsed / 60)}m {int(elapsed % 60)}s"
    else:
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
        return f"{hours}h {minutes}m"


# ---------------------------------------------------------------------------
# server status
# ---------------------------------------------------------------------------


def _run_status(args: argparse.Namespace) -> int:
    """Show orchestrator daemon status. Idempotent — pure read."""
    meta_path, meta = _find_metadata(args)

    if meta is None:
        print("Orchestrator daemon: NOT RUNNING")
        print("  No orchestrator metadata found.")
        print("  Hint: Start with 'clawcodex orchestrator server start --workflow WORKFLOW.md'")
        return 0  # idempotent: not-running is a valid status, not an error

    pid = meta.get("pid")
    started_at = meta.get("started_at", 0)
    project_slug = meta.get("project_slug", "unknown")
    workspace_root = meta.get("workspace_root", "unknown")
    workflow_path = meta.get("workflow_path")

    if pid and _is_pid_alive(pid):
        uptime = _format_uptime(started_at) if started_at else "unknown"
        print("Orchestrator daemon: RUNNING")
        print(f"  PID            : {pid}")
        print(f"  Uptime         : {uptime}")
        print(f"  Project        : {project_slug}")
        print(f"  Workspace root : {workspace_root}")
        if workflow_path:
            print(f"  Workflow       : {workflow_path}")
        print(f"  Metadata       : {meta_path}")
    else:
        stale_age = _format_uptime(started_at) if started_at else "unknown"
        print(f"Orchestrator daemon: STOPPED (stale metadata from {stale_age} ago)")
        print(f"  Project        : {project_slug}")
        print(f"  Workspace root : {workspace_root}")
        print(f"  Metadata       : {meta_path} (stale — clean up with 'server stop')")

    return 0


# ---------------------------------------------------------------------------
# server stop
# ---------------------------------------------------------------------------


def _stop_daemon_process(pid: int, sig: int, timeout: float, force: bool, meta_path: Path) -> bool:
    """Send a stop signal, wait for exit, and clean metadata. Returns True on success."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return True
    except PermissionError:
        print(f"  Permission denied: cannot signal PID {pid}.", file=sys.stderr)
        return False

    if not force:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not _is_pid_alive(pid):
                break
            time.sleep(0.2)
        else:
            # Timed out — process still alive; keep metadata for later retry
            print(f"  Process did not exit within {timeout}s timeout. Use --force for SIGKILL.")
            return False
    else:
        # Brief pause so SIGKILL takes effect
        time.sleep(0.3)

    try:
        meta_path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        print(f"  failed to clean metadata: {exc}")
        return False


def _run_stop_all(args: argparse.Namespace) -> int:
    """Stop all running orchestrator daemons and clean up all stale metadata.

    Iterates every metadata file under ``~/.clawcodex/orchestrator/*/metadata.json``.
    - Live PIDs → send signal (SIGTERM / SIGKILL) and wait for graceful exit.
    - Dead PIDs → clean up stale metadata immediately.
    """
    orchestrator_dir = Path.home() / ".clawcodex" / "orchestrator"
    if not orchestrator_dir.exists():
        print("No orchestrator metadata directory found — nothing to stop.")
        return 0

    metadata_files: list[tuple[Path, dict]] = []
    for md_dir in orchestrator_dir.iterdir():
        if not md_dir.is_dir():
            continue
        mf = md_dir / "metadata.json"
        if not mf.exists():
            continue
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            metadata_files.append((mf, data))
        except Exception:
            continue

    if not metadata_files:
        print("No orchestrator metadata found — nothing to stop.")
        return 0

    sig = signal.SIGKILL if args.force else signal.SIGTERM  # pylint: disable=no-member
    sig_name = "SIGKILL" if args.force else "SIGTERM"
    timeout = args.timeout

    stopped = 0
    cleaned = 0
    errors = 0

    print(f"Stopping all orchestrator daemons ({len(metadata_files)} metadata files found)...")
    print()

    for meta_path, meta in metadata_files:
        pid = meta.get("pid")
        slug = meta.get("project_slug", meta_path.parent.name)
        ws = meta.get("workspace_root", "?")

        if pid is None or not _is_pid_alive(pid):
            pid_str = pid or "N/A"
            print(f"  [{slug}] already stopped (PID {pid_str}) — cleaning up stale metadata")
            try:
                meta_path.unlink(missing_ok=True)
                cleaned += 1
            except OSError as exc:
                print(f"    ⚠ failed to clean metadata: {exc}")
                errors += 1
            continue

        # Send signal, wait for exit, clean metadata (shared logic)
        print(f"  [{slug}] stopping daemon (PID {pid}, workspace: {ws})...")
        print(f"    Sending {sig_name}...")
        if _stop_daemon_process(pid, sig, timeout, args.force, meta_path):
            stopped += 1
        else:
            errors += 1

    print()
    print(f"Done: {stopped} stopped, {cleaned} stale cleaned, {errors} error(s).")
    return 1 if errors else 0


def _run_stop(args: argparse.Namespace) -> int:
    """Stop the orchestrator daemon. Idempotent — already-stopped → exit 0."""
    if getattr(args, "all", False):
        return _run_stop_all(args)

    meta_path, meta = _find_metadata(args)

    if meta is None:
        print("Orchestrator daemon: already stopped (no metadata found)")
        return 0  # idempotent

    pid = meta.get("pid")
    project_slug = meta.get("project_slug", "unknown")

    if pid is None or not _is_pid_alive(pid):
        print(f"Orchestrator daemon: already stopped (PID {pid or 'N/A'} not running)")
        # Clean up stale metadata
        if meta_path and meta_path.exists():
            meta_path.unlink()
            print("  Stale metadata cleaned up.")
        return 0  # idempotent

    # Send stop signal
    sig = signal.SIGKILL if args.force else signal.SIGTERM  # pylint: disable=no-member
    sig_name = "SIGKILL" if args.force else "SIGTERM"
    timeout = args.timeout
    print(f"Stopping orchestrator daemon (PID {pid}, project: {project_slug})...")
    print(f"  Sending {sig_name}...")

    if not _stop_daemon_process(pid, sig, timeout, args.force, meta_path):
        print(f"  You may also kill manually: kill -9 {pid}")
        return 1

    print(f"  Metadata cleaned up: {meta_path}")

    print("Orchestrator daemon stopped.")
    return 0
