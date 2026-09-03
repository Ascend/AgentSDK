#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""orchestrator workspace — manage preserved workspaces.

Usage (noun-verb):

  clawcodex orchestrator workspace list [--status completed|failed|abandoned|all]
  clawcodex orchestrator workspace show --id <issue-id>
  clawcodex orchestrator workspace cd --id <issue-id>
  clawcodex orchestrator workspace cleanup --id <issue-id> [--force]
  clawcodex orchestrator workspace cleanup --all-completed [--force]
  clawcodex orchestrator workspace verify --id <issue-id>

Design principles:
  - ``list`` / ``show`` / ``cd`` are pure reads (idempotent)
  - ``cleanup`` is destructive (requires --force or confirmation)
  - ``verify`` runs the auto-generated verify.sh in the workspace
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_workspace_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``workspace`` sub-subcommands."""
    ws_parser = subparsers.add_parser(
        "workspace",
        help="Manage preserved workspaces (list, show, cd, cleanup, verify)",
        description="List, inspect, enter, clean up, or verify preserved workspaces. "
        "Preserved workspaces are retained after issue completion for manual "
        "inspection and verification.",
    )
    ws_sub = ws_parser.add_subparsers(dest="workspace_subcommand", required=True)

    # --- workspace list ---
    list_p = ws_sub.add_parser(
        "list",
        help="List all preserved workspaces",
    )
    list_p.add_argument(
        "--status",
        choices=["completed", "failed", "abandoned", "all"],
        default="all",
        help="Filter by issue status (default: all)",
    )
    list_p.add_argument("--workspace", type=str, default=None, metavar="PATH")
    list_p.add_argument("--workflow", type=str, default=None, metavar="PATH")

    # --- workspace show ---
    show_p = ws_sub.add_parser(
        "show",
        help="Show details for a specific workspace",
    )
    show_p.add_argument("--id", type=str, required=True, metavar="ISSUE_ID")
    show_p.add_argument("--workspace", type=str, default=None, metavar="PATH")
    show_p.add_argument("--workflow", type=str, default=None, metavar="PATH")

    # --- workspace cd ---
    cd_p = ws_sub.add_parser(
        "cd",
        help="Print workspace path for shell cd integration",
        description="Usage: cd $(clawcodex-dev orchestrator workspace cd --id X)",
    )
    cd_p.add_argument("--id", type=str, required=True, metavar="ISSUE_ID")
    cd_p.add_argument("--workspace", type=str, default=None, metavar="PATH")
    cd_p.add_argument("--workflow", type=str, default=None, metavar="PATH")

    # --- workspace cleanup ---
    cleanup_p = ws_sub.add_parser(
        "cleanup",
        help="Remove preserved workspaces",
    )
    cleanup_p.add_argument("--id", type=str, default=None, metavar="ISSUE_ID")
    cleanup_p.add_argument(
        "--all-completed",
        action="store_true",
        help="Remove all workspaces for completed issues",
    )
    cleanup_p.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    cleanup_p.add_argument("--workspace", type=str, default=None, metavar="PATH")
    cleanup_p.add_argument("--workflow", type=str, default=None, metavar="PATH")

    # --- workspace verify ---
    verify_p = ws_sub.add_parser(
        "verify",
        help="Run verify.sh in a preserved workspace",
    )
    verify_p.add_argument("--id", type=str, required=True, metavar="ISSUE_ID")
    verify_p.add_argument("--workspace", type=str, default=None, metavar="PATH")
    verify_p.add_argument("--workflow", type=str, default=None, metavar="PATH")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate workspace subcommand."""
    cmd = args.workspace_subcommand

    from extensions.orchestrator.workspace_locator import resolve_for_cli

    workspace_root, registry_path = resolve_for_cli(
        getattr(args, "workspace", None),
        getattr(args, "workflow", None),
    )

    if cmd == "list":
        return _cmd_list(workspace_root, registry_path, args)
    elif cmd == "show":
        return _cmd_show(workspace_root, registry_path, args)
    elif cmd == "cd":
        return _cmd_cd(workspace_root, registry_path, args)
    elif cmd == "cleanup":
        return _cmd_cleanup(workspace_root, registry_path, args)
    elif cmd == "verify":
        return _cmd_verify(workspace_root, registry_path, args)

    print(f"error: unknown workspace subcommand '{cmd}'", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_registry(registry_path: Path | None) -> Any:
    """Load issue registry, return None if unavailable."""
    if not registry_path or not registry_path.exists():
        return None
    from extensions.orchestrator.issue_registry import IssueRegistry

    return IssueRegistry(registry_path)


def _find_workspace_path(workspace_root: Path | None, issue_id: str) -> Path | None:
    """Find workspace directory for an issue by scanning workspace root."""
    if not workspace_root or not workspace_root.exists():
        return None
    # Try direct match: workspace_root / <safe_identifier>
    for entry in workspace_root.iterdir():
        if entry.is_dir() and issue_id in entry.name:
            manifest = entry / ".orchestrator_workspace" / ".workspace_preserved.json"
            if manifest.exists():
                return entry
    return None


def _find_workspace_from_registry(
    workspace_root: Path | None,
    registry: Any,
    issue_id: str,
) -> Path | None:
    """Find workspace path from registry record, then verify it exists."""
    if registry is None:
        return None
    record = registry.get_by_issue_ref(issue_id)
    if record is None or not record.workspace_path:
        return None
    ws = Path(record.workspace_path)
    if ws.exists():
        return ws
    # Fallback: try workspace_root / identifier
    if workspace_root:
        from extensions.orchestrator.workspace import _safe_identifier

        safe = _safe_identifier(record.issue_identifier)
        candidate = workspace_root / safe
        if candidate.exists():
            return candidate
    return None


def _resolve_workspace(
    workspace_root: Path | None,
    registry_path: Path | None,
    issue_id: str,
) -> tuple[Path | None, Any]:
    """Resolve workspace path for an issue. Returns (path, registry_record)."""
    registry = _load_registry(registry_path)
    ws_path = _find_workspace_from_registry(workspace_root, registry, issue_id)
    if ws_path is None:
        ws_path = _find_workspace_path(workspace_root, issue_id)
    record = None
    if registry is not None:
        record = registry.get_by_issue_ref(issue_id)
    return ws_path, record


def _read_manifest(ws_path: Path) -> dict[str, Any]:
    """Read .workspace_preserved.json from a workspace."""
    manifest_path = ws_path / ".orchestrator_workspace" / ".workspace_preserved.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# workspace list
# ---------------------------------------------------------------------------


def _cmd_list(
    workspace_root: Path | None,
    registry_path: Path | None,
    args: argparse.Namespace,
) -> int:
    """List all preserved workspaces. Idempotent — pure read."""
    if not workspace_root or not workspace_root.exists():
        print("No workspace root found.", file=sys.stderr)
        return 1

    registry = _load_registry(registry_path)
    status_filter = getattr(args, "status", "all")

    # Scan workspace root for preserved directories
    preserved: list[tuple[Path, dict[str, Any], Any]] = []
    for entry in sorted(workspace_root.iterdir()):
        if not entry.is_dir():
            continue
        manifest = entry / ".orchestrator_workspace" / ".workspace_preserved.json"
        if not manifest.exists():
            continue
        m = _read_manifest(entry)
        # Cross-reference with registry
        record = None
        if registry is not None:
            iid = m.get("issue_id") or m.get("identifier", "")
            record = registry.get_by_issue_ref(str(iid))
        # Apply status filter
        if status_filter and status_filter != "all":
            r_status = ""
            if record is not None:
                r_status = record.status.value
            elif m.get("end_status"):
                r_status = m["end_status"]
            if not r_status or r_status.lower() != status_filter.lower():
                continue
        preserved.append((entry, m, record))

    if not preserved:
        print("No preserved workspaces found.")
        if status_filter != "all":
            print(f"  (filtered by status: {status_filter})")
        return 0

    print(f"Preserved Workspaces ({len(preserved)} total)")
    print(f"  {'ISSUE':<20} {'STATUS':<18} {'BRANCH':<25} {'PATH'}")
    print(f"  {'-' * 20} {'-' * 18} {'-' * 25} {'-' * 40}")

    for ws_dir, manifest, record in preserved:
        identifier = manifest.get("identifier", ws_dir.name)
        if record is not None:
            status = record.status.value
            branch = record.branch_name or "-"
        else:
            status = manifest.get("end_status", "unknown")
            branch = "-"
        print(f"  {identifier:<20} {status:<18} {branch:<25} {ws_dir}")

    return 0


# ---------------------------------------------------------------------------
# workspace show
# ---------------------------------------------------------------------------


def _cmd_show(
    workspace_root: Path | None,
    registry_path: Path | None,
    args: argparse.Namespace,
) -> int:
    """Show workspace details. Idempotent — pure read."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    ws_path, record = _resolve_workspace(workspace_root, registry_path, issue_id)
    if ws_path is None:
        print(f"No preserved workspace found for issue {issue_id}.", file=sys.stderr)
        return 1

    manifest = _read_manifest(ws_path)

    print(f"Workspace: {ws_path}")
    print(f"  Issue ID     : {manifest.get('issue_id', '-')}")
    print(f"  Identifier   : {manifest.get('identifier', '-')}")
    print(f"  End Status   : {manifest.get('end_status', '-')}")
    print(f"  End Reason   : {manifest.get('end_reason', '-')}")
    preserved_at = manifest.get("preserved_at")
    if preserved_at:
        try:
            # Handle both numeric timestamps and ISO strings
            if isinstance(preserved_at, (int, float)):
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(preserved_at))
            else:
                ts = str(preserved_at)
            print(f"  Preserved At : {ts}")
        except Exception:
            print(f"  Preserved At : {preserved_at}")

    if record is not None:
        print(f"  Branch       : {record.branch_name or '-'}")
        print(f"  Commit SHA   : {record.commit_sha or '-'}")
        print(f"  PR URL       : {record.pr_url or '-'}")
        print(f"  Verification : {record.verification_status or '-'}")

    # File listing
    print("\n  Files:")
    exclude = {".orchestrator_workspace", ".git", "node_modules", "__pycache__"}
    for item in sorted(ws_path.iterdir()):
        if item.name in exclude:
            continue
        marker = "[DIR] " if item.is_dir() else "      "
        print(f"    {marker}{item.name}")

    return 0


# ---------------------------------------------------------------------------
# workspace cd
# ---------------------------------------------------------------------------


def _cmd_cd(
    workspace_root: Path | None,
    registry_path: Path | None,
    args: argparse.Namespace,
) -> int:
    """Print workspace path. Idempotent — pure read."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    ws_path, _ = _resolve_workspace(workspace_root, registry_path, issue_id)
    if ws_path is None:
        print(f"No preserved workspace found for issue {issue_id}.", file=sys.stderr)
        return 1

    # Print only the path (for shell integration: cd $(...workspace cd --id X))
    print(ws_path)
    return 0


# ---------------------------------------------------------------------------
# workspace cleanup
# ---------------------------------------------------------------------------


def _cmd_cleanup(
    workspace_root: Path | None,
    registry_path: Path | None,
    args: argparse.Namespace,
) -> int:
    """Remove preserved workspaces. Destructive — requires --force or confirmation."""
    if not workspace_root or not workspace_root.exists():
        print("No workspace root found.", file=sys.stderr)
        return 1

    issue_id = getattr(args, "id", None)
    all_completed = getattr(args, "all_completed", False)
    force = getattr(args, "force", False)

    if not issue_id and not all_completed:
        print("error: specify --id <issue-id> or --all-completed", file=sys.stderr)
        return 2

    targets: list[tuple[Path, str | None]] = []  # (path, issue_id)

    if issue_id:
        ws_path, _ = _resolve_workspace(workspace_root, registry_path, issue_id)
        if ws_path is None:
            print(f"No preserved workspace found for issue {issue_id}.", file=sys.stderr)
            return 1
        targets.append((ws_path, issue_id))
    elif all_completed:
        registry = _load_registry(registry_path)
        for entry in workspace_root.iterdir():
            if not entry.is_dir():
                continue
            manifest = entry / ".orchestrator_workspace" / ".workspace_preserved.json"
            if not manifest.exists():
                continue
            m = _read_manifest(entry)
            end_status = (m.get("end_status") or "").lower()
            iid = m.get("issue_id") or m.get("identifier", "")
            if end_status == "completed":
                targets.append((entry, str(iid) if iid else None))
            elif registry is not None:
                record = registry.get_by_issue_ref(str(iid))
                if record is not None and record.status.value == "completed":
                    targets.append((entry, str(iid) if iid else None))

    if not targets:
        print("No workspaces to clean up.")
        return 0

    print(f"Will remove {len(targets)} workspace(s):")
    for t, _ in targets:
        print(f"  {t}")

    if not force:
        try:
            answer = input("\nProceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    registry = _load_registry(registry_path)
    removed = 0
    for t, iid in targets:
        try:
            shutil.rmtree(t)
            if t.exists():
                print(f"  Failed to remove {t}: directory still exists", file=sys.stderr)
                continue
            print(f"  Removed: {t}")
            removed += 1
            # Clear retry intent in registry to prevent orchestrator from re-creating workspace
            if registry is not None and iid:
                try:
                    registry.clear_intent(iid)
                    record = registry.get_by_issue_ref(iid)
                    if record is not None:
                        record.workspace_path = None
                        registry._save()
                except Exception as reg_exc:
                    print(
                        f"  Warning: failed to update registry for {iid}: {reg_exc}",
                        file=sys.stderr,
                    )
        except Exception as exc:
            print(f"  Failed to remove {t}: {exc}", file=sys.stderr)

    print(f"\nRemoved {removed}/{len(targets)} workspace(s).")
    return 0


# ---------------------------------------------------------------------------
# workspace verify
# ---------------------------------------------------------------------------


def _cmd_verify(
    workspace_root: Path | None,
    registry_path: Path | None,
    args: argparse.Namespace,
) -> int:
    """Run verify.sh in a preserved workspace."""
    issue_id = getattr(args, "id", None)
    if not issue_id:
        print("error: --id is required", file=sys.stderr)
        return 2

    ws_path, record = _resolve_workspace(workspace_root, registry_path, issue_id)
    if ws_path is None:
        print(f"No preserved workspace found for issue {issue_id}.", file=sys.stderr)
        return 1

    verify_script = ws_path / ".orchestrator_workspace" / "verify.sh"
    if not verify_script.exists():
        print(f"verify.sh not found in {ws_path}/.orchestrator_workspace", file=sys.stderr)
        print("Hint: verify.sh is auto-generated when a workspace is preserved", file=sys.stderr)
        print("      after successful verification. You can also create it manually.", file=sys.stderr)
        return 1

    print(f"Running verify.sh in {ws_path} ...")
    print("=" * 60)
    try:
        result = subprocess.run(
            [shutil.which("bash") or "bash", str(verify_script)],
            cwd=str(ws_path),
            timeout=300,
            check=False,
        )
        print("=" * 60)
        if result.returncode == 0:
            print("Verification passed ✓")
        else:
            print(f"Verification failed (exit code {result.returncode})")
        return result.returncode
    except subprocess.TimeoutExpired:
        print("Verification timed out (300s)", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to run verify.sh: {exc}", file=sys.stderr)
        return 1
