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

"""Shared workspace location resolution for orchestrator + CLI.

Provides unified workspace root resolution from:
1. CLAWCODEX_WORKSPACE_ROOT environment variable
2. Orchestrator metadata file (~/.clawcodex/orchestrator/{slug}/metadata.json)
3. --workflow parameter (parse WORKFLOW.md for workspace.root)
4. --workspace parameter (direct path)
5. CWD fallback
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ._file_utils import read_json, read_text_utf8, write_text_utf8

if TYPE_CHECKING:
    pass  # pylint: disable=import-error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAWCODEX_BASE = Path.home() / ".clawcodex"
ORCHESTRATOR_DIR = CLAWCODEX_BASE / "orchestrator"


def _slug_from_workspace(workspace_root: str | Path) -> str:
    """Generate a slug from workspace root for metadata directory naming."""
    path = str(workspace_root).strip().replace("/", "-").replace("\\", "-")
    # Use the last meaningful segment
    parts = [p for p in path.split("-") if p and p not in ("tmp", ".clawcodex", "~")]
    return "-".join(parts[-3:]) if parts else "default"


# ---------------------------------------------------------------------------
# Workspace root resolution
# ---------------------------------------------------------------------------


def get_workspace_root(
    workspace_arg: str | None = None,
    workflow_path: str | None = None,
) -> Path | None:
    """Resolve workspace root from multiple sources.

    Priority (highest to lowest):
    1. workspace_arg - explicit --workspace path
    2. CLAWCODEX_WORKSPACE_ROOT env var
    3. workflow_path - parse workspace.root from WORKFLOW.md
    4. orchestrator metadata file
    5. CWD fallback

    Args:
        workspace_arg: Direct --workspace path (highest priority)
        workflow_path: Path to WORKFLOW.md file (parse workspace.root from it)

    Returns:
        Resolved workspace root path, or None if not found
    """
    # 1. Explicit workspace path
    if workspace_arg:
        path = Path(workspace_arg).expanduser().resolve()
        if path.exists() or path.parent.exists():
            return path
        # Still return - may not exist yet for new orchestrator runs
        return path

    # 2. Environment variable
    env_path = os.environ.get("CLAWCODEX_WORKSPACE_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()

    # 3. Parse from WORKFLOW.md
    if workflow_path:
        workspace_root = _parse_workspace_from_workflow(workflow_path)
        if workspace_root:
            return workspace_root

    # 4. Check orchestrator metadata (latest for any project)
    metadata_path = _find_latest_metadata()
    if metadata_path:
        try:
            metadata = read_json(metadata_path)
            return Path(metadata["workspace_root"])
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Failed to read metadata file %s: %s", metadata_path, exc)

    # 5. CWD fallback
    cwd_registry = Path.cwd() / ".clawcodex_issue_registry.json"
    if cwd_registry.exists():
        return Path.cwd()

    # 6. Default
    default_workspace_root = CLAWCODEX_BASE / "workspace"
    if default_workspace_root.exists():
        return default_workspace_root

    return None


def get_registry_path(
    workspace_arg: str | None = None,
    workflow_path: str | None = None,
) -> Path | None:
    """Get registry path from resolved workspace root."""
    root = get_workspace_root(workspace_arg=workspace_arg, workflow_path=workflow_path)
    if root:
        return root / ".clawcodex_issue_registry.json"

    # No workspace found
    return None


# ---------------------------------------------------------------------------
# Workflow parsing
# ---------------------------------------------------------------------------


def _parse_workflow_front_matter(workflow_path: str | Path) -> dict | None:
    """Parse YAML front matter from a WORKFLOW.md file.

    Returns the parsed dict, or ``None`` if the file has no front matter
    or cannot be parsed.  Shared by ``_parse_workspace_from_workflow``
    and ``write_orchestrator_metadata`` (D-14).
    """
    try:
        content = read_text_utf8(Path(workflow_path))
        if not content.startswith("---"):
            return None
        lines = content.splitlines()
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                front_matter_raw = "\n".join(lines[1:i])
                front_matter = yaml.safe_load(front_matter_raw)
                return front_matter if isinstance(front_matter, dict) else None
    except (OSError, yaml.YAMLError) as exc:
        logger.debug("Failed to parse workflow front matter %s: %s", workflow_path, exc)
    return None


def _parse_workspace_from_workflow(workflow_path: str | Path) -> Path | None:
    """Parse workspace.root from WORKFLOW.md YAML front matter."""
    front_matter = _parse_workflow_front_matter(workflow_path)
    if front_matter is None:
        return None
    workspace_root = front_matter.get("workspace", {}).get("root")
    if workspace_root:
        root_path = Path(os.path.expanduser(workspace_root))
        if not root_path.is_absolute():
            root_path = (Path(workflow_path).parent / root_path).resolve()
        return root_path
    return None


# ---------------------------------------------------------------------------
# Orchestrator metadata management
# ---------------------------------------------------------------------------


def _find_latest_metadata() -> Path | None:
    """Find the most recently modified orchestrator metadata file."""
    if not ORCHESTRATOR_DIR.exists():
        return None
    metadata_files = []
    for project_dir in ORCHESTRATOR_DIR.iterdir():
        if project_dir.is_dir():
            metadata_file = project_dir / "metadata.json"
            if metadata_file.exists():
                metadata_files.append((metadata_file.stat().st_mtime, metadata_file))
    if not metadata_files:
        return None
    metadata_files.sort(key=lambda x: x[0], reverse=True)
    return metadata_files[0][1]


def write_orchestrator_metadata(
    workspace_root: str | Path,
    workflow_path: str | None = None,
    started_at: float | None = None,
) -> Path:
    """Write orchestrator metadata for later CLI discovery.

    Creates ~/.clawcodex/orchestrator/{slug}/metadata.json

    Args:
        workspace_root: The orchestrator's workspace root
        workflow_path: Optional path to WORKFLOW.md (for project identification)

    Returns:
        Path to the metadata file written
    """
    import time

    workspace_root_str = str(workspace_root)
    slug = _slug_from_workspace(workspace_root_str)

    # Create metadata directory
    metadata_dir = ORCHESTRATOR_DIR / slug
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = metadata_dir / "metadata.json"

    # Determine project slug from workflow if available
    project_slug = None
    if workflow_path:
        front_matter = _parse_workflow_front_matter(workflow_path)
        if front_matter:
            tracker = front_matter.get("tracker", {})
            owner = tracker.get("owner", "")
            repo = tracker.get("repo", "")
            if owner and repo:
                project_slug = f"{owner}-{repo}"

    data = {
        "workspace_root": workspace_root_str,
        "pid": os.getpid(),
        "started_at": started_at if started_at is not None else time.time(),
        "project_slug": project_slug or slug,
        "workflow_path": str(workflow_path) if workflow_path else None,
    }

    write_text_utf8(
        metadata_file,
        json.dumps(data, indent=2, ensure_ascii=False),
    )

    return metadata_file


def clear_orchestrator_metadata(workspace_root: str | Path) -> None:
    """Remove orchestrator metadata file."""
    slug = _slug_from_workspace(str(workspace_root))
    metadata_file = ORCHESTRATOR_DIR / slug / "metadata.json"
    if metadata_file.exists():
        metadata_file.unlink()


def list_orchestrator_projects() -> list[dict]:
    """List all known orchestrator projects from metadata files."""
    projects = []
    if not ORCHESTRATOR_DIR.exists():
        return projects

    for metadata_dir in ORCHESTRATOR_DIR.iterdir():
        if metadata_dir.is_dir():
            metadata_file = metadata_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    data = read_json(metadata_file)
                    projects.append(data)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.debug("Failed to read metadata %s: %s", metadata_file, exc)

    return projects


def get_live_projects(projects: list[dict] | None = None) -> list[dict]:
    """Filter orchestrator metadata to projects whose PID is still alive.

    Args:
        projects: List of project metadata dicts. If None, fetches from
                  ``list_orchestrator_projects()``.

    Returns:
        List of live project dicts, each containing at minimum:
        workspace_root, pid, started_at, project_slug, workflow_path.
    """

    if projects is None:
        projects = list_orchestrator_projects()

    live: list[dict] = []

    for project in projects:
        pid = project.get("pid")
        if pid is None:
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            continue  # Process not alive

        live.append(
            {
                "workspace_root": project.get("workspace_root"),
                "pid": pid,
                "started_at": project.get("started_at"),
                "project_slug": project.get("project_slug"),
                "workflow_path": project.get("workflow_path"),
            }
        )

    return live


def print_multi_project_hint(
    live_projects: list[dict],
    command_hint: str,
) -> None:
    """Print a hint to stderr when multiple live orchestrator projects exist.

    The hint tells the user to use ``--workspace`` or ``--workflow`` to
    disambiguate.

    Args:
        live_projects: List of live project dicts (from ``get_live_projects``).
        command_hint: The command the user ran, for context in the message.
    """
    import time
    import sys

    lines: list[str] = [
        f"⚠  {len(live_projects)} running orchestrator projects detected.",
        f"   Command: {command_hint}",
        "",
        "   Running projects:",
    ]

    now = time.time()
    for project in live_projects:
        workspace_root = project.get("workspace_root", "?")
        slug = project.get("project_slug", "?")
        pid = project.get("pid", "?")
        uptime_seconds = now - project.get("started_at", now) if project.get("started_at") else 0
        uptime_str = f"{uptime_seconds:.0f}s" if uptime_seconds < 120 else f"{uptime_seconds / 60:.0f}m"
        lines.append(f"     [{slug}]  pid={pid}  uptime={uptime_str}  workspace={workspace_root}")

    lines.extend(
        [
            "",
            "   Use --workspace <path> or --workflow <path> to target a specific project.",
            "",
        ]
    )

    print("\n".join(lines), file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def resolve_for_cli(
    workspace_arg: str | None,
    workflow_arg: str | None,
) -> tuple[Path | None, Path | None]:
    """Resolve workspace root and registry path for CLI commands.

    Returns:
        tuple of (workspace_root, registry_path)
    """
    root = get_workspace_root(workspace_arg=workspace_arg, workflow_path=workflow_arg)
    if root:
        registry = root / ".clawcodex_issue_registry.json"
        return root, registry

    return None, None


def print_workspace_info(workspace_root: Path | None, workflow_path: str | None = None) -> str:
    """Generate a human-readable workspace info string."""
    if workspace_root:
        parts = [f"workspace: {workspace_root}"]
    else:
        parts = ["workspace: (not found)"]

    if workflow_path:
        parts.append(f"workflow: {workflow_path}")

    return " | ".join(parts)
