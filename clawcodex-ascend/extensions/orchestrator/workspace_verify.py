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

"""Auto-generation of verify.sh and README.md for preserved workspaces.

When a workspace is preserved after an issue completes (or fails), these
functions generate helper files to make manual verification easier:

- verify.sh: One-click script to re-run test/build/lint commands
- README.md: Documentation about the workspace, issue, and changes
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any

from ._file_utils import orchestrator_metadata_dir, write_text_utf8

logger = logging.getLogger(__name__)


def _verification_step_lines(label: str, cmd: str) -> list[str]:
    """Build the echo + cmd + echo-OK lines for one verification step (D-11)."""
    return [
        f"echo '>>> Running {label}...'",
        cmd,
        f"echo '{label.capitalize()}: OK'",
        "echo ''",
        "",
    ]


def _safe_write_text(path: Path, content: str, description: str) -> bool:
    """Write *content* to *path*, logging a warning on failure (D-12 / X-02).

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        write_text_utf8(path, content)
        return True
    except Exception as exc:
        logger.warning("Failed to write %s: %s", description, exc)
        return False


def generate_verify_script(
    workspace_path: Path,
    agent_config: Any,
    issue_record: Any,
) -> None:
    """Generate verify.sh script for manual verification.

    Args:
        workspace_path: Path to the preserved workspace
        agent_config: AgentConfig with test/build/lint commands
        issue_record: IssueRecord with issue metadata
    """
    test_cmd = getattr(agent_config, "test_command", None)
    build_cmd = getattr(agent_config, "build_command", None)
    lint_cmd = getattr(agent_config, "lint_command", None)

    # Skip if no verification commands configured
    if not any([test_cmd, build_cmd, lint_cmd]):
        logger.debug("No verification commands configured, skipping verify.sh generation")
        return

    metadata_dir = orchestrator_metadata_dir(workspace_path)
    metadata_dir.mkdir(exist_ok=True)
    verify_path = metadata_dir / "verify.sh"

    lines = [
        "#!/usr/bin/env bash",
        "# ClawCodex Verify Script",
        "#",
        "# This script re-runs the verification commands that were executed",
        "# during the automated issue processing.",
        "#",
        f"# Issue: {getattr(issue_record, 'issue_id', 'unknown')}",
        f"# Branch: {getattr(issue_record, 'branch_name', 'unknown')}",
        f"# Commit: {getattr(issue_record, 'commit_sha', 'unknown')}",
        f"# Status: {getattr(issue_record, 'status', 'unknown')}",
        "#",
        "# Usage: ./verify.sh",
        "#",
        "set -e",
        "",
        "echo '=== ClawCodex Verification ==='",
        f"echo Issue: {shlex.quote(str(getattr(issue_record, 'issue_id', 'unknown')))}",
        f"echo Branch: {shlex.quote(str(getattr(issue_record, 'branch_name', 'unknown')))}",
        f"echo Commit: {shlex.quote(str(getattr(issue_record, 'commit_sha', 'unknown')))}",
        "echo ''",
        "",
    ]

    if build_cmd:
        lines.extend(_verification_step_lines("build", build_cmd))

    if lint_cmd:
        lines.extend(_verification_step_lines("lint", lint_cmd))

    if test_cmd:
        lines.extend(_verification_step_lines("tests", test_cmd))

    lines.extend(
        [
            "echo '=== All verification steps passed ==='",
        ]
    )

    if _safe_write_text(verify_path, "\n".join(lines), "verify.sh"):
        verify_path.chmod(0o755)
        logger.info("Generated verify.sh at %s", verify_path)


def generate_workspace_readme(
    workspace_path: Path,
    issue_record: Any,
) -> None:
    """Generate README.md documenting the preserved workspace.

    Args:
        workspace_path: Path to the preserved workspace
        issue_record: IssueRecord with issue metadata
    """
    metadata_dir = orchestrator_metadata_dir(workspace_path)
    metadata_dir.mkdir(exist_ok=True)
    readme_path = metadata_dir / "README.md"

    issue_id = getattr(issue_record, "issue_id", "unknown")
    identifier = getattr(issue_record, "identifier", "unknown")
    status = getattr(issue_record, "status", "unknown")
    branch = getattr(issue_record, "branch_name", "unknown")
    commit = getattr(issue_record, "commit_sha", "unknown")
    pr_url = getattr(issue_record, "pr_url", None)
    verification = getattr(issue_record, "verification_status", None)

    lines = [
        f"# Workspace: {identifier}",
        "",
        "This workspace was preserved by ClawCodex after automated issue processing.",
        "",
        "## Issue Information",
        "",
        f"- **Issue ID**: {issue_id}",
        f"- **Identifier**: {identifier}",
        f"- **Status**: {status}",
        f"- **Branch**: `{branch}`",
        f"- **Commit**: `{commit}`",
    ]

    if pr_url:
        lines.append(f"- **Pull Request**: {pr_url}")

    if verification:
        lines.append(f"- **Verification**: {verification}")

    lines.extend(
        [
            "",
            "## Workspace Contents",
            "",
            "This directory contains the code changes made by ClawCodex for this issue.",
            "",
        ]
    )

    # List top-level files/directories
    try:
        entries = sorted(workspace_path.iterdir())
        if entries:
            visible = [e for e in entries if not e.name.startswith(".")]
            if visible:
                lines.append("### Files and Directories")
                lines.append("")
                for dir_entry in visible[:20]:
                    prefix = "[DIR]" if dir_entry.is_dir() else "[FILE]"
                    lines.append(f"- {prefix} `{dir_entry.name}`")
                if len(visible) > 20:
                    lines.append(f"- ... and {len(visible) - 20} more")
                lines.append("")
    except OSError as exc:
        logger.debug("Failed to list workspace contents: %s", exc)

    lines.extend(
        [
            "## Verification",
            "",
            "To verify the changes manually, run:",
            "",
            "```bash",
            "./.orchestrator_workspace/verify.sh",
            "```",
            "",
            "Or run individual commands:",
            "",
        ]
    )

    # Add manual verification instructions
    test_cmd = "pytest"  # default
    build_cmd = None
    lint_cmd = None

    # Try to detect commands from common files
    if (workspace_path / "package.json").exists():
        test_cmd = "npm test"
        build_cmd = "npm run build"
        lint_cmd = "npm run lint"
    elif (workspace_path / "Cargo.toml").exists():
        test_cmd = "cargo test"
        build_cmd = "cargo build"
        lint_cmd = "cargo clippy"
    elif (workspace_path / "go.mod").exists():
        test_cmd = "go test ./..."
        build_cmd = "go build ./..."
        lint_cmd = "golangci-lint run"
    elif (workspace_path / "pyproject.toml").exists() or (workspace_path / "setup.py").exists():
        test_cmd = "pytest"
        lint_cmd = "ruff check ."

    if build_cmd:
        lines.append("```bash")
        lines.append("# Build")
        lines.append(f"{build_cmd}")
        lines.append("```")
        lines.append("")

    if lint_cmd:
        lines.append("```bash")
        lines.append("# Lint")
        lines.append(f"{lint_cmd}")
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "```bash",
            "# Test",
            f"{test_cmd}",
            "```",
            "",
            "## Cleanup",
            "",
            "To remove this preserved workspace:",
            "",
            "```bash",
            f"clawcodex-dev orchestrator workspace cleanup --id {issue_id} --force",
            "```",
            "",
            "---",
            "",
            "*Generated by [ClawCodex](https://github.com/your-org/clawcodex)*",
        ]
    )

    if _safe_write_text(readme_path, "\n".join(lines), "README.md"):
        logger.info("Generated README.md at %s", readme_path)
