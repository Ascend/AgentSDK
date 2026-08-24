# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
"""Prompt context and workspace-Python helpers for Agent Runtime."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

USER_MESSAGE_MARKER = "<!-- === USER MESSAGE === -->"


def _build_sequential_workspace_context(session: Any) -> str:
    return "\n".join(
        [
            "---",
            "## Sequential Workspace Context",
            "",
            "This issue is running in a sequential shared workspace.",
            f"- Workspace strategy: `{getattr(session, 'workspace_strategy', 'sequential')}`",
            f"- Integration branch: `{getattr(session, 'integration_branch', None) or 'current branch'}`",
            f"- Start commit: `{getattr(session, 'start_commit_sha', None) or 'unknown'}`",
            f"- Base commit: `{getattr(session, 'base_commit_sha', None) or 'unknown'}`",
            f"- Previous issue: `{getattr(session, 'previous_issue_id', None) or 'none'}`",
            f"- Sequence index: `{getattr(session, 'sequence_index', None) or 'unknown'}`",
            "",
            "Build on the existing commit chain in this workspace. Do not redo earlier issues.",
            "If the expected prior commit chain appears to be missing, stop and report it.",
            "---",
        ]
    )


def _expand_agent_mentions_in_prompt(
    system_part: str,
    user_part: str,
    *,
    session: Any | None = None,
) -> tuple[str, str]:
    """F-89: expand ``@agent-<type>`` mentions across the rendered prompt.

    Mirrors the REPL/TUI/headless behaviour using the shared
    :func:`clawcodex_ext.command_system.input_processing` helpers. Returns
    ``(system_part, user_part)`` with agent attachments prepended to the
    user half (so the model sees both the reminder and the original
    issue text). Unknown mentions are stripped with a logged warning —
    orchestrator runs must keep going on a typo in the issue body,
    whereas interactive entry points can show a friendly error and
    drop the turn.

    Agent discovery is best-effort internally, but missing imports are
    allowed to surface because they indicate an incomplete installation.
    """
    from src.command_system.input_processing import (
        expand_agent_mentions,
        find_unknown_agent_mentions,
        format_at_mention_attachments,
        strip_agent_mentions,
    )

    from clawcodex_ext.agent.load_agents_dir import get_agents_for_mentions

    workspace_path = _resolve_agent_expansion_workspace(session)
    agents = get_agents_for_mentions(str(workspace_path)) if workspace_path else []

    if not agents:
        return system_part, user_part

    # Concatenate for a single sweep so an @agent- mention that splits
    # across the marker line is still detected. We then re-split using
    # known markers after stripping/injecting.
    combined = f"{system_part}\n\n{user_part}"

    unknown = find_unknown_agent_mentions(combined, agents)
    if unknown:
        logger.warning(
            "F-89: stripping unknown agent mention(s) from orchestrator prompt: %s",
            ", ".join(unknown),
        )
        combined = strip_agent_mentions(combined)

    attachments = expand_agent_mentions(combined, agents)
    if attachments:
        extra = format_at_mention_attachments(attachments)
        if extra:
            combined = f"{extra}\n\n{combined}"

    # If the original render split cleanly, keep the split; otherwise
    # everything collapses back into user_part (the marker line is gone
    # after our edit, which is fine — the LLM still sees the reminder
    # before the body).
    marker = USER_MESSAGE_MARKER
    if marker in combined:
        new_system, new_user = combined.split(marker, 1)
        return new_system.strip(), new_user.strip()
    return "", combined.strip()


def _resolve_agent_expansion_workspace(session: Any | None) -> Path | None:
    """Extract a workspace root path suitable for agent discovery."""
    return _resolve_workspace_path(session)


def _to_jinja_value(value: Any) -> Any:
    """Coerce a value into Jinja2-friendly shapes."""
    if isinstance(value, dict):
        return {str(k): _to_jinja_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jinja_value(v) for v in value]
    return value


def _resolve_workspace_path(session: Any) -> Path | None:
    """Extract the workspace root path from a session object.

    Returns None when there is no session or no workspace, which means
    the workspace-diff context is silently skipped.
    """
    if session is None:
        return None
    ws = getattr(session, "workspace", None)
    if ws is None:
        return None
    path = getattr(ws, "path", None)
    if path is None:
        return None
    return Path(path)


def _get_workspace_diff(ws_path: Path) -> str | None:
    """Run ``git diff --stat`` and ``git status --short`` in the
    workspace to produce a compact summary of uncommitted changes.

    Returns ``None`` when the workspace is clean (no changes), so the
    caller can skip injecting the diff context block entirely.
    """
    try:
        proc = subprocess.run(  # nosec B607 -- fixed git executable
            ["git", "diff", "--stat", "HEAD"],
            cwd=str(ws_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        diff_stat = proc.stdout.strip()
        proc2 = subprocess.run(  # nosec B607 -- fixed git executable
            ["git", "status", "--short"],
            cwd=str(ws_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status_short = proc2.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None

    if not diff_stat and not status_short:
        return None  # clean workspace — nothing to inject
    parts = []
    if diff_stat:
        parts.append(f"```\n{diff_stat}\n```")
    if status_short:
        parts.append(f"Uncommitted files:\n```\n{status_short}\n```")
    return "\n".join(parts)


def _get_operator_hints(ws_path: Path) -> str | None:
    """Read ``.operator_hints.md`` from workspace, return contents.

    Inject hints are one-shot: once read they are removed so the agent
    sees them exactly once (the next turn-boundary prompt) and historical
    injects do not accumulate across turns or runs.

    However, ``repro_gate.append_repro_hint`` also writes to this file —
    a ``## Reproduction established`` section that MUST persist across
    every turn's prompt until the fix is complete. This method preserves
    that section when clearing inject hints.

    Returns ``None`` when the file is missing or empty so callers can
    skip injecting the operator-hints block entirely.
    """
    hints_file = ws_path / ".operator_hints.md"
    if not hints_file.exists():
        return None
    try:
        content = hints_file.read_text(encoding="utf-8").strip()
        # Preserve the ## Reproduction established section (from
        # repro_gate.append_repro_hint) which must persist across turns.
        # Only clear inject hints (one-shot semantics).
        repro_idx = content.find("## Reproduction established")
        repro_section = content[repro_idx:] if repro_idx >= 0 else ""
        hints_file.write_text(repro_section, encoding="utf-8")
        if content:
            return content
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read operator hints from %s: %s", hints_file, exc)
    return None


def _get_git_log_summary(session: Any) -> str:
    """Run ``git log --oneline -3`` in the workspace and return a
    compact summary of recent commits, or an empty string when there
    is no session / workspace / git history.

    F-54 root-cause fix: injected into continuation prompts so the
    LLM can see what has already been committed in previous turns
    and avoid re-exploring from scratch.
    """
    if session is None:
        return ""
    ws = getattr(session, "workspace", None)
    if ws is None:
        return ""
    ws_path = getattr(ws, "path", None)
    if ws_path is None:
        return ""
    try:
        proc = subprocess.run(  # nosec B607 -- fixed git executable
            ["git", "log", "--oneline", "-3"],
            cwd=str(ws_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        log_out = proc.stdout.strip()
        if not log_out:
            return ""
        return f"\nRecent commits in workspace:\n```\n{log_out}\n```\n"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


# ─── Python interpreter detection + cascade resolver ────────────────
# F-?? workspace-level python_executable: detect the interpreter
# inside the target repo via project-level signals, then expose a
# cascade so the prompt builder can pick the most specific value.


def _parse_pyvenv_home(cfg_path: Path) -> str:
    """Extract ``home = <path>`` from a pyvenv.cfg file.

    Returns the home directory string or ``""`` on parse failure /
    missing file. Soft-fails by design: malformed pyvenv.cfg must
    not block prompt rendering.
    """
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("home"):
            _, _, value = stripped.partition("=")
            return value.strip().strip('"').strip("'")
    return ""


def _parse_conda_env_name(yml_path: Path) -> str:
    """Extract the conda env ``name:`` from an environment.yml.

    Returns the env name or ``""`` if no ``name:`` key is set.
    """
    try:
        text = yml_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.lower().startswith("name"):
            _, _, value = stripped.partition(":")
            return value.strip().strip('"').strip("'")
    return ""


# Ordered conda root locations probed when an ``environment.yml`` is
# present. ``CONDA_PREFIX`` is consulted first when set (covers any
# non-standard install location), then the common defaults.
_WINDOWS_CONDA_ROOT_CANDIDATES: tuple[str, ...] = (
    (
        str(Path.home() / "anaconda3"),
        str(Path.home() / "miniconda3"),
        str(Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "anaconda3"),
        str(Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "miniconda3"),
    )
    if os.name == "nt"
    else ()
)

_CONDA_ROOT_CANDIDATES: tuple[str, ...] = (
    "/opt/conda",
    "/root/anaconda3",
    "/root/miniconda3",
    "/usr/local/anaconda3",
    "/usr/local/miniconda3",
    "/opt/anaconda3",
) + _WINDOWS_CONDA_ROOT_CANDIDATES


# The nested branches mirror the ordered workspace-signal cascade and keep
# each detector local to the candidate it owns.
# pylint: disable=too-many-nested-blocks
def _detect_python_in_workspace(
    workspace_path: Path | None,
    candidates: list[str],
) -> str:
    """Walk a list of project-level signals and return the absolute
    path of the first Python interpreter that can be derived from
    them. Returns ``""`` when nothing matches.

    Soft-fails: missing files, malformed contents, or non-existent
    interpreter binaries are silently skipped — the function is
    best-effort and never raises.

    Recognised probe kinds (matched by relative path):

    * ``.python-version`` — pyenv version spec; resolved against
      ``$PYENV_ROOT`` or ``~/.pyenv/versions/<v>/bin/python3``.
    * ``pyvenv.cfg`` and ``.venv/pyvenv.cfg`` — venv / uv / poetry
      venv markers; the ``home = ...`` line gives the venv root.
    * ``environment.yml`` — conda env file; ``name:`` is matched
      against ``$CONDA_PREFIX`` and a set of well-known conda
      install prefixes.
    * ``Pipfile`` and ``pyproject.toml`` — recognised but skipped
      because they describe dependencies rather than interpreter
      paths. Listed in the default candidates so operators can
      disable them via ``python_detect_files`` if desired.
    """
    if workspace_path is None:
        return ""
    workspace_path = Path(workspace_path)
    if not workspace_path.exists():
        return ""

    for rel in candidates:
        f = workspace_path / rel
        if not f.exists() or not f.is_file():
            continue
        try:
            if rel == ".python-version":
                version = f.read_text(encoding="utf-8", errors="replace").strip()
                if version:
                    pyenv_root = Path(os.environ.get("PYENV_ROOT", str(Path.home() / ".pyenv")))
                    py = pyenv_root / "versions" / version / "bin" / "python3"
                    if py.exists():
                        return str(py)
            elif rel.endswith("pyvenv.cfg"):
                # ``home`` identifies the base interpreter used to create
                # the environment, not the environment itself.  The active
                # venv interpreter lives beside this pyvenv.cfg file.
                venv_root = f.parent
                for py in (
                    venv_root / "bin" / "python3",
                    venv_root / "Scripts" / "python.exe",
                ):
                    if py.exists():
                        return str(py)
            elif rel == "environment.yml":
                env_name = _parse_conda_env_name(f)
                if env_name:
                    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
                    roots: list[str] = [conda_prefix] if conda_prefix else []
                    roots += list(_CONDA_ROOT_CANDIDATES)
                    for root in roots:
                        if not root:
                            continue
                        env_root = Path(root) / "envs" / env_name
                        for py in (
                            env_root / "bin" / "python3",
                            env_root / "python.exe",
                        ):
                            if py.exists():
                                return str(py)
            elif rel in ("Pipfile", "pyproject.toml"):
                continue
        except OSError:
            continue
    return ""


def resolve_python_executable(
    *,
    workspace_path: Path | None,
    agent_cfg: Any,
    workspace_cfg: Any,
    issue_executable: str = "",
) -> str:
    """Cascade resolver: pick the most specific Python interpreter
    path available.

    Resolution order (first non-empty wins):

    1. ``issue_executable`` — per-issue override (e.g. from
       ``LocalTracker`` frontmatter ``python_executable: ...``).
       Highest priority because a single issue may legitimately
       need a different interpreter than its sibling issues in the
       same workspace.
    2. ``workspace_cfg.python_executable`` — explicit per-workspace
       override (handles "different repo needs different python").
    3. Auto-detected path via
       :func:`_detect_python_in_workspace` when
       ``workspace_cfg.python_auto_detect`` is True.
    4. ``agent_cfg.python_executable`` — workflow-wide default
       (the MVP-1 knob).
    5. Empty string — caller should treat as "no constraint"; the
       agent will rely on PATH ``python3``.

    Args:
        workspace_path: Absolute path to the workspace directory
            (``Workspace.path``), or ``None`` when there is no
            workspace yet (e.g. unit tests).
        agent_cfg: An ``AgentConfig``-like object exposing
            ``python_executable``.
        workspace_cfg: A ``WorkspaceConfig``-like object exposing
            ``python_executable``, ``python_auto_detect`` and
            ``python_detect_files``.
        issue_executable: Per-issue override string. ``""`` (the
            default) skips this level entirely. Provided by the
            caller from ``Issue.python_executable`` (populated by
            ``LocalTrackerAdapter`` from the issue markdown
            frontmatter).

    Returns:
        Absolute path string, or ``""`` when no constraint applies.
    """
    issue_override = (issue_executable or "").strip()
    if issue_override:
        return issue_override

    ws_explicit = getattr(workspace_cfg, "python_executable", "") or ""
    if ws_explicit:
        return ws_explicit

    auto_detect = getattr(workspace_cfg, "python_auto_detect", True)
    if auto_detect:
        detect_files = list(
            getattr(workspace_cfg, "python_detect_files", None)
            or [
                ".python-version",
                "pyvenv.cfg",
                ".venv/pyvenv.cfg",
                "Pipfile",
                "environment.yml",
            ]
        )
        detected = _detect_python_in_workspace(workspace_path, detect_files)
        if detected:
            return detected

    agent_default = getattr(agent_cfg, "python_executable", "") or ""
    if agent_default:
        return agent_default

    return ""
