#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=E0402

"""Board identity resolution and filesystem path helpers.

Implements the 5-tier board identity resolution specified in §5.2:
  1. Explicit board_id parameter
  2. Environment variable CLAWCODEX_LKB_BOARD_ID
  3. Project LKB config (.claude/config.json -> lkb.board_id)
  4. Stable derivation from repository origin/relative path, or workspace path
  5. Session-scoped Board when no workspace is available

Also provides ``safe_board_id`` (readable prefix + hash, LKB-STORE-028),
``board_dir`` / subpath helpers, and ``normalize_workspace_root``
(case-normalized on Windows, LKB-BOARD-004).

Home-directory precedence follows the same pattern as
``clawcodex_ext.goal.store.goals_db_path``:
explicit *home* param > ``CLAWCODEX_HOME`` env > ``Path.home()/.clawcodex``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import warnings
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any

from .error_codes import LkbErrorCode
from .graph_types import Board, BoardPolicy

# ── public API ───────────────────────────────────────────────────────

__all__ = [
    "BoardResolutionError",
    "LKB_BOARDS_SUBDIR",
    "board_dir",
    "board_file_paths",
    "resolve_board",
    "safe_board_id",
    "normalize_workspace_root",
]

LKB_BOARDS_SUBDIR = "lkb/boards"

# Characters that are never allowed in a board_id (path traversal /
# filesystem-safety guard).  Board IDs are validated on every load
# (LKB-STORE-028) so even if a malicious caller bypasses this check,
# the store will reject it.
#
# We only reject characters that would make path traversal possible
# or cause severe filesystem issues.  Spaces, dots (non-leading),
# and most punctuation are allowed in board_id; the safe_board_id
# function further sanitizes the *prefix* used in directory names.
_BOARD_ID_FORBIDDEN_RE = re.compile(r"[\x00-\x1f/\\\x7f]")
_SESSION_BOARD_LOCK = threading.Lock()
_PROCESS_SESSION_BOARD_ID: str | None = None


class BoardResolutionError(ValueError):
    """Raised when board identity cannot be resolved safely."""

    code = LkbErrorCode.INVALID_BOARD_ID


# ── 1. resolve_board (spec §5.2, 5-tier priority) ────────────────────


def resolve_board(
    workspace_root: str | Path | None = None,
    *,
    explicit_id: str | None = None,
    session_id: str | None = None,
    home: Path | None = None,
) -> Board:
    """Resolve a Board identity using the 5-tier priority from spec §5.2.

    Parameters
    ----------
    workspace_root:
        Optional path to the project/workspace root.  When provided,
        tiers 3 (project config) and 4 (derived ID) are attempted.
    explicit_id:
        Explicit board_id (tier 1 — highest priority).
    home:
        Accepted temporarily for source compatibility.  Board identity
        resolution never used this parameter; use :func:`board_dir` or
        :func:`board_file_paths` to select a storage home.

    Returns
    -------
    Board
        A Board instance populated with board_id, project_uri,
        display_name, schema_version=1, store_revision=0, and
        default BoardPolicy.  ``scope`` is encoded in project_uri:
        "project:<uri>" for project boards, "session:<id>" for
        session boards.
    """

    if home is not None:
        warnings.warn(
            "resolve_board(..., home=...) is deprecated because home does not affect "
            "board identity; use board_dir(..., home=...) for storage paths.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Tier 1: explicit board_id
    if explicit_id is not None:
        _validate_board_id(explicit_id)
        project_uri = _project_uri_from_root(workspace_root) if workspace_root else ""
        return Board(
            board_id=explicit_id,
            project_uri=project_uri or f"explicit:{explicit_id}",
            display_name=explicit_id,
            schema_version=1,
            store_revision=0,
            policy=BoardPolicy(),
        )

    # Tier 2: environment variable
    env_id = os.environ.get("CLAWCODEX_LKB_BOARD_ID")
    if env_id:
        _validate_board_id(env_id)
        project_uri = _project_uri_from_root(workspace_root) if workspace_root else ""
        return Board(
            board_id=env_id,
            project_uri=project_uri or f"env:{env_id}",
            display_name=env_id,
            schema_version=1,
            store_revision=0,
            policy=BoardPolicy(),
        )

    # Tier 3: project LKB config
    if workspace_root is not None:
        config_id = _read_project_board_id(workspace_root)
        if config_id is not None:
            _validate_board_id(config_id)
            project_uri = _project_uri_from_root(workspace_root)
            return Board(
                board_id=config_id,
                project_uri=project_uri,
                display_name=config_id,
                schema_version=1,
                store_revision=0,
                policy=BoardPolicy(),
            )

    # Tier 4: derive from credential-free repository identity, falling back
    # to the normalized workspace root outside Git.
    if workspace_root is not None:
        derived_id = _derive_board_id(workspace_root)
        project_uri = _project_uri_from_root(workspace_root)
        return Board(
            board_id=derived_id,
            project_uri=project_uri,
            display_name=_display_name_from_root(workspace_root),
            schema_version=1,
            store_revision=0,
            policy=BoardPolicy(),
        )

    # Tier 5: session board
    resolved_session_id = _session_board_id(session_id)
    return Board(
        board_id=resolved_session_id,
        project_uri=f"session:{resolved_session_id}",
        display_name="Session Board",
        schema_version=1,
        store_revision=0,
        policy=BoardPolicy(),
    )


# ── 2. safe_board_id (readable prefix + hash) ────────────────────────


def safe_board_id(board_id: str) -> str:
    """Convert a board_id into a safe filesystem directory name.

    Format: ``<sanitized-prefix>_<sha256(board_id)[:16]>``

    The prefix is derived from the board_id itself, truncated to ~12
    filesystem-safe characters.  The 16-hex-char SHA-256 suffix provides
    collision resistance.

    The original board_id is re-validated on every load (LKB-STORE-028)
    so directory name manipulation cannot fool the store into loading
    a different board.

    Raises BoardResolutionError if board_id contains forbidden
    path-traversal characters.
    """
    _validate_board_id(board_id)

    # Build a readable prefix: keep alphanumeric + dash + underscore,
    # collapse runs, trim, lowercase, limit to 12 chars.
    prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", board_id)
    prefix = prefix.strip("-").lower()
    if len(prefix) > 12:
        prefix = prefix[:12].rstrip("-")
    if not prefix:
        prefix = "board"

    digest = hashlib.sha256(board_id.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


# ── 3. board_dir + subpath helpers (spec §7.2) ───────────────────────


def resolve_home(*, home: Path | None = None) -> Path:
    """Resolve the clawcodex home directory.

    Precedence: explicit *home* > ``CLAWCODEX_HOME`` env > ``Path.home()``.
    Mirrors ``clawcodex_ext.goal.store.goals_db_path`` (lines 337-343).
    """
    if home is not None:
        return Path(home)
    env_home = os.environ.get("CLAWCODEX_HOME")
    if env_home:
        return Path(env_home)
    return Path.home() / ".clawcodex"


def board_dir(board_id: str, *, home: Path | None = None) -> Path:
    """Return the per-board directory for *board_id*.

    Path: ``<home>/lkb/boards/<safe-board-id>/``

    The caller is responsible for creating the directory when needed;
    this function only computes the path and validates board_id.
    """
    _validate_board_id(board_id)
    home_path = resolve_home(home=home)
    return home_path / LKB_BOARDS_SUBDIR / safe_board_id(board_id)


def board_file_paths(board_id: str, *, home: Path | None = None) -> dict[str, Path]:
    """Return a dict of named filesystem paths for a board.

    Keys match the spec §7.2 directory layout:
      board_json, board_json_bak, lock_file, lock_owner_json,
      tmp_dir, history_dir, quarantine_dir
    """
    d = board_dir(board_id, home=home)
    return {
        "board_json": d / "board.json",
        "board_json_bak": d / "board.json.bak",
        "lock_file": d / ".lock",
        "lock_owner_json": d / ".lock.owner.json",
        "tmp_dir": d / ".tmp",
        "history_dir": d / "history",
        "quarantine_dir": d / "quarantine",
    }


# ── 4. normalize_workspace_root (LKB-BOARD-004) ──────────────────────


def normalize_workspace_root(path: str | Path) -> str:
    """Normalize a workspace root path for use as a stable identity input.

    - Resolves to an absolute path.
    - On Windows, case-normalizes the drive letter and uses lower-case
      for path components that are case-insensitive on NTFS (the whole
      path, effectively).
    - Normalizes separators to ``/``.
    - Resolves symlinks where possible (best-effort).

    The result is a canonical string suitable for hashing into a
    stable board_id (tier 4 derivation).
    """
    p = Path(path).expanduser().resolve()
    result = str(p)

    # Normalize separators to forward slash.
    result = result.replace("\\", "/")

    # Case-normalize on Windows (NTFS is case-insensitive by default).
    # We detect Windows via os.name rather than sys.platform so the
    # behaviour matches the runtime filesystem semantics.
    if os.name == "nt":
        # Lowercase the entire path — on NTFS this is safe and gives
        # a canonical form regardless of how the path was entered.
        result = result.lower()

    # Strip trailing slash (but keep root slash).
    if len(result) > 1 and result.endswith("/"):
        result = result[:-1]

    return result


# ── internal helpers ──────────────────────────────────────────────────


def _validate_board_id(board_id: str) -> None:
    """Validate that *board_id* is safe for use in filesystem paths.

    Rejects empty strings and any character that could be used for
    path traversal or filesystem confusion.  This is a defence-in-depth
    check; the store re-validates on load (LKB-STORE-028).
    """
    if not board_id:
        raise BoardResolutionError("board_id must not be empty")
    if _BOARD_ID_FORBIDDEN_RE.search(board_id):
        raise BoardResolutionError(f"board_id contains forbidden characters: {board_id!r}")
    # Reject common path-traversal patterns explicitly.
    if board_id in (".", "..") or board_id.startswith("."):
        raise BoardResolutionError(f"board_id must not start with a dot: {board_id!r}")
    if len(board_id) > 255:
        raise BoardResolutionError(f"board_id too long ({len(board_id)} chars, max 255)")


def _project_uri_from_root(workspace_root: str | Path) -> str:
    """Derive a project: URI from a workspace root path."""
    normalized = normalize_workspace_root(workspace_root)
    return f"project:{normalized}"


def _display_name_from_root(workspace_root: str | Path) -> str:
    """Derive a human-readable display name from the workspace basename."""
    normalized = normalize_workspace_root(workspace_root)
    basename = normalized.rstrip("/").rsplit("/", 1)[-1]
    return basename or "Workspace Board"


def _read_project_board_id(workspace_root: str | Path) -> str | None:
    """Read board_id from ``<workspace_root>/.claude/config.json``.

    Looks for ``lkb.board_id``.  Returns None if the file doesn't
    exist, isn't valid JSON, or doesn't contain the key.  This is a
    best-effort lookup — we never raise for a missing/malformed config;
    we just fall through to tier 4.
    """
    try:
        config_path = Path(workspace_root) / ".claude" / "config.json"
        if not config_path.is_file():
            return None
        # Guard against very large files.
        if config_path.stat().st_size > 1_000_000:
            return None
        data: Any = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        lkb_cfg = data.get("lkb")
        if not isinstance(lkb_cfg, dict):
            return None
        board_id = lkb_cfg.get("board_id")
        if isinstance(board_id, str) and board_id:
            return board_id
    except (OSError, ValueError):
        # Missing file, permission error, invalid JSON — all are
        # non-fatal; we fall through to tier 4.
        pass
    return None


def _derive_board_id(workspace_root: str | Path) -> str:
    """Derive a stable board ID from the workspace and repository identity.

    Author identity (``user.name`` / ``user.email``) is deliberately not
    included: it is mutable, machine-local configuration and would make the
    same checkout silently resolve to a different board.  When Git is
    available, repository identity is the credential-free origin URL plus
    the repository-relative workspace path.  A non-Git workspace falls back
    to its normalized root.
    """
    normalized_root = normalize_workspace_root(workspace_root)
    repository_key = _git_repository_identity(workspace_root)
    # A configured remote is portable across clones/machines, so the local
    # absolute checkout path must not participate.  Without a remote, path is
    # the only stable repository identity available.
    composite = repository_key if repository_key.startswith("remote:") else f"{normalized_root}|{repository_key}"
    digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]
    return f"proj-{digest}"


def _run_git(args: list[str], cwd: str | Path) -> str | None:
    """Run a read-only Git query and return its stripped output."""
    try:
        # The executable and argument shapes are fixed read-only Git queries.
        result = subprocess.run(  # nosec B607
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            return value if value else None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _credential_free_remote(remote: str) -> str:
    """Normalize HTTPS/SSH/SCP transports to one credential-free identity."""
    remote = remote.strip()
    if "://" not in remote:
        # SCP-style: [user@]host:path.  Local paths deliberately remain paths.
        match = re.fullmatch(r"(?:[^@/:]+@)?([^/:]+):(.+)", remote)
        if match:
            host, path = match.groups()
            return f"{host.lower()}/{path.lstrip('/')}".rstrip("/").removesuffix(".git")
        return remote.replace("\\", "/").rstrip("/").removesuffix(".git")

    parts = urlsplit(remote)
    hostname = parts.hostname or ""
    default_port = (
        (parts.scheme.lower() == "ssh" and parts.port == 22)
        or (parts.scheme.lower() == "https" and parts.port == 443)
        or (parts.scheme.lower() == "http" and parts.port == 80)
    )
    if parts.port is not None and not default_port:
        hostname = f"{hostname}:{parts.port}"
    return f"{hostname.lower()}/{parts.path.lstrip('/')}".rstrip("/").removesuffix(".git")


def _git_repository_identity(workspace_root: str | Path) -> str:
    """Return stable repository identity, or an empty string outside Git."""
    top_level = _run_git(["rev-parse", "--show-toplevel"], workspace_root)
    if not top_level:
        return ""

    normalized_top = normalize_workspace_root(top_level)
    normalized_workspace = normalize_workspace_root(workspace_root)
    try:
        relative = Path(normalized_workspace).relative_to(Path(normalized_top)).as_posix()
    except ValueError:
        relative = "."

    origin = _run_git(["remote", "get-url", "origin"], workspace_root)
    if origin:
        return f"remote:{_credential_free_remote(origin)}|{relative or '.'}"
    return f"path:{normalized_top}|{relative or '.'}"


def _session_board_id(session_id: str | None = None) -> str:
    """Return a stable, non-identifying session-scoped board identifier.

    A host-provided session identity is hashed so it is stable without
    leaking the raw value. Callers without one share a lazily-created
    process identity, keeping repeated operations on one board.
    """
    import uuid

    if session_id:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
        return f"session-{digest}"

    global _PROCESS_SESSION_BOARD_ID
    with _SESSION_BOARD_LOCK:
        if _PROCESS_SESSION_BOARD_ID is None:
            _PROCESS_SESSION_BOARD_ID = f"session-{uuid.uuid4().hex[:16]}"
        return _PROCESS_SESSION_BOARD_ID
