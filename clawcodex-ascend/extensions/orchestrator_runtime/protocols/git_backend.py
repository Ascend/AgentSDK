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

"""GitBackend protocol.

Replaces direct calls from ``extensions/orchestrator`` into
``clawcodex_ext.utils.git._run_git``. ``DefaultGitBackend`` in utils is
a subprocess wrapper; a Clawcodex backend can wrap ``_run_git`` later.
"""

from __future__ import annotations
# pylint: disable=W2301

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class FileStatusLike(Protocol):
    """Structural type — runtime doesn't introspect fields beyond :attr:`path`
    and :attr:`status`. Concrete impl may come from
    ``clawcodex_ext.utils.git.FileStatus`` (frozen=True) or from
    ``extensions.orchestrator_runtime.utils.git_backend_impl.FileStatus``
    (slots=True). Both satisfy this Protocol.
    """

    path: str
    status: str

    @property
    def is_modified(self) -> bool: ...

    @property
    def is_added(self) -> bool: ...

    @property
    def is_deleted(self) -> bool: ...

    @property
    def is_renamed(self) -> bool: ...


class GitBackend(Protocol):
    """Shim over ``git`` CLI. Implementations can swap for libgit2 later.

    All methods are synchronous from the agent's POV; agent_runner wraps
    blocking calls with ``asyncio.to_thread`` if needed (Phase 3 will
    standardise async wrappers).
    """

    def status(self, repo_root: Path) -> list[FileStatusLike]:
        """Return list of changed files (incl. untracked). Empty if clean."""
        ...

    def current_branch(self, repo_root: Path) -> str | None:
        """Return active branch name; ``None`` for detached HEAD."""
        ...

    def default_branch(self, repo_root: Path) -> str:
        """Resolve default branch (e.g. ``main`` / ``master``)."""
        ...

    def remote_url(self, repo_root: Path) -> str:
        """Return ``origin`` URL (or empty string if no remote)."""
        ...

    def run(self, args: list[str], cwd: Path, *, check: bool = True) -> str:
        """Run raw git command; returns stdout. ``check=False`` returns ""
        on non-zero exit without raising.
        """
        ...

    def fetch(self, repo_root: Path, remote: str = "origin") -> None: ...

    def push(self, repo_root: Path, *, force: bool = False, set_upstream: bool = False) -> None: ...

    def rebase(self, repo_root: Path, upstream: str) -> None: ...


__all__ = ["FileStatusLike", "GitBackend"]
