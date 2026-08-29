#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Basic tests for git_sync_rebase module.

Tests PRRebaseResult dataclass, _slugify, _ahead_behind,
_git_rebase_abort, and rebase_for_pr using real (in-process) git repos.

``clawcodex_compat`` is not yet migrated; a minimal working stub is
injected via ``sys.modules`` so the real git subprocess calls work.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# clawcodex_compat stub (not yet migrated to AgentSDK)
# Provides _run_git and get_current_branch so git_sync_rebase can import.
# ---------------------------------------------------------------------------


def _run_git(
    args: list[str],
    cwd: str | None = None,
    timeout: float = 30.0,
) -> tuple[str, str, int]:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            timeout=timeout,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1
    except FileNotFoundError:
        return "", "git not found", -1


def _get_current_branch(cwd: str | None = None) -> str | None:
    stdout, _stderr, rc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    return stdout or None if rc == 0 else None


_compat = types.ModuleType("extensions.orchestrator_runtime.adapters.clawcodex_compat")
_compat._run_git = _run_git
_compat.get_current_branch = _get_current_branch

for _name in (
    "extensions.orchestrator_runtime",
    "extensions.orchestrator_runtime.adapters",
):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
sys.modules["extensions.orchestrator_runtime.adapters.clawcodex_compat"] = _compat

# ---------------------------------------------------------------------------
# Now safe to import git_sync_rebase
# ---------------------------------------------------------------------------

from extensions.orchestrator.git_sync_rebase import (  # noqa: E402
    GitSyncError,
    PRRebaseResult,
    _ahead_behind,
    _git_rebase_abort,
    _slugify,
    rebase_for_pr,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _init_repo(path: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    subprocess.check_call(["git", "init", "-q", "-b", "main", str(path)], env=env)
    subprocess.check_call(["git", "config", "user.email", "t@example.com"], cwd=path, env=env)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=path, env=env)
    subprocess.check_call(["git", "config", "commit.gpgsign", "false"], cwd=path, env=env)
    (path / "README.md").write_text("init\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "README.md"], cwd=path, env=env)
    subprocess.check_call(["git", "commit", "-q", "-m", "init"], cwd=path, env=env)


def _make_remote(tmp: Path) -> Path:
    remote = tmp / "origin.git"
    subprocess.check_call(["git", "init", "-q", "--bare", str(remote)])
    return remote


def _push_initial(tmp: Path, path: Path, remote: Path, branch: str, base: str) -> None:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    subprocess.check_call(["git", "remote", "add", "origin", str(remote)], cwd=path, env=env)
    subprocess.check_call(["git", "push", "-q", "origin", branch], cwd=path, env=env)
    if branch != base:
        subprocess.check_call(["git", "branch", base, branch], cwd=path, env=env)
        subprocess.check_call(["git", "push", "-q", "origin", base], cwd=path, env=env)
        subprocess.check_call(["git", "checkout", "-q", base], cwd=path, env=env)


def _commit(path: Path, message: str, files: dict[str, str] | None = None) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    if files:
        for name, content in files.items():
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        subprocess.check_call(["git", "add", "-A"], cwd=path, env=env)
    subprocess.check_output(["git", "commit", "-q", "-m", message], cwd=path, env=env, text=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, env=env, text=True).strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSlugify(unittest.TestCase):
    def test_basic_slug(self) -> None:
        self.assertEqual(_slugify("Hello World"), "hello-world")

    def test_collapses_dashes(self) -> None:
        self.assertEqual(_slugify("a---b"), "a-b")

    def test_empty_returns_default(self) -> None:
        self.assertEqual(_slugify(""), "issue-update")

    def test_strips_leading_trailing_dashes(self) -> None:
        self.assertEqual(_slugify("--abc--"), "abc")


class TestGitSyncError(unittest.TestCase):
    def test_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(GitSyncError, RuntimeError))

    def test_raisable(self) -> None:
        with self.assertRaises(GitSyncError):
            raise GitSyncError("test")


class TestPRRebaseResultDataclass(unittest.TestCase):
    def test_default_values(self) -> None:
        r = PRRebaseResult(rebased=False)
        self.assertFalse(r.rebased)
        self.assertFalse(r.has_conflict)
        self.assertEqual(r.conflict_files, ())
        self.assertIsNone(r.new_head_sha)
        self.assertFalse(r.pushed)
        self.assertEqual(r.push_method, "none")
        self.assertTrue(r.workspace_clean)

    def test_conflict_result(self) -> None:
        r = PRRebaseResult(
            rebased=False,
            has_conflict=True,
            conflict_files=("a.py", "b.py"),
        )
        self.assertTrue(r.has_conflict)
        self.assertEqual(r.conflict_files, ("a.py", "b.py"))

    def test_frozen(self) -> None:
        r = PRRebaseResult(rebased=True)
        with self.assertRaises(Exception):
            r.rebased = False  # type: ignore[misc]


class TestAheadBehind(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="a3_rebase_ab_"))
        self.path = self.tmp / "repo"
        self.path.mkdir()
        _init_repo(self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_zero_zero_when_same_commit(self) -> None:
        ahead, behind = _ahead_behind(str(self.path), "main", "main")
        self.assertEqual((ahead, behind), (0, 0))

    def test_parse_failure_returns_zero_zero(self) -> None:
        ahead, behind = _ahead_behind(str(self.path), "main", "nonexistent-branch-xyz")
        self.assertEqual((ahead, behind), (0, 0))


class TestGitRebaseAbort(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="a3_rebase_abort_"))
        self.path = self.tmp / "repo"
        self.path.mkdir()
        _init_repo(self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_abort_silently_when_no_rebase(self) -> None:
        try:
            _git_rebase_abort(str(self.path))
        except Exception as exc:
            self.fail(f"_git_rebase_abort raised unexpectedly: {exc}")


class TestRebaseForPr(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="a3_rebase_"))
        self.path = self.tmp / "repo"
        self.path.mkdir()
        self.remote = _make_remote(self.tmp)
        _init_repo(self.path)
        _push_initial(self.tmp, self.path, self.remote, "main", "main")
        subprocess.check_call(["git", "checkout", "-q", "-b", "feature"], cwd=self.path)
        _commit(self.path, "feature-1", {"feature.txt": "feature-1\n"})
        subprocess.check_call(["git", "push", "-q", "origin", "feature"], cwd=self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_already_up_to_date(self) -> None:
        result = rebase_for_pr(
            workspace_path=str(self.path),
            branch_name="feature",
            base_branch="main",
        )
        self.assertTrue(result.rebased)
        self.assertFalse(result.has_conflict)
        self.assertFalse(result.pushed)
        self.assertEqual(result.push_method, "none")
        self.assertTrue(result.workspace_clean)

    def test_clean_rebase_force_with_lease(self) -> None:
        subprocess.check_call(["git", "checkout", "-q", "main"], cwd=self.path)
        _commit(self.path, "main-2", {"main.txt": "main-2\n"})
        subprocess.check_call(["git", "push", "-q", "origin", "main"], cwd=self.path)
        subprocess.check_call(["git", "checkout", "-q", "feature"], cwd=self.path)
        result = rebase_for_pr(
            workspace_path=str(self.path),
            branch_name="feature",
            base_branch="main",
        )
        self.assertTrue(result.rebased)
        self.assertFalse(result.has_conflict)
        self.assertTrue(result.pushed)
        self.assertEqual(result.push_method, "force_with_lease")
        self.assertTrue(result.workspace_clean)
        self.assertIsNotNone(result.new_head_sha)

    def test_content_conflict(self) -> None:
        subprocess.check_call(["git", "checkout", "-q", "main"], cwd=self.path)
        _commit(self.path, "main-conflict", {"README.md": "main-side\n"})
        subprocess.check_call(["git", "push", "-q", "origin", "main"], cwd=self.path)
        subprocess.check_call(["git", "checkout", "-q", "feature"], cwd=self.path)
        _commit(self.path, "feature-conflict", {"README.md": "feature-side\n"})
        subprocess.check_call(["git", "push", "-q", "origin", "feature"], cwd=self.path)
        result = rebase_for_pr(
            workspace_path=str(self.path),
            branch_name="feature",
            base_branch="main",
        )
        self.assertFalse(result.rebased)
        self.assertTrue(result.has_conflict)
        self.assertIn("README.md", result.conflict_files)
        self.assertFalse(result.workspace_clean)
        self.assertFalse(result.pushed)


if __name__ == "__main__":
    unittest.main()
