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

"""Basic tests for git_sync main module (GitSyncService + dataclasses).

All unmigrated dependencies are stubbed via sys.modules.
"""

from __future__ import annotations

import sys
import types
import unittest

# --- compact stub registration ---


def _reg(name: str, **attrs: object) -> None:
    _SKIP = {"extensions", "extensions.orchestrator"}
    parts = name.split(".")
    for i in range(1, len(parts)):
        p = ".".join(parts[:i])
        if p not in _SKIP and p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


_reg(
    "extensions.orchestrator_runtime.adapters.clawcodex_compat",
    _run_git=lambda *a, **k: ("", "", 0),
    get_current_branch=lambda *a, **k: None,
    get_default_branch=lambda *a, **k: "main",
    get_file_status=lambda *a, **k: "",
    get_repo_root=lambda *a, **k: "",
)
_reg(
    "extensions.orchestrator.config.schema",
    AgentConfig=type("AgentConfig", (), {}),
    HooksConfig=type("HooksConfig", (), {"pre_commit": "", "post_sync": "", "timeout_ms": 30000}),
    PrTemplateConfig=type("PrTemplateConfig", (), {"title": "", "body": ""}),
)
_reg("extensions.orchestrator.issue", Issue=type("Issue", (), {}))
_reg("extensions.orchestrator.prompt_builder", resolve_python_executable=lambda: "python3")
_reg(
    "extensions.orchestrator.tracker",
    PullRequestCapability=type("PullRequestCapability", (), {}),
    PullRequestMaintenanceCapability=type("PullRequestMaintenanceCapability", (), {}),
    PullRequestRef=type("PullRequestRef", (), {}),
    TrackerAdapter=type("TrackerAdapter", (), {}),
    supports=lambda *a, **k: False,
)
_reg("extensions.orchestrator.git_sync_ops", GitSyncOpsMixin=type("GitSyncOpsMixin", (), {}))
_reg(
    "extensions.orchestrator.git_sync_rebase",
    GitSyncError=type("GitSyncError", (RuntimeError,), {}),
    PRRebaseResult=type("PRRebaseResult", (), {}),
    _ahead_behind=lambda *a, **k: (0, 0),
    _git_rebase_abort=lambda *a, **k: None,
    rebase_for_pr=lambda *a, **k: None,
)

# --- import after stubs ---

from extensions.orchestrator.git_sync import (  # noqa: E402
    GitSyncError,
    GitSyncPostCommitError,
    GitSyncResult,
    GitSyncService,
    HookFailedError,
    VerificationFailed,
)


class TestGitSyncResult(unittest.TestCase):
    def test_defaults(self) -> None:
        r = GitSyncResult(branch_name="feature", base_branch="main")
        self.assertFalse(r.pushed)
        self.assertIsNone(r.commit_sha)

    def test_with_values(self) -> None:
        r = GitSyncResult(branch_name="f", base_branch="main", pushed=True)
        self.assertTrue(r.pushed)


class TestExceptionHierarchy(unittest.TestCase):
    def test_verification_failed_is_git_sync_error(self) -> None:
        self.assertTrue(issubclass(VerificationFailed, GitSyncError))

    def test_hook_failed_is_git_sync_error(self) -> None:
        self.assertTrue(issubclass(HookFailedError, GitSyncError))

    def test_post_commit_error_wraps_cause(self) -> None:
        cause = VerificationFailed("fail")
        r = GitSyncResult(branch_name="f", base_branch="main")
        err = GitSyncPostCommitError(cause, r)
        self.assertIs(err.cause, cause)


class TestGitSyncServiceInit(unittest.TestCase):
    def test_default_gitignore_includes_artifacts(self) -> None:
        svc = GitSyncService(tracker=object())
        self.assertIn("analysis.md", svc._gitignore_patterns)
        self.assertIn("verification_report.md", svc._gitignore_patterns)


if __name__ == "__main__":
    unittest.main()
