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

"""Basic tests for GitSyncOpsMixin (git_sync_ops module).

``clawcodex_compat``, ``issue``, ``git_sync_rebase``, and ``tracker``
are not yet migrated to this branch; minimal stubs injected via
``sys.modules`` so the mixin can be imported and tested.
"""

from __future__ import annotations

import re
import sys
import types
import unittest
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Stub modules (not yet migrated to this branch)
# ---------------------------------------------------------------------------


def _stub_run_git(args: list[str], cwd: str | None = None, timeout: float = 30.0) -> tuple[str, str, int]:
    return "", "", 0


def _stub_get_current_branch(cwd: str | None = None) -> str | None:
    return None


class GitSyncError(RuntimeError):
    pass


def _git_rebase_abort(repo_root: str) -> None:
    pass


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9._/-]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-") or "issue-update"


@dataclass
class _Issue:
    id: str | None = None
    identifier: str | None = None
    title: str | None = None
    url: str | None = None


@dataclass
class _PullRequestRef:
    number: str | None = None
    url: str | None = None
    title: str | None = None


@dataclass
class _PrTemplateConfig:
    title: str = ""
    body: str = ""


# --- Register stubs in sys.modules ---


def _ensure_parent(dotted_name: str) -> None:
    """Create parent packages in sys.modules only if they don't exist.
    Skip real on-disk packages — let Python import them naturally.
    """
    _REAL_PACKAGES = {"extensions", "extensions.orchestrator"}
    parts = dotted_name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent in _REAL_PACKAGES:
            continue
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)


def _register_stub(dotted_name: str, **attrs: Any) -> None:
    _ensure_parent(dotted_name)
    mod = types.ModuleType(dotted_name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[dotted_name] = mod


_register_stub(
    "extensions.orchestrator_runtime.adapters.clawcodex_compat",
    _run_git=_stub_run_git,
    get_current_branch=_stub_get_current_branch,
)
_register_stub(
    "extensions.orchestrator.git_sync_rebase",
    GitSyncError=GitSyncError,
    _git_rebase_abort=_git_rebase_abort,
    _slugify=_slugify,
)
_register_stub("extensions.orchestrator.issue", Issue=_Issue)
_register_stub("extensions.orchestrator.tracker", PullRequestRef=_PullRequestRef)

# ---------------------------------------------------------------------------
# Import after stubs are registered
# ---------------------------------------------------------------------------

from extensions.orchestrator.git_sync_ops import GitSyncOpsMixin  # noqa: E402

Issue = _Issue
PullRequestRef = _PullRequestRef
PrTemplateConfig = _PrTemplateConfig


class _TestService(GitSyncOpsMixin):
    def __init__(self, pr_template=None, branch_prefix=None):
        self._pr_template = pr_template or PrTemplateConfig()
        self._branch_prefix = branch_prefix


# ---------------------------------------------------------------------------
# Tests (2-3 core cases — full suite in A.3-tests PR)
# ---------------------------------------------------------------------------


class TestRenderPrTemplate(unittest.TestCase):
    def test_renders_known_and_unknown_variables(self) -> None:
        result = GitSyncOpsMixin._render_pr_template(
            "Known={{ issue.identifier }}, unknown={{ no_such_value }}",
            {"issue.identifier": "ISSUE-1"},
        )
        self.assertEqual(result, "Known=ISSUE-1, unknown=")


class TestBuildPrTitle(unittest.TestCase):
    def test_uses_template_when_set(self) -> None:
        svc = _TestService(pr_template=PrTemplateConfig(title="PR: {{ issue.identifier }} / {{ issue.title }}"))
        issue = Issue(id="77", identifier="ISSUE-77", title="Fix bug")
        self.assertEqual(svc._build_pr_title(issue), "PR: ISSUE-77 / Fix bug")


class TestMergePrRef(unittest.TestCase):
    def test_preserves_existing_number_and_url(self) -> None:
        svc = _TestService()
        result = svc._merge_pr_ref(
            PullRequestRef(title="New title"),
            PullRequestRef(number="9", url="https://x/pr/9", title="Old"),
        )
        self.assertEqual(result.number, "9")
        self.assertEqual(result.url, "https://x/pr/9")
        self.assertEqual(result.title, "New title")


if __name__ == "__main__":
    unittest.main()
