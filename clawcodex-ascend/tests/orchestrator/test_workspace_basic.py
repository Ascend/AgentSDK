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

"""Basic tests for workspace preservation policy.

Extracted from ``test_workspace_preserve.py`` to ship with the source-only
PR. Full test suite migrates in a follow-up PR.

Tests the ``_should_preserve()`` decision matrix:
- completed → preserve_on_terminal
- failed/verification_failed → preserve_on_failure
- abandoned → preserve_on_abandoned
- timeout/budget_exhausted/stagnation → preserve_on_timeout
- others/None → preserve_on_terminal (default)
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from extensions.orchestrator.workspace import WorkspaceConfig, WorkspaceManager


class _FakeIssue(SimpleNamespace):
    def __init__(self, issue_id: str = "issue-1", identifier: str = "test-issue-1") -> None:
        super().__init__(id=issue_id, identifier=identifier)


class TestShouldPreserve(unittest.TestCase):
    """Test the _should_preserve() decision matrix."""

    def setUp(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self._tmpdir), ignore_errors=True)

    def _make_manager(self, **kwargs: bool) -> WorkspaceManager:
        config = WorkspaceConfig(
            root=self._tmpdir,
            preserve_on_terminal=kwargs.get("preserve_on_terminal", True),
            preserve_on_failure=kwargs.get("preserve_on_failure", True),
            preserve_on_abandoned=kwargs.get("preserve_on_abandoned", True),
            preserve_on_timeout=kwargs.get("preserve_on_timeout", True),
        )
        return WorkspaceManager(config)

    def test_completed_preserve_when_enabled(self) -> None:
        mgr = self._make_manager(preserve_on_terminal=True)
        self.assertTrue(mgr._should_preserve("completed", "task_complete"))

    def test_completed_delete_when_disabled(self) -> None:
        mgr = self._make_manager(preserve_on_terminal=False)
        self.assertFalse(mgr._should_preserve("completed", "task_complete"))

    def test_failed_preserve_when_enabled(self) -> None:
        mgr = self._make_manager(preserve_on_failure=True)
        self.assertTrue(mgr._should_preserve("failed", None))

    def test_failed_delete_when_disabled(self) -> None:
        mgr = self._make_manager(preserve_on_failure=False)
        self.assertFalse(mgr._should_preserve("failed", None))

    def test_verification_failed_preserve(self) -> None:
        mgr = self._make_manager(preserve_on_failure=True)
        self.assertTrue(mgr._should_preserve("verification_failed", None))

    def test_abandoned_preserve_when_enabled(self) -> None:
        mgr = self._make_manager(preserve_on_abandoned=True)
        self.assertTrue(mgr._should_preserve("abandoned", "stagnation"))

    def test_abandoned_delete_when_disabled(self) -> None:
        mgr = self._make_manager(preserve_on_abandoned=False)
        self.assertFalse(mgr._should_preserve("abandoned", "stagnation"))

    def test_budget_exhausted_preserve_when_enabled(self) -> None:
        mgr = self._make_manager(preserve_on_timeout=True)
        self.assertTrue(mgr._should_preserve("failed", "budget_exhausted"))

    def test_budget_exhausted_delete_when_disabled(self) -> None:
        mgr = self._make_manager(preserve_on_timeout=False)
        self.assertFalse(mgr._should_preserve("failed", "budget_exhausted"))

    def test_stagnation_preserve(self) -> None:
        mgr = self._make_manager(preserve_on_abandoned=True)
        self.assertTrue(mgr._should_preserve("abandoned", "stagnation"))

    def test_stagnation_delete_when_abandoned_disabled(self) -> None:
        mgr = self._make_manager(preserve_on_abandoned=False)
        self.assertFalse(mgr._should_preserve("abandoned", "stagnation"))

    def test_loop_detected_preserve(self) -> None:
        mgr = self._make_manager(preserve_on_abandoned=True)
        self.assertTrue(mgr._should_preserve("abandoned", "loop_detected"))

    def test_none_status_uses_preserve_on_terminal(self) -> None:
        mgr_enabled = self._make_manager(preserve_on_terminal=True)
        self.assertTrue(mgr_enabled._should_preserve(None, None))

        mgr_disabled = self._make_manager(preserve_on_terminal=False)
        self.assertFalse(mgr_disabled._should_preserve(None, None))

    def test_unknown_status_uses_preserve_on_terminal(self) -> None:
        mgr = self._make_manager(preserve_on_terminal=False)
        self.assertFalse(mgr._should_preserve("some_unknown_status", None))

    def test_cancelled_uses_preserve_on_terminal(self) -> None:
        mgr_enabled = self._make_manager(preserve_on_terminal=True)
        self.assertTrue(mgr_enabled._should_preserve("cancelled", "operator_stopped"))

        mgr_disabled = self._make_manager(preserve_on_terminal=False)
        self.assertFalse(mgr_disabled._should_preserve("cancelled", "operator_stopped"))

    def test_case_insensitive_status(self) -> None:
        mgr = self._make_manager(preserve_on_failure=True)
        self.assertTrue(mgr._should_preserve("FAILED", None))
        self.assertTrue(mgr._should_preserve("Failed", None))

    def test_case_insensitive_reason(self) -> None:
        mgr = self._make_manager(preserve_on_timeout=True)
        self.assertTrue(mgr._should_preserve("failed", "BUDGET_EXHAUSTED"))


if __name__ == "__main__":
    unittest.main()
