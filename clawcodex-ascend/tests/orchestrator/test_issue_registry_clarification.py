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

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from typing import Any

from extensions.orchestrator.issue_registry import (
    IssueRecord,
    IssueRegistry,
)


def _make_registry(tmp_dir: str | Path) -> IssueRegistry:
    return IssueRegistry(storage_path=Path(tmp_dir) / "registry.json")


def _make_record(
    issue_id: str = "i1",
    issue_identifier: str = "owner/repo#1",
    **kwargs: Any,
) -> IssueRecord:
    return IssueRecord(issue_id=issue_id, issue_identifier=issue_identifier, **kwargs)


class TestClarificationMutations(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.registry = _make_registry(self.tmp.name)
        self.registry.register("i1", "owner/repo#1")

    def test_update_clarification(self) -> None:
        result = self.registry.update_clarification(
            "i1",
            clarification_status="awaiting_local",
            question="which color?",
            author_login="alice",
            local_answer="blue",
            local_answer_source="dashboard",
            first_response_source="local",
        )
        self.assertEqual(result.clarification_status, "awaiting_local")
        self.assertEqual(result.question_history, ["which color?"])
        self.assertEqual(result.author_login, "alice")
        self.assertEqual(result.local_answer, "blue")
        self.assertEqual(result.local_answer_source, "dashboard")
        self.assertEqual(result.first_response_source, "local")

    def test_update_clarification_appends_question_history(self) -> None:
        self.registry.update_clarification("i1", question="q1")
        self.registry.update_clarification("i1", question="q2")
        self.assertEqual(self.registry.get("i1").question_history, ["q1", "q2"])

    def test_add_stale_answer(self) -> None:
        self.registry.add_stale_answer("i1", "stale1")
        self.registry.add_stale_answer("i1", "stale2")
        self.assertEqual(self.registry.get("i1").stale_answers, ["stale1", "stale2"])

    def test_update_clarification_missing(self) -> None:
        self.assertIsNone(self.registry.update_clarification("missing", clarification_status="x"))


if __name__ == "__main__":
    unittest.main()
