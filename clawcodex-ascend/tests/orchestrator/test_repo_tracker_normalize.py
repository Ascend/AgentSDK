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

from extensions.orchestrator.repo_tracker.normalizers import _normalize_mergeable_status


class TestNormalizeMergeableStatus(unittest.TestCase):
    """Tests against the GitHub/Gitee/GitCode response shapes."""

    def test_github_clean(self) -> None:
        # GitHub returns `mergeable: true, mergeable_state: "clean"`.
        raw = {"mergeable": True, "mergeable_state": "clean", "behind_by": 0}
        s = _normalize_mergeable_status(raw, platform="github")
        self.assertIs(s.mergeable, True)
        self.assertEqual(s.mergeable_state, "clean")
        self.assertFalse(s.has_conflicts)
        self.assertEqual(s.raw["platform"], "github")
        self.assertEqual(s.raw["payload"], raw)

    def test_github_dirty(self) -> None:
        raw = {"mergeable": False, "mergeable_state": "dirty"}
        s = _normalize_mergeable_status(raw, platform="github")
        self.assertIs(s.mergeable, False)
        self.assertTrue(s.has_conflicts)

    def test_gitee_dirty_state(self) -> None:
        # Gitee uses mergeable_state in {"clean", "dirty", "blocked", ...}.
        raw = {"mergeable": False, "mergeable_state": "dirty"}
        s = _normalize_mergeable_status(raw, platform="gitee")
        self.assertEqual(s.mergeable_state, "dirty")
        self.assertTrue(s.has_conflicts)

    def test_gitee_mergeable(self) -> None:
        # Gitee "mergeable" = True when no conflict.
        raw = {"mergeable": True, "mergeable_state": "clean"}
        s = _normalize_mergeable_status(raw, platform="gitee")
        self.assertFalse(s.has_conflicts)

    def test_gitcode_missing_mergeable(self) -> None:
        # GitCode does not always include the `mergeable` field; the
        # normalizer should fall back to has_conflicts=False so the
        # daemon PR scan is a no-op (operator must use CLI / label /
        # comment to trigger rebase).
        raw = {"mergeable_state": "clean"}
        s = _normalize_mergeable_status(raw, platform="gitcode")
        self.assertIsNone(s.mergeable)
        self.assertFalse(s.has_conflicts)

    def test_gitcode_state_only_dirty(self) -> None:
        raw = {"mergeable_state": "dirty"}
        s = _normalize_mergeable_status(raw, platform="gitcode")
        self.assertEqual(s.mergeable_state, "dirty")
        self.assertTrue(s.has_conflicts)

    def test_gitcode_nested_conflict_passed(self) -> None:
        # GitCode can return a nested mergeable_state dict with
        # conflict_passed flag.
        raw = {
            "mergeable": None,
            "mergeable_state": {
                "state": "open",
                "conflict_passed": False,
            },
        }
        s = _normalize_mergeable_status(raw, platform="gitcode")
        self.assertTrue(s.has_conflicts)

    def test_gitcode_nested_conflict_passed_true(self) -> None:
        raw = {
            "mergeable": None,
            "mergeable_state": {
                "state": "open",
                "conflict_passed": True,
            },
        }
        s = _normalize_mergeable_status(raw, platform="gitcode")
        self.assertFalse(s.has_conflicts)

    def test_ahead_behind_populated(self) -> None:
        raw = {"ahead_by": 3, "behind_by": 5, "mergeable": True}
        s = _normalize_mergeable_status(raw, platform="github")
        self.assertEqual(s.ahead_by, 3)
        self.assertEqual(s.behind_by, 5)

    def test_string_mergeable_normalized(self) -> None:
        # Some APIs return "true"/"false" as strings.
        raw = {"mergeable": "true", "mergeable_state": "clean"}
        s = _normalize_mergeable_status(raw, platform="github")
        self.assertIs(s.mergeable, True)
        raw2 = {"mergeable": "false", "mergeable_state": "dirty"}
        s2 = _normalize_mergeable_status(raw2, platform="github")
        self.assertIs(s2.mergeable, False)
        self.assertTrue(s2.has_conflicts)
