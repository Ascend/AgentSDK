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

"""P108-G — auto-recovery strategy catalogue tests."""

from __future__ import annotations

import unittest

from clawcodex_ext.diagnostics import (
    RecoveryAction,
    RecoverySpec,
    describe,
    recovery_actions,
)


class TestRecoveryTable(unittest.TestCase):
    def test_five_paths_match_plan(self):
        actions = {spec.action for spec in recovery_actions()}
        self.assertIn(RecoveryAction.PERMISSION_AUTO_DENY, actions)
        self.assertIn(RecoveryAction.ASK_USER_EMPTY, actions)
        self.assertIn(RecoveryAction.LLM_TURN_TIMEOUT, actions)
        self.assertIn(RecoveryAction.TOOL_TIMEOUT, actions)
        self.assertIn(RecoveryAction.AGENT_LOOP_TIMEOUT, actions)
        self.assertEqual(len(actions), 5)

    def test_describe_returns_spec(self):
        spec = describe(RecoveryAction.PERMISSION_AUTO_DENY)
        self.assertIsInstance(spec, RecoverySpec)
        self.assertEqual(spec.action, RecoveryAction.PERMISSION_AUTO_DENY)
        self.assertIn("agent_bridge", spec.integration_point)

    def test_describe_raises_for_unknown(self):
        # RecoveryAction is an Enum so we can't construct an unknown
        # value; ensure the str-based path raises for an invalid
        # value via enum_value coercion.
        with self.assertRaises(KeyError):
            describe("not_a_real_action")  # type: ignore[arg-type]

    def test_user_perception_strings_have_chinese_or_arrows(self):
        # Loose check — the plan categorised each path with a Chinese
        # perception phrase or a short English tag. Pin that we never
        # surfaced an empty user-perception string after a refactor.
        for spec in recovery_actions():
            self.assertTrue(spec.user_perception, f"{spec.action} missing perception")
            self.assertTrue(spec.mechanism, f"{spec.action} missing mechanism")
            self.assertTrue(spec.integration_point, f"{spec.action} missing integration_point")


if __name__ == "__main__":
    unittest.main()
