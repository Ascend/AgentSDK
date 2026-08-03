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

"""F-108 P108-E — schema integration tests for FreezeSettings."""

from __future__ import annotations

import unittest

from clawcodex_ext.settings.types import FreezeSettings, SettingsSchema


class TestSettingsSchemaFreeze(unittest.TestCase):
    def test_schema_has_freeze_field(self):
        s = SettingsSchema()
        self.assertIsInstance(s.freeze, FreezeSettings)
        # Defaults match the dataclass defaults.
        self.assertEqual(s.freeze.permission_timeout_s, 30.0)
        self.assertEqual(s.freeze.threshold_s, 60.0)
        self.assertEqual(s.freeze.tool_timeout_s, 120.0)
        self.assertEqual(s.freeze.turn_timeout_s, 300.0)
        self.assertEqual(s.freeze.agent_loop_timeout_s, 600.0)

    def test_from_dict_loads_freeze_block(self):
        raw = {
            "model": "claude-sonnet-4-6",
            "freeze": {
                "permission_timeout_s": 45.0,
                "tool_timeout_s": 200.0,
                "threshold_s": 90.0,
                "agent_loop_timeout_s": 1200.0,
                "turn_timeout_s": 600.0,
            },
        }
        s = SettingsSchema.from_dict(raw)
        self.assertEqual(s.freeze.permission_timeout_s, 45.0)
        self.assertEqual(s.freeze.tool_timeout_s, 200.0)
        self.assertEqual(s.freeze.threshold_s, 90.0)

    def test_to_dict_round_trips_freeze_block(self):
        s = SettingsSchema()
        s.freeze = FreezeSettings(permission_timeout_s=12.5, tool_timeout_s=99.9)
        d = s.to_dict()
        # freeze block survives to_dict
        self.assertIn("freeze", d)
        self.assertEqual(d["freeze"]["permission_timeout_s"], 12.5)
        self.assertEqual(d["freeze"]["tool_timeout_s"], 99.9)

    def test_round_trip_from_dict_to_dict(self):
        raw = {
            "freeze": {
                "permission_timeout_s": 12.0,
                "tool_timeout_s": 88.0,
                "threshold_s": 33.0,
                "turn_timeout_s": 222.0,
                "agent_loop_timeout_s": 555.0,
                "dump_dir": "/tmp/example-freeze",
            }
        }
        s = SettingsSchema.from_dict(raw)
        d = s.to_dict()
        self.assertEqual(d["freeze"]["permission_timeout_s"], 12.0)
        self.assertEqual(d["freeze"]["dump_dir"], "/tmp/example-freeze")


if __name__ == "__main__":
    unittest.main()
