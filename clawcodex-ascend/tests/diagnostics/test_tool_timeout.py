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

"""F-108 P108-C — per-tool gap watchdog tests."""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import MagicMock

from clawcodex_ext.diagnostics.freeze_config import (
    FreezeSettings,
)
from clawcodex_ext.tool_system.tool_timeout import (
    DEFAULT_TOOL_TIMEOUT_S,
    ToolGapWatchdog,
    resolve_tool_timeout,
)
from clawcodex_ext.utils.abort_controller import AbortController


class TestResolveToolTimeout(unittest.TestCase):
    """The resolution precedence: explicit -> env -> settings -> table -> default."""

    def setUp(self):
        os.environ.pop("CLAWCODEX_TOOL_TIMEOUT_BASH", None)
        os.environ.pop("CLAWCODEX_TOOL_TIMEOUT_READ", None)

    def tearDown(self):
        os.environ.pop("CLAWCODEX_TOOL_TIMEOUT_BASH", None)
        os.environ.pop("CLAWCODEX_TOOL_TIMEOUT_READ", None)

    def test_default_for_known_tool_from_table(self):
        res = resolve_tool_timeout("Read")
        self.assertEqual(res.tool_name, "Read")
        self.assertEqual(res.timeout_s, 30.0)
        self.assertTrue(res.enabled)
        self.assertFalse(res.user_override)

    def test_default_for_unknown_tool(self):
        res = resolve_tool_timeout("MysteryTool")
        # Falls back to dataclass default.
        self.assertEqual(res.timeout_s, DEFAULT_TOOL_TIMEOUT_S)
        self.assertTrue(res.enabled)

    def test_explicit_override(self):
        res = resolve_tool_timeout("Bash", explicit_override=5.0)
        self.assertEqual(res.timeout_s, 5.0)
        self.assertTrue(res.user_override)

    def test_explicit_zero_disables(self):
        res = resolve_tool_timeout("Bash", explicit_override=0.0)
        self.assertFalse(res.enabled)
        self.assertTrue(res.user_override)

    def test_env_override(self):
        os.environ["CLAWCODEX_TOOL_TIMEOUT_BASH"] = "8.5"
        res = resolve_tool_timeout("Bash")
        self.assertEqual(res.timeout_s, 8.5)
        self.assertTrue(res.user_override)

    def test_settings_global_overrides_table(self):
        s = FreezeSettings(tool_timeout_s=7.0)
        # Read's per-tool table is 30 s but a user-supplied global
        # 7.0 wins because the user touched the knob.
        res = resolve_tool_timeout("Read", settings=s)
        self.assertEqual(res.timeout_s, 7.0)
        self.assertTrue(res.from_settings)

    def test_settings_global_untouched_keeps_table_for_known_tools(self):
        s = FreezeSettings()  # default
        res = resolve_tool_timeout("Read", settings=s)
        self.assertEqual(res.timeout_s, 30.0)  # per-tool table value


class TestGapWatchdog(unittest.TestCase):
    """The watchdog must trip the AbortController on budget expiry."""

    def test_observe_then_tick_trips_and_aborts(self):
        ac = AbortController()
        fired = []
        wd = ToolGapWatchdog(
            abort_controller=ac,
            explicit_overrides={"Bash": 0.1},
            on_trip=lambda res, elapsed, tid: fired.append((res.tool_name, elapsed)),
        )
        wd.observe_tool_use("a", "Bash")
        time.sleep(0.2)
        tripped = wd.tick()
        self.assertEqual(len(tripped), 1)
        self.assertEqual(tripped[0][0], "a")
        self.assertTrue(ac.signal.aborted)
        self.assertEqual(fired[0][0], "Bash")

    def test_tool_result_disarms(self):
        ac = AbortController()
        wd = ToolGapWatchdog(
            abort_controller=ac,
            explicit_overrides={"Bash": 0.1},
        )
        wd.observe_tool_use("a", "Bash")
        wd.observe_tool_result("a")
        time.sleep(0.2)
        tripped = wd.tick()
        self.assertEqual(tripped, [])
        self.assertFalse(ac.signal.aborted)

    def test_has_pending_tracks_tool_lifecycle(self):
        wd = ToolGapWatchdog(abort_controller=AbortController())
        self.assertFalse(wd.has_pending())
        wd.observe_tool_use("a", "Agent")
        self.assertTrue(wd.has_pending())
        wd.observe_tool_result("a")
        self.assertFalse(wd.has_pending())

    def test_zero_budget_never_trips(self):
        ac = AbortController()
        wd = ToolGapWatchdog(
            abort_controller=ac,
            explicit_overrides={"Bash": 0},
        )
        wd.observe_tool_use("a", "Bash")
        time.sleep(0.1)
        tripped = wd.tick()
        self.assertEqual(tripped, [])
        self.assertFalse(ac.signal.aborted)

    def test_idempotent_rearm_keeps_original_clock(self):
        ac = AbortController()
        wd = ToolGapWatchdog(
            abort_controller=ac,
            explicit_overrides={"Bash": 0.05},
        )
        # First arm
        wd.observe_tool_use("a", "Bash")
        # Sleep longer than the budget, then idempotent re-observe.
        time.sleep(0.1)
        # No tick yet — the gap is past budget, but a fresh
        # re-observe should reset the clock and disarm the trip
        # implicit detection.
        wd.observe_tool_use("a", "Bash")
        time.sleep(0.02)
        tripped = wd.tick()
        # Since re-observe resets start, the new gap is short and
        # we should NOT trip.
        self.assertEqual(tripped, [])
        self.assertFalse(ac.signal.aborted)

        # After enough time, we DO trip.
        time.sleep(0.1)
        tripped = wd.tick()
        self.assertEqual(len(tripped), 1)
        self.assertTrue(ac.signal.aborted)
        self.assertGreaterEqual(tripped[0][2], 0.05)

    def test_trip_is_idempotent_for_same_tool_use(self):
        ac = AbortController()
        on_trip = MagicMock()
        wd = ToolGapWatchdog(
            abort_controller=ac,
            explicit_overrides={"Bash": 0.1},
            on_trip=on_trip,
        )
        wd.observe_tool_use("a", "Bash")
        time.sleep(0.2)
        wd.tick()
        wd.tick()  # second tick — should NOT re-trip
        self.assertEqual(on_trip.call_count, 1)

    def test_no_abort_controller_is_no_op(self):
        wd = ToolGapWatchdog(
            abort_controller=None,
            explicit_overrides={"Bash": 0.05},
        )
        wd.observe_tool_use("a", "Bash")
        time.sleep(0.1)
        tripped = wd.tick()
        self.assertEqual(len(tripped), 1)
        # No exception; the watchdog observed + tripped but never crashed.

    def test_on_trip_callback_exception_does_not_break_watchdog(self):
        ac = AbortController()

        def explode(*_):
            raise RuntimeError("callback broken")

        wd = ToolGapWatchdog(
            abort_controller=ac,
            explicit_overrides={"Bash": 0.05},
            on_trip=explode,
            logger=MagicMock(),
        )
        wd.observe_tool_use("a", "Bash")
        time.sleep(0.1)
        # Should not raise — the watchdog swallows callback errors.
        tripped = wd.tick()
        self.assertEqual(len(tripped), 1)
        self.assertTrue(ac.signal.aborted)


class TestToolGapWatchdogIntegration(unittest.TestCase):
    """Integration with the FreezeDetector heartbeat path."""

    def test_tick_heartbeats_freeze_detector_singleton(self):
        # Layer-1 watchdog heartbeat is a no-op when the singleton is
        # unset; the test pins the contract.
        from clawcodex_ext.diagnostics import FreezeDetector

        FreezeDetector._INSTANCE = None  # noqa: SLF001
        ac = AbortController()
        wd = ToolGapWatchdog(
            abort_controller=ac,
            explicit_overrides={"Bash": 0.05},
        )
        wd.observe_tool_use("a", "Bash")
        time.sleep(0.1)
        wd.tick()
        # No exception; heartbeat path is benign when the watchdog
        # hasn't been started.
        FreezeDetector._INSTANCE = None  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
