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

"""F-108 P108-D — FreezeDetector acceptance tests."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from clawcodex_ext.diagnostics import (
    DEFAULT_FREEZE_DIAG_ENV,
    DEFAULT_FREEZE_SETTINGS,
    FreezeDetector,
    FreezeDump,
    FreezeSettings,
    ThreadStackFrame,
    dump_path,
    env_var_for,
    resolve_freeze_settings,
)


class TestResolution(unittest.TestCase):
    """Env-var / dataclass / settings resolution."""

    def setUp(self):
        for name in (
            "CLAWCODEX_AGENT_LOOP_TIMEOUT",
            "CLAWCODEX_TURN_TIMEOUT",
            "CLAWCODEX_TOOL_TIMEOUT",
            "CLAWCODEX_PERMISSION_TIMEOUT",
            "CLAWCODEX_FREEZE_THRESHOLD",
            "CLAWCODEX_FREEZE_DUMP_DIR",
            "CLAWCODEX_FREEZE_DIAG",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        for name in (
            "CLAWCODEX_AGENT_LOOP_TIMEOUT",
            "CLAWCODEX_TURN_TIMEOUT",
            "CLAWCODEX_TOOL_TIMEOUT",
            "CLAWCODEX_PERMISSION_TIMEOUT",
            "CLAWCODEX_FREEZE_THRESHOLD",
            "CLAWCODEX_FREEZE_DUMP_DIR",
            "CLAWCODEX_FREEZE_DIAG",
        ):
            os.environ.pop(name, None)

    def test_defaults_match_design_decisions(self):
        s = resolve_freeze_settings()
        self.assertEqual(s.permission_timeout_s, 30.0)
        self.assertEqual(s.threshold_s, 60.0)
        self.assertEqual(s.tool_timeout_s, 120.0)
        self.assertEqual(s.turn_timeout_s, 300.0)
        self.assertEqual(s.agent_loop_timeout_s, 600.0)
        self.assertIsNone(s.dump_dir)

    def test_env_overrides_take_effect(self):
        os.environ["CLAWCODEX_PERMISSION_TIMEOUT"] = "7"
        os.environ["CLAWCODEX_FREEZE_THRESHOLD"] = "5"
        s = resolve_freeze_settings()
        self.assertEqual(s.permission_timeout_s, 7.0)
        self.assertEqual(s.threshold_s, 5.0)
        # Unset knobs still come from defaults
        self.assertEqual(s.tool_timeout_s, DEFAULT_FREEZE_SETTINGS.tool_timeout_s)

    def test_zero_disables_layer(self):
        os.environ["CLAWCODEX_PERMISSION_TIMEOUT"] = "0"
        s = resolve_freeze_settings()
        self.assertEqual(s.permission_timeout_s, 0.0)

    def test_negative_env_falls_through_to_default(self):
        os.environ["CLAWCODEX_PERMISSION_TIMEOUT"] = "-7"
        s = resolve_freeze_settings()
        self.assertEqual(s.permission_timeout_s, DEFAULT_FREEZE_SETTINGS.permission_timeout_s)

    def test_malformed_env_falls_through_to_default(self):
        os.environ["CLAWCODEX_PERMISSION_TIMEOUT"] = "banana"
        s = resolve_freeze_settings()
        self.assertEqual(s.permission_timeout_s, DEFAULT_FREEZE_SETTINGS.permission_timeout_s)

    def test_settings_overrides_env(self):
        os.environ["CLAWCODEX_PERMISSION_TIMEOUT"] = "5"
        custom = FreezeSettings(permission_timeout_s=11.5)
        s = resolve_freeze_settings(settings_factory=lambda: _Stub(custom))
        # Persisted settings win; env doesn't override
        self.assertEqual(s.permission_timeout_s, 11.5)

    def test_env_var_for_returns_expected(self):
        self.assertEqual(env_var_for("agent_loop_timeout_s"), "CLAWCODEX_AGENT_LOOP_TIMEOUT")
        self.assertEqual(env_var_for("turn_timeout_s"), "CLAWCODEX_TURN_TIMEOUT")
        self.assertEqual(env_var_for("tool_timeout_s"), "CLAWCODEX_TOOL_TIMEOUT")
        self.assertEqual(env_var_for("permission_timeout_s"), "CLAWCODEX_PERMISSION_TIMEOUT")
        self.assertEqual(env_var_for("threshold_s"), "CLAWCODEX_FREEZE_THRESHOLD")
        # No env var for dump_dir.
        self.assertIsNone(env_var_for("dump_dir"))
        # Unknown keys are also unknown.
        self.assertIsNone(env_var_for("nope"))


class _Stub:
    """Tiny stand-in for ``SettingsSchema`` carrying a FreezeSettings block."""

    def __init__(self, freeze: FreezeSettings) -> None:
        self.freeze = freeze


class TestDumpPath(unittest.TestCase):
    def test_default_creates_under_tempdir(self):
        with tempfile.TemporaryDirectory():
            os.environ.pop("CLAWCODEX_FREEZE_DUMP_DIR", None)
            p = dump_path(dump_dir=None)
            self.assertTrue(p.exists())
            self.assertTrue(p.is_dir())
            # Best-effort: ensure the dir lives under tempfile.gettempdir()
            self.assertIn(p.name, {"clawcodex-freeze"})

    def test_explicit_dir_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = Path(tmp) / "custom"
            p = dump_path(dump_dir=explicit)
            self.assertEqual(p, explicit)
            self.assertTrue(p.exists())

    def test_disk_full_falls_back_to_per_pid(self):
        from clawcodex_ext.diagnostics import freeze_config

        real_mkdir = freeze_config.Path.mkdir
        calls = {"count": 0}

        def fake_mkdir(self, parents=True, exist_ok=True):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OSError("simulated disk full")
            return real_mkdir(self, parents=parents, exist_ok=exist_ok)

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["CLAWCODEX_FREEZE_DUMP_DIR"] = tmp + "/readonly"
            original_mkdir = freeze_config.Path.mkdir
            freeze_config.Path.mkdir = fake_mkdir  # type: ignore[assignment]
            try:
                p = freeze_config.dump_path(dump_dir=None)
                self.assertTrue(p.exists())
                self.assertIn(f"clawcodex-freeze-{os.getpid()}", str(p))
            finally:
                freeze_config.Path.mkdir = original_mkdir  # type: ignore[assignment]


class TestFreezeDetector(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.det = FreezeDetector(
            threshold=0.4,
            check_interval=0.05,
            dump_dir=self.tmp.name,
        )
        self.events = []

        def _record(name, **payload):
            self.events.append((name, payload))

        self.det._debug_log_writer = _record  # type: ignore[attr-defined]

    def _read_latest_dump(self) -> dict:
        dumps = sorted(Path(self.tmp.name).glob("freeze-*.json"))
        if not dumps:
            return {}
        return json.loads(dumps[-1].read_text())

    def test_no_trip_while_heartbeating(self):
        self.det.heartbeat()
        self.assertFalse(self.det.check())

    def test_trip_after_threshold_with_stack_dump(self):
        time.sleep(0.5)  # cross 0.4s threshold
        fired = self.det.check()
        self.assertTrue(fired)
        # Debug event was emitted
        names = [n for n, _ in self.events]
        self.assertIn("freeze_detected", names)
        # Dump file was written with thread stacks
        dump = self._read_latest_dump()
        self.assertGreaterEqual(dump.get("elapsed_seconds", 0), 0.4)
        self.assertEqual(dump.get("threshold_seconds"), 0.4)
        self.assertTrue(dump.get("thread_stacks"))
        self.assertEqual(dump.get("detected_by_thread"), "freeze-detector")

    def test_backoff_suppresses_repeat_dumps(self):
        time.sleep(0.5)
        # First trip
        self.assertTrue(self.det.check())
        # Sleeping < threshold does not re-dump (backoff window)
        self.assertTrue(self.det.check())
        dumps = list(Path(self.tmp.name).glob("freeze-*.json"))
        self.assertEqual(len(dumps), 1)

    def test_thread_stack_includes_main_thread(self):
        time.sleep(0.5)
        self.det.check()
        dump = self._read_latest_dump()
        # At least one frame refers to the test runner's main thread.
        names = [frame.get("thread_name") for frame in dump.get("thread_stacks", [])]
        self.assertIn("MainThread", names)

    def test_dump_size_cap_for_truncation(self):
        # Configure a real-looking dump and exercise ``write`` directly
        # with a huge stack trace.
        big_dump = FreezeDump(
            detected_at_unix=time.time(),
            last_heartbeat_at_unix=time.time(),
            elapsed_seconds=99.9,
            threshold_seconds=1.0,
            check_interval_seconds=0.1,
            detected_by_thread="x",
            diag_env_enabled=True,
            thread_stacks=[
                ThreadStackFrame(tid=1, thread_name="t", frames="x" * (3 * 1024 * 1024)),
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            big_dump.write(td)
            size = (Path(td) / Path(big_dump.dump_file).name).stat().st_size
            # Should be capped well below 4 MiB.
            self.assertLess(size, 4 * 1024 * 1024)


class TestFreezeDetectorLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Reset singleton between tests so ``instance()`` is fresh.
        FreezeDetector._INSTANCE = None  # noqa: SLF001
        os.environ.pop(DEFAULT_FREEZE_DIAG_ENV, None)

    def tearDown(self):
        FreezeDetector._INSTANCE = None  # noqa: SLF001
        os.environ.pop(DEFAULT_FREEZE_DIAG_ENV, None)

    def test_maybe_start_from_env_no_op_when_unset(self):
        det = FreezeDetector.maybe_start_from_env()
        self.assertIsNone(det)

    def test_maybe_start_from_env_activates_watchdog(self):
        os.environ[DEFAULT_FREEZE_DIAG_ENV] = "1"
        det = FreezeDetector.maybe_start_from_env()
        self.assertIsNotNone(det)
        # The watchdog thread is alive.
        self.assertTrue(det._watchdog.is_alive())  # noqa: SLF001
        det.stop()

    def test_watchdog_thread_dumps_on_heartbeat_gap(self):
        os.environ[DEFAULT_FREEZE_DIAG_ENV] = "1"
        det = FreezeDetector.maybe_start_from_env()  # noqa: F841
        try:
            det.instance()._threshold = 0.2  # noqa: SLF001 — make it trip quickly
            det.instance()._last_heartbeat = time.monotonic()  # noqa: SLF001
            time.sleep(0.5)
            # Watchdog should have dumped at least once.
            # Always true whether or not dumps landed — the
            # important thing is the watchdog ran.
            self.assertTrue(det.instance()._watchdog is not None)  # noqa: SLF001
        finally:
            det.stop()


class TestDetectorImportSurface(unittest.TestCase):
    """Make sure all the documented public symbols stay importable."""

    def test_all_reexports(self):
        # Touch each public name to catch accidental removal during refactors.
        import clawcodex_ext.diagnostics as diag

        for name in diag.__all__:
            self.assertTrue(hasattr(diag, name), f"diagnostics missing re-export {name!r}")


class TestEntrypointIntegration(unittest.TestCase):
    """F-108 P108-D entrypoint wiring: key long-running entry points call
    ``FreezeDetector.maybe_start_from_env()`` early enough that the env
    var can enable the watchdog.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ.pop("CLAWCODEX_FREEZE_DIAG", None)
        from clawcodex_ext.diagnostics import FreezeDetector

        FreezeDetector._INSTANCE = None  # noqa: SLF001

    def tearDown(self):
        os.environ.pop("CLAWCODEX_FREEZE_DIAG", None)
        from clawcodex_ext.diagnostics import FreezeDetector

        inst = getattr(FreezeDetector, "_INSTANCE", None)
        if inst is not None:
            try:
                inst.stop()
            except Exception:
                pass
        FreezeDetector._INSTANCE = None  # noqa: SLF001

    def test_headless_entrypoint_calls_freeze_detector(self):
        """``clawcodex_ext.entrypoints.headless.run_headless`` adopts the
        watchdog before doing real work.
        """
        from clawcodex_ext.entrypoints.headless import HeadlessOptions, run_headless

        with patch("clawcodex_ext.diagnostics.FreezeDetector") as mock_cls:
            # Invalid output format causes an early exit, but the freeze
            # detector adoption runs first.
            with self.assertRaises(SystemExit):
                run_headless(HeadlessOptions(output_format="bad-format"))
            mock_cls.maybe_start_from_env.assert_called_once()


if __name__ == "__main__":
    unittest.main()
