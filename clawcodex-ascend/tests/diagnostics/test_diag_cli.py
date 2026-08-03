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

"""F-108 P108-H — ``clawcodex-dev diag`` CLI tests."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest

from clawcodex_ext.cli.diag_cmd import run_diag_command
from clawcodex_ext.cli.subcommand_registry import get_subcommand
from clawcodex_ext.diagnostics import (
    DEFAULT_FREEZE_DIAG_ENV,
    FreezeDetector,
    resolve_freeze_settings,
)


def _capture_stdio(callable_, *args, **kwargs):
    """Run ``callable_`` with stdout/stderr captured. Return (rc, out, err)."""
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        rc = callable_(*args, **kwargs)
        out = sys.stdout.getvalue()  # type: ignore[union-attr]
        err = sys.stderr.getvalue()  # type: ignore[union-attr]
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
    return rc, out, err


class TestHelp(unittest.TestCase):
    def test_help_prints_usage(self):
        rc, out, _ = _capture_stdio(run_diag_command, ["--help"])
        self.assertEqual(rc, 0)
        self.assertIn("usage:", out)
        self.assertIn("freeze-report", out)
        self.assertIn("status", out)

    def test_unknown_subcommand_returns_2(self):
        rc, _, err = _capture_stdio(run_diag_command, ["wat"])
        self.assertEqual(rc, 2)
        self.assertIn("unknown diag subcommand", err)


class TestStatus(unittest.TestCase):
    def setUp(self):
        os.environ.pop(DEFAULT_FREEZE_DIAG_ENV, None)
        FreezeDetector._INSTANCE = None  # noqa: SLF001

    def tearDown(self):
        os.environ.pop(DEFAULT_FREEZE_DIAG_ENV, None)
        FreezeDetector._INSTANCE = None  # noqa: SLF001

    def test_text_status_includes_settings(self):
        rc, out, _ = _capture_stdio(run_diag_command, ["status"])
        self.assertEqual(rc, 0)
        s = resolve_freeze_settings()
        for key, value in s.as_dict().items():
            self.assertIn(str(value), out)
            self.assertIn(key, out)

    def test_json_status_pure(self):
        rc, out, _ = _capture_stdio(run_diag_command, ["status", "--json"])
        self.assertEqual(rc, 0)
        # Output must be valid JSON.
        payload = json.loads(out)
        self.assertIn("diag_env_var", payload)
        self.assertEqual(payload["diag_env_var"], DEFAULT_FREEZE_DIAG_ENV)
        self.assertIn("settings", payload)
        self.assertEqual(
            payload["settings"]["permission_timeout_s"],
            30.0,
        )

    def test_diag_env_enabled_flips_status(self):
        os.environ[DEFAULT_FREEZE_DIAG_ENV] = "1"
        rc, out, _ = _capture_stdio(run_diag_command, ["status"])
        self.assertEqual(rc, 0)
        self.assertIn("diag env var (CLAWCODEX_FREEZE_DIAG): True", out)


class TestFreezeReport(unittest.TestCase):
    def setUp(self):
        os.environ.pop(DEFAULT_FREEZE_DIAG_ENV, None)
        FreezeDetector._INSTANCE = None  # noqa: SLF001
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        os.environ.pop(DEFAULT_FREEZE_DIAG_ENV, None)
        FreezeDetector._INSTANCE = None  # noqa: SLF001

    def test_empty_dump_dir_returns_zero(self):
        rc, _, err = _capture_stdio(run_diag_command, ["freeze-report", "--dump-dir", self.tmp.name])
        self.assertEqual(rc, 0)
        self.assertIn("no freeze dumps", err)

    def test_text_report_lists_dump(self):
        # Generate a real dump first.
        det = FreezeDetector(
            threshold=0.2,
            check_interval=0.05,
            dump_dir=self.tmp.name,
        )
        time_module = sys.modules["time"]
        time_module.sleep(0.3)
        det.check()
        rc, out, _ = _capture_stdio(run_diag_command, ["freeze-report", "--dump-dir", self.tmp.name])
        self.assertEqual(rc, 0)
        self.assertIn("Freeze dumps", out)
        self.assertIn("elapsed", out)
        self.assertIn("threads", out)

    def test_json_report_lists_dump(self):
        det = FreezeDetector(
            threshold=0.2,
            check_interval=0.05,
            dump_dir=self.tmp.name,
        )
        time_module = sys.modules["time"]
        time_module.sleep(0.3)
        det.check()
        rc, out, _ = _capture_stdio(
            run_diag_command,
            ["freeze-report", "--dump-dir", self.tmp.name, "--json"],
        )
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertIn("thread_stacks", payload[0])

    def test_viewer_alias(self):
        rc, out, err = _capture_stdio(run_diag_command, ["viewer", "--dump-dir", self.tmp.name])
        self.assertEqual(rc, 0)
        self.assertIn("no freeze dumps", err)


class TestSubcommandRegistration(unittest.TestCase):
    def test_diag_in_registry(self):
        handler = get_subcommand("diag")
        self.assertIsNotNone(handler)
        self.assertTrue(callable(handler))


if __name__ == "__main__":
    unittest.main()
