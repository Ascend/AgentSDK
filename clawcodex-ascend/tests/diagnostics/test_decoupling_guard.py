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

"""Acceptance guard: no implementation files may live under ``src/``.

This test pins the decoupling invariant by *structural*
inspection: every diagnostics module must live under
``clawcodex_ext/``, ``extensions/``, or ``tests/`` — never under
``src/``. Run via:

    python -m pytest tests/diagnostics/test_decoupling_guard.py

It does NOT diff the working tree against HEAD because other
other features and refactor batches modify ``src/`` on this
branch. The structural check is what actually enforces the
constraint: ``import clawcodex_ext.diagnostics.freeze_detector``
must work, and the source path of every diagnostics module must resolve
to a non-``src/`` directory.

It also exercises the public surface used by extensions at a
*behavioural* level so a regression that wires up the watchdog into
the canonical loop is caught even if the imports are clean.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.utils.abort_controller import AbortController

REPO_ROOT = Path(__file__).resolve().parents[2]


# Implementation modules. Each entry is the dotted import
# path used to look up the module's source file. Adding a new diagnostics
# module = adding an entry here.
F108_MODULES = (
    "clawcodex_ext.diagnostics",
    "clawcodex_ext.diagnostics.freeze_config",
    "clawcodex_ext.diagnostics.freeze_detector",
    "clawcodex_ext.diagnostics.recovery",
    "clawcodex_ext.cli.diag_cmd",
    "clawcodex_ext.cli.subcommand_registry",
    "clawcodex_ext.tool_system.tool_timeout",
)


class TestDecouplingGuard(unittest.TestCase):
    """Structural guard against diagnostics modules under ``src/``."""

    def _resolve(self, dotted: str) -> Path:
        """Return the on-disk path of an importable module."""
        import importlib

        mod = importlib.import_module(dotted)
        path = Path(getattr(mod, "__file__", "") or "")
        self.assertTrue(
            path.exists() and path.is_file(),
            f"{dotted} did not resolve to a real file (got {path!r})",
        )
        return path

    def test_no_f108_module_lives_under_src(self):
        offenders: list[tuple[str, str]] = []
        for dotted in F108_MODULES:
            try:
                path = self._resolve(dotted)
            except AssertionError:
                # Module missing or path unresolved — handled by
                # the test_no_f108_module_unimportable_* cases
                # elsewhere. Skip here.
                continue
            try:
                rel = path.resolve().relative_to(REPO_ROOT.resolve())
            except ValueError:
                # Outside the repo — treat as out-of-scope.
                continue
            parts = rel.parts
            if parts and parts[0] == "src":
                offenders.append((dotted, str(rel)))
        self.assertEqual(
            offenders,
            [],
            "F-108 §十八 acceptance #7 forbids src/ implementations; "
            "offending modules: " + ", ".join(f"{m}->{p}" for m, p in offenders),
        )

    def test_f108_extensions_modules_importable(self):
        """Verify that every diagnostics module imports cleanly."""
        for dotted in F108_MODULES:
            with self.subTest(module=dotted):
                try:
                    __import__(dotted)
                except Exception as exc:  # pragma: no cover - diagnostic
                    self.fail(f"{dotted} import failed: {exc!r}")


class TestExtensionsAbortionPath(unittest.TestCase):
    """Behavioural smoke for the abort -> cancellation bridge."""

    def test_abort_controller_engages_via_timeout(self):
        """The watchdog hands off to ``AbortController`` — verify that."""
        from clawcodex_ext.tool_system.tool_timeout import ToolGapWatchdog

        ac = AbortController()
        wd = ToolGapWatchdog(
            abort_controller=ac,
            explicit_overrides={"Bash": 0.05},
        )
        wd.observe_tool_use("a", "Bash")
        import time

        time.sleep(0.1)
        tripped = wd.tick()
        self.assertEqual(len(tripped), 1)
        self.assertTrue(ac.signal.aborted)

    def test_tool_context_unchanged(self):
        """The diagnostics layer does not mutate ToolContext defaults.

        We rely on the existing ``abort_controller`` field. A
        regression that adds a new required field would break
        downstream callers that build bare ToolContext()s.
        """
        # Build a minimal context without overriding defaults; the
        # factory should populate ``abort_controller`` to a fresh
        # AbortController that is NOT aborted.
        ctx = ToolContext(workspace_root=os.path.join(os.sep, "tmp"))
        self.assertIsNotNone(ctx.abort_controller)
        self.assertFalse(ctx.abort_controller.signal.aborted)


if __name__ == "__main__":
    unittest.main()
