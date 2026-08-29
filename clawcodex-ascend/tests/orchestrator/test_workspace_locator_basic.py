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

"""Basic tests for :mod:`extensions.orchestrator.workspace_locator`.

Extracted from ``test_orchestrator_workspace_locator.py`` to ship with
the source-only PR. Full test suite migrates in a follow-up PR.

Covers:
* :func:`_slug_from_workspace` path-slug generation
* :func:`get_workspace_root` priority chain basics (none-match, arg priority)
"""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from extensions.orchestrator import workspace_locator as wsl
from extensions.orchestrator.workspace_locator import (
    _slug_from_workspace,
    get_workspace_root,
)


def _isolated_home() -> tuple[ExitStack, Path]:
    """Return (exit_stack, temp_home) that redirects all module writes."""
    stack = ExitStack()
    tmp = tempfile.TemporaryDirectory()
    stack.callback(tmp.cleanup)
    home = Path(tmp.name) / "home"
    home.mkdir()
    fake_base = home / ".clawcodex"
    fake_orch = fake_base / "orchestrator"
    fake_orch.mkdir(parents=True, exist_ok=True)
    stack.enter_context(patch.object(wsl, "CLAWCODEX_BASE", fake_base))
    stack.enter_context(patch.object(wsl, "ORCHESTRATOR_DIR", fake_orch))
    return stack, home


class TestSlugFromWorkspace(unittest.TestCase):
    def test_absolute_path_slug(self) -> None:
        slug = _slug_from_workspace("/tmp/symphony_workspaces/proj-foo")
        self.assertIsInstance(slug, str)
        self.assertNotIn("/", slug)
        self.assertNotIn("\\", slug)
        self.assertNotEqual(slug, "")

    def test_relative_path_slug(self) -> None:
        slug = _slug_from_workspace("home/user/code")
        self.assertIsInstance(slug, str)
        self.assertNotEqual(slug, "")

    def test_strips_tmp_segments(self) -> None:
        slug = _slug_from_workspace("/tmp/proj-foo")
        self.assertNotIn("tmp", slug.split("-"))

    def test_empty_falls_back_to_default(self) -> None:
        slug = _slug_from_workspace("///")
        self.assertEqual(slug, "default")


class TestGetWorkspaceRootBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.stack, self.home = _isolated_home()
        self.addCleanup(self.stack.close)
        self.cwd_fake = self.home / "cwd"
        self.cwd_fake.mkdir()

    def test_returns_none_when_nothing_matches(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLAWCODEX_WORKSPACE_ROOT", None)
            with patch("os.getcwd", return_value=str(self.home / "no-cwd")):
                result = get_workspace_root()
        self.assertIsNone(result)

    def test_workspace_arg_highest_priority(self) -> None:
        explicit = self.home / "from-arg"
        explicit.mkdir()
        with patch.dict(
            os.environ,
            {"CLAWCODEX_WORKSPACE_ROOT": "/from/env"},
        ):
            result = get_workspace_root(workspace_arg=str(explicit))
        self.assertEqual(result, Path(str(explicit)).resolve())


if __name__ == "__main__":
    unittest.main()
