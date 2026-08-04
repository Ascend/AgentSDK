#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for DEV-5: bundled-skill catalogue + init orchestrator.

Acceptance criteria:
1. After init, get_bundled_skills returns ≥ 5 named skills.
2. SkillTool /simplify returns prompt containing "Phase 1: Identify Changes".
3. SkillTool /loop "5m hello" → fixed-prompt branch with 5m + hello body.
4. SkillTool /loop "" → dynamic-rescheduling branch.
5. SkillTool /debug returns a prompt that contains the debug log path
   and does not throw on a missing log file.
6. init_bundled_skills is idempotent (no double-registration).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.skills.bundled import (
    init_bundled_skills,
    register_simplify_skill,
    reset_bundled_skills_init_flag,
)
from src.skills.bundled.loop import (
    ParsedLoopArgs,
    parse_loop_args,
)
from src.skills.bundled_skills import (
    clear_bundled_skills,
    get_bundled_skills,
)
from src.skills.loader import (
    clear_dynamic_skills,
    clear_skill_caches,
    clear_skill_registry,
)
from src.tool_system.context import ToolContext
from src.tool_system.tools import SkillTool
from src.tool_system.tools.skill import run_user_invoked_skill

from ._assertions import rendered_prompt


def _reset_all() -> None:
    clear_bundled_skills()
    reset_bundled_skills_init_flag()
    clear_skill_caches()
    clear_dynamic_skills()
    clear_skill_registry()


class TestInitOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all()

    def tearDown(self) -> None:
        _reset_all()

    def test_init_seeds_at_least_five(self) -> None:
        # AC#1
        init_bundled_skills()
        names = {s.name for s in get_bundled_skills()}
        self.assertGreaterEqual(len(names), 5)
        for required in (
            "simplify",
            "debug",
            "loop",
            "orchestrator",
            "spec-audit",
            "stuck",
            "verify-content",
        ):
            self.assertIn(required, names)

    def test_init_idempotent(self) -> None:
        # AC#6
        init_bundled_skills()
        first = len(get_bundled_skills())
        init_bundled_skills()
        init_bundled_skills()
        self.assertEqual(len(get_bundled_skills()), first)


class TestSimplifySkill(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        _reset_all()
        self._tmp.cleanup()

    def test_simplify_returns_phase1_marker(self) -> None:
        # AC#2
        ctx = ToolContext(workspace_root=self.root)
        result = SkillTool.call({"skill": "simplify"}, ctx)
        out = result.output
        self.assertTrue(out["success"])
        self.assertIn("Phase 1: Identify Changes", rendered_prompt(result))

    def test_simplify_appends_args(self) -> None:
        ctx = ToolContext(workspace_root=self.root)
        result = SkillTool.call({"skill": "simplify", "args": "look at /auth"}, ctx)
        prompt = rendered_prompt(result)
        self.assertIn("Additional Focus", prompt)
        self.assertIn("look at /auth", prompt)


class TestLoopSkillParser(unittest.TestCase):
    def test_empty_args_dynamic_maintenance(self) -> None:
        self.assertEqual(parse_loop_args(""), ParsedLoopArgs(mode="dynamic-maintenance"))

    def test_bare_interval_fixed_maintenance(self) -> None:
        out = parse_loop_args("5m")
        self.assertEqual(out.mode, "fixed-maintenance")
        self.assertEqual(out.interval, "5m")

    def test_leading_interval_with_prompt(self) -> None:
        out = parse_loop_args("5m hello world")
        self.assertEqual(out.mode, "fixed-prompt")
        self.assertEqual(out.interval, "5m")
        self.assertEqual(out.prompt, "hello world")

    def test_trailing_every_clause(self) -> None:
        out = parse_loop_args("ping the build every 10 minutes")
        self.assertEqual(out.mode, "fixed-prompt")
        self.assertEqual(out.interval, "10m")
        self.assertEqual(out.prompt, "ping the build")

    def test_unit_aliases(self) -> None:
        # Variants accepted by normalize_interval_unit.
        for token, expect in (
            ("3secs", "3s"),
            ("2 hours", "2h"),
            ("1day", "1d"),
            ("10 minute", "10m"),
        ):
            out = parse_loop_args(token)
            self.assertEqual(
                out,
                ParsedLoopArgs(mode="fixed-maintenance", interval=expect),
                f"failed for token={token!r}",
            )

    def test_plain_prompt_defaults_to_fixed_10m(self) -> None:
        out = parse_loop_args("just some prose with no interval")
        self.assertEqual(out.mode, "fixed-prompt")
        self.assertEqual(out.interval, "10m")
        self.assertEqual(out.prompt, "just some prose with no interval")


class TestLoopSkillEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        _reset_all()
        self._tmp.cleanup()

    def test_loop_5m_hello_fixed_prompt_branch(self) -> None:
        # AC#3
        ctx = ToolContext(workspace_root=self.root)
        result = SkillTool.call({"skill": "loop", "args": "5m hello"}, ctx)
        out = result.output
        self.assertTrue(out["success"])
        prompt = rendered_prompt(result)
        self.assertIn("fixed recurring interval", prompt)
        self.assertIn("Requested interval: 5m", prompt)
        self.assertIn("hello", prompt)
        self.assertIn("--- BEGIN PROMPT ---", prompt)
        self.assertIn("CronCreate", prompt)

    def test_loop_empty_args_dynamic_branch(self) -> None:
        # AC#4
        ctx = ToolContext(workspace_root=self.root)
        result = SkillTool.call({"skill": "loop"}, ctx)
        out = result.output
        self.assertTrue(out["success"])
        prompt = rendered_prompt(result)
        self.assertIn("dynamic rescheduling", prompt)
        self.assertIn("--- BEGIN MAINTENANCE PROMPT ---", prompt)

    def test_loop_plain_prompt_defaults_to_10m_fixed_branch(self) -> None:
        ctx = ToolContext(workspace_root=self.root)
        result = SkillTool.call({"skill": "loop", "args": "check the deploy"}, ctx)
        out = result.output
        self.assertTrue(out["success"])
        prompt = rendered_prompt(result)
        self.assertIn("fixed recurring interval", prompt)
        self.assertIn("Requested interval: 10m", prompt)
        self.assertIn("check the deploy", prompt)


class TestDebugSkill(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        _reset_all()
        self._tmp.cleanup()

    def test_debug_prompt_includes_log_path_no_log_file(self) -> None:
        # AC#5: surfaces log path; no exception on missing file.
        nonexistent = self.root / "no-such-debug.log"
        with patch.dict(
            os.environ,
            {"CLAUDE_CODE_DEBUG_LOG_PATH": str(nonexistent)},
        ):
            ctx = ToolContext(workspace_root=self.root)
            result = run_user_invoked_skill("debug", "", ctx)
        out = result.output
        prompt = rendered_prompt(result)
        self.assertTrue(out["success"])
        self.assertIn(str(nonexistent), prompt)
        # The "no debug log exists yet" hint must be in the body.
        self.assertIn("No debug log exists at the resolved path", prompt)

    def test_debug_prompt_with_existing_log(self) -> None:
        log = self.root / "debug.log"
        log.write_text("[INFO] hi\n[ERROR] crash\n")
        with patch.dict(os.environ, {"CLAUDE_CODE_DEBUG_LOG_PATH": str(log)}):
            ctx = ToolContext(workspace_root=self.root)
            result = run_user_invoked_skill("debug", "", ctx)
        out = result.output
        prompt = rendered_prompt(result)
        self.assertTrue(out["success"])
        self.assertIn(str(log), prompt)
        self.assertIn("[ERROR] crash", prompt)

    def test_debug_prompt_includes_user_issue(self) -> None:
        ctx = ToolContext(workspace_root=self.root)
        result = run_user_invoked_skill("debug", "auth keeps failing", ctx)
        self.assertIn("auth keeps failing", rendered_prompt(result))


class TestStuckAndVerifySkills(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        _reset_all()
        self._tmp.cleanup()

    def test_stuck_renders(self) -> None:
        ctx = ToolContext(workspace_root=self.root)
        result = SkillTool.call({"skill": "stuck"}, ctx)
        out = result.output
        self.assertTrue(out["success"])
        self.assertIn("Reset and Re-Approach", rendered_prompt(result))

    def test_stuck_with_args_appends_user_context(self) -> None:
        ctx = ToolContext(workspace_root=self.root)
        result = SkillTool.call({"skill": "stuck", "args": "tests fail"}, ctx)
        prompt = rendered_prompt(result)
        self.assertIn("tests fail", prompt)
        self.assertIn("User Context", prompt)

    def test_verify_content_renders(self) -> None:
        ctx = ToolContext(workspace_root=self.root)
        result = SkillTool.call({"skill": "verify-content"}, ctx)
        out = result.output
        self.assertTrue(out["success"])
        self.assertIn("Verify Recent Edits Match Intent", rendered_prompt(result))


class TestRegisterFunctionsIdempotent(unittest.TestCase):
    """Each register_*_skill is allowed to be called explicitly; doing
    so should not double-register if init has already run.
    """

    def setUp(self) -> None:
        _reset_all()

    def tearDown(self) -> None:
        _reset_all()

    def test_individual_register_does_not_double(self) -> None:
        # Register-by-hand path (used by tests + the init orchestrator).
        register_simplify_skill()
        first_count = len(get_bundled_skills())
        register_simplify_skill()
        self.assertEqual(len(get_bundled_skills()), first_count)


if __name__ == "__main__":
    unittest.main()
