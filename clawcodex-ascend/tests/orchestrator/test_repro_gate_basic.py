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

"""Tests for the repro-first gate (reproduce before fixing).

Covers build_repro_prompt, evaluate_repro_gate, append_repro_hint,
and format_repro_gate_comment.

Sections requiring Orchestrator / config.schema / git_sync (not yet
migrated) are deferred to a later PR.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from extensions.orchestrator.repro_gate import (
    NOT_REPRODUCIBLE_FILE,
    REPRO_COMMAND_FILE,
    ReproGateResult,
    append_repro_hint,
    build_repro_prompt,
    evaluate_repro_gate,
    format_repro_gate_comment,
)


class _Issue:
    def __init__(self, labels: list[str] | None = None) -> None:
        self.id = "19"
        self.identifier = "PROBE-19"
        self.title = "Crash in retry backoff"
        self.description = "timeout=0 causes ZeroDivisionError"
        self.labels = labels or []


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _exit_script(root: Path, code: int) -> str:
    _write(root, ".orchestrator_control/repro/check.py", f"import sys\nsys.exit({code})\n")
    return f'"{sys.executable}" .orchestrator_control/repro/check.py'


class TestBuildReproPrompt(unittest.TestCase):
    def test_prompt_states_the_contract(self) -> None:
        prompt = build_repro_prompt(_Issue())
        self.assertIn("PROBE-19", prompt)
        self.assertIn(REPRO_COMMAND_FILE, prompt)
        self.assertIn(NOT_REPRODUCIBLE_FILE, prompt)
        self.assertIn("NON-ZERO", prompt)
        self.assertIn("Do NOT fix anything", prompt)


class TestEvaluateReproGate(unittest.IsolatedAsyncioTestCase):
    async def test_no_artifacts_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = await evaluate_repro_gate(tmp)
            self.assertEqual(result.verdict, "missing")
            self.assertFalse(result.proceed)

    async def test_not_reproducible_report_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                NOT_REPRODUCIBLE_FILE,
                json.dumps({"reason": "file does not exist", "attempts": ["grep", "git log"]}),
            )
            result = await evaluate_repro_gate(tmp)
            self.assertEqual(result.verdict, "not_reproducible")
            assert result.payload is not None
            self.assertEqual(result.payload["reason"], "file does not exist")

    async def test_malformed_report_still_closes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write(Path(tmp), NOT_REPRODUCIBLE_FILE, "not json at all")
            result = await evaluate_repro_gate(tmp)
            self.assertEqual(result.verdict, "not_reproducible")

    async def test_failing_command_opens_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _exit_script(root, 1)
            _write(root, REPRO_COMMAND_FILE, command + "\n")
            result = await evaluate_repro_gate(tmp)
            self.assertEqual(result.verdict, "reproduced")
            self.assertTrue(result.proceed)
            self.assertEqual(result.command, command)

    async def test_green_command_is_not_a_demonstration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = _exit_script(root, 0)
            _write(root, REPRO_COMMAND_FILE, command + "\n")
            result = await evaluate_repro_gate(tmp)
            self.assertEqual(result.verdict, "not_demonstrated")
            self.assertFalse(result.proceed)

    async def test_comment_only_command_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write(Path(tmp), REPRO_COMMAND_FILE, "# TODO figure it out\n\n")
            result = await evaluate_repro_gate(tmp)
            self.assertEqual(result.verdict, "missing")


class TestReproHint(unittest.TestCase):
    def test_hint_created_and_appended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_repro_hint(root, "pytest tests/test_probe.py -q")
            hints = (root / ".operator_hints.md").read_text(encoding="utf-8")
            self.assertIn("Reproduction established", hints)
            self.assertIn("pytest tests/test_probe.py -q", hints)

            (root / ".operator_hints.md").write_text("existing operator note\n", encoding="utf-8")
            append_repro_hint(root, "make repro")
            hints = (root / ".operator_hints.md").read_text(encoding="utf-8")
            self.assertTrue(hints.startswith("existing operator note"))
            self.assertIn("make repro", hints)


class TestGateComment(unittest.TestCase):
    def test_not_reproducible_comment_lists_attempts(self) -> None:
        comment = format_repro_gate_comment(
            _Issue(),
            ReproGateResult(
                verdict="not_reproducible",
                payload={"reason": "no such file", "attempts": ["grep -r", "read docs"]},
            ),
        )
        self.assertIn("PROBE-19", comment)
        self.assertIn("no such file", comment)
        self.assertIn("- grep -r", comment)
        self.assertIn("No fix was attempted", comment)

    def test_not_demonstrated_comment_shows_command(self) -> None:
        comment = format_repro_gate_comment(
            _Issue(),
            ReproGateResult(verdict="not_demonstrated", command="pytest -q", output="all green"),
        )
        self.assertIn("exits 0", comment)
        self.assertIn("pytest -q", comment)


if __name__ == "__main__":
    unittest.main()
