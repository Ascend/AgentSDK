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

"""Tests for @property handling in the pos-converter tool bridge."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from extensions.sop_converter.source_parser import SourceCodeParser, SourceOperation
from extensions.sop_converter.tool_registry_bridge import (
    _generate_method_stub,
    _generate_wrapper_script,
)


class TestPropertyDetection:
    def test_parse_property_methods(self) -> None:
        source = '''
class LoopCoordinator:
    """Coordinates the outer task-loop lifecycle."""

    @property
    def current_iteration(self) -> int:
        """Number of completed rounds."""
        return self._iteration

    @property
    def is_aborted(self) -> bool:
        """Whether abort has been requested."""
        return self._aborted

    def increment_iteration(self) -> None:
        """Record one completed round."""
        self._iteration += 1
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            py_file = tmp / "loop_coordinator.py"
            py_file.write_text(source)

            parser = SourceCodeParser(tmp)
            components = parser.parse()

        ops = {op.name: op for op in components[0].operations}
        assert ops["current_iteration"].is_property is True
        assert ops["is_aborted"].is_property is True
        assert ops["increment_iteration"].is_property is False


class TestPropertyStubGeneration:
    def test_class_property_stub_uses_attribute_access(self) -> None:
        op = SourceOperation(
            name="current_iteration",
            description="Number of completed rounds.",
            return_type="int",
            class_name="LoopCoordinator",
            is_property=True,
        )
        stub, _imports = _generate_method_stub(
            op,
            is_class_method=True,
            module_name="openjiuwen.harness.task_loop.loop_coordinator",
        )
        assert "def current_iteration() -> int:" in stub
        assert ".current_iteration" in stub
        assert ".current_iteration(" not in stub

    def test_class_method_stub_still_calls(self) -> None:
        op = SourceOperation(
            name="increment_iteration",
            description="Record one completed round.",
            class_name="LoopCoordinator",
        )
        stub, _imports = _generate_method_stub(
            op,
            is_class_method=True,
            module_name="openjiuwen.harness.task_loop.loop_coordinator",
        )
        assert ".increment_iteration(" in stub


class TestPropertyWrapperExecution:
    def test_wrapper_property_returns_value_not_callable_error(self) -> None:
        source = '''
class Counter:
    def __init__(self) -> None:
        self._value = 0

    @property
    def value(self) -> int:
        """Current counter value."""
        return self._value

    def bump(self) -> None:
        """Increment the counter."""
        self._value += 1
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / "demo_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            py_file = pkg / "counter.py"
            py_file.write_text(source)

            parser = SourceCodeParser(tmp)
            components = parser.parse()
            ops = components[0].operations

            script_path = _generate_wrapper_script(
                ops,
                class_name="Counter",
                module_name="demo_pkg.counter",
                file_stem="counter",
                source_dir=str(tmp),
            )

            for method, expected in (("value", 0), ("bump", None)):
                result = subprocess.run(
                    [sys.executable, str(script_path), method, "{}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert result.returncode == 0, result.stderr
                if expected is None:
                    assert result.stdout.strip() == "null"
                else:
                    assert json.loads(result.stdout.strip()) == expected

            result = subprocess.run(
                [sys.executable, str(script_path), "value", "{}"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout.strip()) == 0
