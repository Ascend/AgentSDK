#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------
"""Tests for class __init__ parameter extraction in SourceCodeParser."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extensions.sop_converter.source_parser import SourceCodeParser


class TestClassInitParamsExtraction(unittest.TestCase):
    def test_extracts_init_params_without_exporting_init_as_tool(self) -> None:
        source = '''
class SharedMemoryManager:
    """Team shared memory."""

    def __init__(self, team_memory_dir: str) -> None:
        self.team_memory_dir = team_memory_dir

    def ensure_dir(self) -> None:
        """Ensure team-memory directory exists."""
        pass
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            py_file = tmp / "memory.py"
            py_file.write_text(source)

            parser = SourceCodeParser(tmp, extern_only=False)
            components = parser.parse()

        self.assertEqual(len(components), 1)
        comp = components[0]
        self.assertIn("SharedMemoryManager", comp.class_init_params)
        init_names = [p.name for p in comp.class_init_params["SharedMemoryManager"]]
        self.assertEqual(init_names, ["team_memory_dir"])
        op_names = [op.name for op in comp.operations]
        self.assertIn("ensure_dir", op_names)
        self.assertNotIn("__init__", op_names)


class TestAsyncGeneratorDetection(unittest.TestCase):
    def test_detects_async_iterator_return_annotation(self) -> None:
        source = '''
from typing import AsyncIterator, Any

class Runner:
    async def run_agent_team_streaming(self, agent_team, inputs) -> AsyncIterator[Any]:
        """Stream-run a team."""
        yield agent_team
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "runner.py").write_text(source)
            components = SourceCodeParser(tmp, extern_only=False).parse()

        op = components[0].operations[0]
        self.assertTrue(op.is_async)
        self.assertTrue(op.is_async_generator)

    def test_plain_async_coroutine_is_not_async_generator(self) -> None:
        source = '''
class Runner:
    async def run_agent_team(self, agent_team, inputs):
        """Run a team."""
        return agent_team
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "runner.py").write_text(source)
            components = SourceCodeParser(tmp, extern_only=False).parse()

        op = components[0].operations[0]
        self.assertTrue(op.is_async)
        self.assertFalse(op.is_async_generator)


if __name__ == "__main__":
    unittest.main()
