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

from __future__ import annotations


# pylint: disable=E0611
import unittest

from clawcodex_ext.services.api.tool_normalization import (
    has_tool_field_mapping,
    normalize_tool_arguments,
)


class TestHasToolFieldMapping(unittest.TestCase):
    def test_bash_has_mapping(self) -> None:
        self.assertTrue(has_tool_field_mapping("Bash"))

    def test_read_has_mapping(self) -> None:
        self.assertTrue(has_tool_field_mapping("Read"))

    def test_unknown_no_mapping(self) -> None:
        self.assertFalse(has_tool_field_mapping("CustomTool"))


class TestNormalizeToolArguments(unittest.TestCase):
    def test_none_returns_empty_dict(self) -> None:
        self.assertEqual(normalize_tool_arguments("Bash", None), {})

    def test_valid_json_object(self) -> None:
        result = normalize_tool_arguments("Bash", '{"command": "ls"}')
        self.assertEqual(result, {"command": "ls"})

    def test_plain_string_for_bash(self) -> None:
        result = normalize_tool_arguments("Bash", "ls -la")
        self.assertEqual(result, {"command": "ls -la"})

    def test_plain_string_for_read(self) -> None:
        result = normalize_tool_arguments("Read", "/path/to/file")
        self.assertEqual(result, {"file_path": "/path/to/file"})

    def test_plain_string_for_unknown_tool(self) -> None:
        result = normalize_tool_arguments("Unknown", "some value")
        self.assertEqual(result, {})

    def test_json_string_value_wrapping(self) -> None:
        result = normalize_tool_arguments("Bash", '"ls -la"')
        self.assertEqual(result, {"command": "ls -la"})

    def test_blank_string(self) -> None:
        result = normalize_tool_arguments("Bash", "   ")
        self.assertEqual(result, {})

    def test_structured_object_literal(self) -> None:
        result = normalize_tool_arguments("Bash", "{ 'command': 'ls' }")
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
