#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Interactive-terminal heuristic must use whole-word docstring tokens."""

from __future__ import annotations

import unittest

from extensions.sop_converter.core.source_parser import SourceComponent, SourceOperation
from extensions.sop_converter.core.sop_prompts import SOP_INTERACTIVE_TERMINAL_STOP_LOSS
from extensions.sop_converter.runtime.task_guide import _looks_like_interactive_terminal


class TestInteractiveTerminalHeuristic(unittest.TestCase):
    def _app_comp(self) -> SourceComponent:
        return SourceComponent(
            name="chat_with_ascend",
            file_path="Samples/chat_with_ascend/app.py",
            description="",
        )

    def test_http_client_substring_is_not_cli(self) -> None:
        op = SourceOperation(
            name="parse_pdf_file",
            description=(
                "backend: vlm-http-client: Faster(client). without method specified, pipeline will be used by default."
            ),
            file_stem="app",
            has_docstring=True,
        )
        self.assertFalse(_looks_like_interactive_terminal(self._app_comp(), op))

    def test_replace_substring_is_not_repl(self) -> None:
        op = SourceOperation(
            name="replace_image_paths",
            description="replace imageN with image_docs paths in the response",
            file_stem="app",
            has_docstring=True,
        )
        self.assertFalse(_looks_like_interactive_terminal(self._app_comp(), op))

    def test_standalone_cli_token_still_counts(self) -> None:
        op = SourceOperation(
            name="parse_pdf_file",
            description="Interactive CLI for parsing PDF files.",
            file_stem="app",
            has_docstring=True,
        )
        self.assertTrue(_looks_like_interactive_terminal(self._app_comp(), op))

    def test_function_named_cli_still_counts(self) -> None:
        op = SourceOperation(
            name="run_app_cli",
            description="Start the application.",
            file_stem="app",
            has_docstring=True,
        )
        self.assertTrue(_looks_like_interactive_terminal(self._app_comp(), op))


class TestInteractiveStopLossPrompt(unittest.TestCase):
    def test_forbids_fabricated_cli_commands(self) -> None:
        text = SOP_INTERACTIVE_TERMINAL_STOP_LOSS
        self.assertIn("禁止编造", text)
        self.assertIn("call_impl", text)
        self.assertIn("--flag", text)
        self.assertIn("没有终端入口命令", text)
        self.assertIn("<某模块的入口命令>", text)


if __name__ == "__main__":
    unittest.main()
