#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Macro coverage contract validation tests."""

from __future__ import annotations

import unittest

from extensions.sop_converter.runtime.macros.errors import MacroConvertError
from extensions.sop_converter.runtime.macros.models import MacroDefinition, MacroRoute
from extensions.sop_converter.runtime.macros.validation import validate_macro_definition


def _macro(route: MacroRoute) -> MacroDefinition:
    return MacroDefinition(
        name="demo-macro",
        scope="bundle",
        workflow={
            "inputs": {},
            "steps": [
                {
                    "id": "invoke",
                    "kind": "tool",
                    "callable_ref": "demo-atomic",
                    "args": {},
                }
            ],
            "outputs": {},
        },
        routing=route,
    )


class TestMacroCoverageValidation(unittest.TestCase):
    def test_verified_exclusive_requires_intent_key(self) -> None:
        macro = _macro(
            MacroRoute(
                target_tool="demo-macro",
                selection="exclusive",
                verified=True,
                covered_tools=["demo-atomic"],
            )
        )
        with self.assertRaises(MacroConvertError) as ctx:
            validate_macro_definition(macro, tool_index={"demo-atomic", "demo-macro"})
        self.assertEqual(ctx.exception.error_code, "macro_retrieval_intent_missing")

    def test_verified_exclusive_requires_covered_tools(self) -> None:
        macro = _macro(
            MacroRoute(
                target_tool="demo-macro",
                selection="exclusive",
                verified=True,
                intent_key="demo.invoke",
            )
        )
        with self.assertRaises(MacroConvertError) as ctx:
            validate_macro_definition(macro, tool_index={"demo-atomic", "demo-macro"})
        self.assertEqual(ctx.exception.error_code, "macro_coverage_missing")

    def test_ambiguous_covered_tool_is_rejected(self) -> None:
        macro = _macro(
            MacroRoute(
                target_tool="demo-macro",
                selection="exclusive",
                verified=True,
                intent_key="demo.invoke",
                covered_tools=["send-to-agent"],
            )
        )
        with self.assertRaises(MacroConvertError) as ctx:
            validate_macro_definition(
                macro,
                tool_index={
                    "demo-atomic",
                    "pkg-a-send-to-agent",
                    "pkg-b-send-to-agent",
                },
            )
        self.assertEqual(ctx.exception.error_code, "macro_coverage_unresolved")

    def test_self_coverage_is_rejected(self) -> None:
        macro = _macro(
            MacroRoute(
                target_tool="demo-macro",
                selection="exclusive",
                verified=True,
                intent_key="demo.invoke",
                covered_tools=["demo-macro"],
            )
        )
        with self.assertRaises(MacroConvertError) as ctx:
            validate_macro_definition(
                macro,
                tool_index={"demo-atomic", "demo-macro"},
            )
        self.assertEqual(ctx.exception.error_code, "macro_coverage_self_reference")

    def test_unverified_exclusive_still_downgrades_to_prefer(self) -> None:
        macro = _macro(
            MacroRoute(
                target_tool="demo-macro",
                selection="exclusive",
                verified=False,
            )
        )
        validate_macro_definition(macro, tool_index={"demo-atomic", "demo-macro"})
        self.assertEqual(macro.routing.selection, "prefer")


if __name__ == "__main__":
    unittest.main()
