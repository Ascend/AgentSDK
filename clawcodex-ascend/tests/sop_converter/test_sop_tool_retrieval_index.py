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

"""ToolRetrievalIndex compile and persistence tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extensions.sop_converter.tool_retrieval import (
    ToolRetrievalIndex,
    index_from_routes,
    load_tool_retrieval_index,
    resolve_tool_references,
    write_tool_retrieval_index,
)

# ``runtime.macros.models`` and ``clawcodex_ext.agent.tool_authoring.spec``
# arrive with the macros / tool-authoring PRs (``tool_authoring`` also pulls in
# ``extensions.tool_system_ext``); skip the classes that use them until those
# merge so this PR does not error when run independently or in CI before them.
try:
    from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
    from extensions.sop_converter.runtime.macros.models import MacroRoute

    _HAS_INDEX_DEPENDENCIES = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - dependency not merged yet
    AgentToolSpec = None  # type: ignore[assignment]
    MacroRoute = None  # type: ignore[assignment]
    _HAS_INDEX_DEPENDENCIES = False


class TestToolReferenceResolution(unittest.TestCase):
    def test_normalized_suffix_resolves_namespaced_atomic(self) -> None:
        names = [
            "openjiuwen-core-application-llm-agent-llmagent-invoke",
            "invoke-existing-agent",
        ]
        self.assertEqual(
            resolve_tool_references(["llmagent-invoke"], names, require_unique=True),
            ["openjiuwen-core-application-llm-agent-llmagent-invoke"],
        )

    def test_ambiguous_suffix_is_rejected_by_convert_mode(self) -> None:
        with self.assertRaises(ValueError):
            resolve_tool_references(
                ["send-to-agent"],
                ["pkg-a-send-to-agent", "pkg-b-send-to-agent"],
                require_unique=True,
            )

    def test_runtime_mode_returns_all_namespaced_matches(self) -> None:
        self.assertEqual(
            resolve_tool_references(
                ["send-to-agent"],
                ["pkg-a-send-to-agent", "pkg-b-send-to-agent"],
            ),
            ["pkg-a-send-to-agent", "pkg-b-send-to-agent"],
        )


@unittest.skipUnless(
    _HAS_INDEX_DEPENDENCIES,
    "requires MacroRoute / AgentToolSpec from the macros & tool-authoring PRs",
)
class TestToolRetrievalIndex(unittest.TestCase):
    def test_compile_preserves_source_call_type_and_layers(self) -> None:
        route = MacroRoute(
            target_tool="invoke-existing-agent",
            selection="exclusive",
            verified=True,
            intent_key="agent.invoke_existing",
            covered_tools=["llmagent-invoke"],
        )
        specs = [
            AgentToolSpec(
                name="invoke-existing-agent",
                description="macro",
                input_schema={"type": "object", "properties": {}},
                call_type="workflow",
                call_impl={"catalog_id": "builtin:invoke-existing-agent"},
                source="composite-tool",
            ),
            AgentToolSpec(
                name="pkg-llmagent-invoke",
                description="atomic",
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl="echo ok",
                source="sop-converter",
            ),
        ]
        index = index_from_routes(
            [route],
            [spec.name for spec in specs],
            tool_specs=specs,
            require_unique=True,
        )
        macro = index.profile_for("invoke-existing-agent")
        atomic = index.profile_for("pkg-llmagent-invoke")
        self.assertEqual(macro.layer, "macro")  # type: ignore[union-attr]
        self.assertEqual(macro.call_type, "workflow")  # type: ignore[union-attr]
        self.assertEqual(atomic.layer, "atomic")  # type: ignore[union-attr]
        self.assertEqual(atomic.source, "sop-converter")  # type: ignore[union-attr]

    def test_yaml_round_trip(self) -> None:
        route = MacroRoute(
            target_tool="macro",
            selection="exclusive",
            verified=True,
            intent_key="demo.invoke",
            covered_tools=["atomic"],
        )
        index = index_from_routes(
            [route],
            ["macro", "atomic"],
            require_unique=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_tool_retrieval_index(index, Path(tmp))
            self.assertTrue(path.is_file())
            loaded = load_tool_retrieval_index(Path(tmp))
        self.assertIsInstance(loaded, ToolRetrievalIndex)
        self.assertEqual(loaded.coverage_for_macro("macro").intent_key, "demo.invoke")  # type: ignore[union-attr]
        self.assertEqual(loaded.covered_names("macro", ["atomic"]), ["atomic"])


if __name__ == "__main__":
    unittest.main()
