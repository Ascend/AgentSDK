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

"""Phase 4 handwritten macro convert tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.sop_converter.runtime.macros import (
    MacroConvertError,
    convert_handwritten_macros,
    load_macro_yaml,
)
from extensions.sop_converter.runtime.macros.catalog import resolve_macro
from extensions.sop_converter.runtime.macros.validation import validate_macro_definition
from extensions.sop_converter.tool_retrieval import load_tool_retrieval_index


_SAMPLE_MACRO = """
version: 1
name: create-and-invoke-agent
description: Create then invoke an agent
scope: bundle
enabled: true
workflow:
  inputs:
    agent_config:
      type: object
      required: true
    query:
      type: string
      required: true
  steps:
    - id: create
      kind: tool
      callable_ref: create-llm-agent
      args:
        agent_config: $input.agent_config
      output_schema:
        type: object
        properties:
          agent_id: {type: string}
        required: [agent_id]
    - id: invoke
      kind: tool
      callable_ref: invoke-existing-agent
      args:
        agent_ref: $steps.create.output.agent_id
        query: $input.query
  outputs:
    agent_id: $steps.create.output.agent_id
    output: $steps.invoke.output
routing:
  phrases:
    - 创建并调用 agent
  keywords:
    - 创建
    - agent
  target_tool: create-and-invoke-agent
  selection: exclusive
  verified: false
  priority: 90
provenance:
  kind: handwritten
"""


class TestMacroConvertPhase4(unittest.TestCase):
    def test_exclusive_unverified_downgrades_to_prefer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "macro.yaml"
            path.write_text(_SAMPLE_MACRO, encoding="utf-8")
            macro = load_macro_yaml(path)
            self.assertEqual(macro.routing.selection, "exclusive")
            validate_macro_definition(macro, tool_index={"create-llm-agent", "invoke-existing-agent"})
            self.assertEqual(macro.routing.selection, "prefer")

    def test_forward_binding_rejected(self) -> None:
        bad = _SAMPLE_MACRO.replace(
            "agent_ref: $steps.create.output.agent_id",
            "agent_ref: $steps.invoke.output.agent_id",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "macro.yaml"
            path.write_text(bad, encoding="utf-8")
            macro = load_macro_yaml(path)
            with self.assertRaises(MacroConvertError) as ctx:
                validate_macro_definition(
                    macro,
                    tool_index={"create-llm-agent", "invoke-existing-agent"},
                )
            self.assertEqual(ctx.exception.error_code, "macro_binding_forward")

    def test_unresolved_callable_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "macro.yaml"
            path.write_text(_SAMPLE_MACRO, encoding="utf-8")
            macro = load_macro_yaml(path)
            with self.assertRaises(MacroConvertError) as ctx:
                validate_macro_definition(macro, tool_index={"other-tool"})
            self.assertEqual(ctx.exception.error_code, "macro_callable_unresolved")

    def test_convert_persists_and_registers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sdk"
            source.mkdir()
            macros = source / "sop-macros"
            macros.mkdir()
            (macros / "create-and-invoke-agent.yaml").write_text(_SAMPLE_MACRO, encoding="utf-8")
            bundle = root / "bundle"
            bundle.mkdir()
            result = convert_handwritten_macros(
                source_dir=source,
                bundle_dir=bundle,
                tool_index={"create-llm-agent", "invoke-existing-agent"},
                persist=True,
                register_tools=True,
            )
            self.assertIn("create-and-invoke-agent", result.registered_tools)
            written = bundle / ".clawcodex" / "macros" / "create-and-invoke-agent.yaml"
            self.assertTrue(written.is_file())
            retrieval_path = bundle / ".clawcodex" / "tool-retrieval.yaml"
            self.assertTrue(retrieval_path.is_file())
            retrieval = load_tool_retrieval_index(bundle)
            self.assertEqual(
                retrieval.profile_for("create-and-invoke-agent").layer,  # type: ignore[union-attr]
                "macro",
            )
            spec = resolve_macro(
                {"catalog_id": "bundle:create-and-invoke-agent"},
                bundle_path=bundle,
            )
            self.assertEqual(spec.name, "create-and-invoke-agent")
            self.assertEqual(len(spec.steps), 2)

    def test_atomic_failure_leaves_no_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.yaml"
            first.write_text(_SAMPLE_MACRO, encoding="utf-8")
            second = root / "second.yaml"
            second.write_text(
                _SAMPLE_MACRO.replace(
                    "name: create-and-invoke-agent",
                    "name: second-macro",
                ),
                encoding="utf-8",
            )
            bundle = root / "bundle"
            bundle.mkdir()
            from extensions.sop_converter.runtime.macros import persist as persist_mod

            original = persist_mod.write_macro_yaml
            calls = {"n": 0}

            def _fail_second(path, macro):
                calls["n"] += 1
                if calls["n"] >= 2:
                    raise OSError("injected persist failure")
                original(path, macro)

            with (
                patch.object(persist_mod, "write_macro_yaml", side_effect=_fail_second),
                self.assertRaises(MacroConvertError),
            ):
                convert_handwritten_macros(
                    source_dir=None,
                    bundle_dir=bundle,
                    manifest_paths=[first, second],
                    tool_index={"create-llm-agent", "invoke-existing-agent"},
                    persist=True,
                    register_tools=False,
                )
            macros_dir = bundle / ".clawcodex" / "macros"
            if macros_dir.exists():
                self.assertEqual(list(macros_dir.glob("*.yaml")), [])

    def test_resolve_macro_from_manifest_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            macros = bundle / ".clawcodex" / "macros"
            macros.mkdir(parents=True)
            path = macros / "create-and-invoke-agent.yaml"
            path.write_text(_SAMPLE_MACRO, encoding="utf-8")
            # Ensure exclusive was downgraded when validated via resolve
            spec = resolve_macro(
                {"manifest": ".clawcodex/macros/create-and-invoke-agent.yaml"},
                bundle_path=bundle,
            )
            self.assertEqual(spec.steps[0].callable_ref, "create-llm-agent")


if __name__ == "__main__":
    unittest.main()
