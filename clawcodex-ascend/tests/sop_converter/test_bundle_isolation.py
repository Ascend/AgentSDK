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

"""Tests for L2/L3 bundle isolation (skill + tool registry, per-bundle storage)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clawcodex_ext.agent.tool_authoring.factory import build_tool_from_spec
from clawcodex_ext.agent.tool_authoring.persistence import (
    bundle_tool_dir,
    list_persisted_specs,
    save_spec,
)
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.skills.loader import clear_dynamic_skills, get_all_skills, get_registered_skill
from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.registry import ToolRegistry
from extensions.sop_converter.bundle_context import (
    activate_bundle_isolation,
    build_bundle_context,
    filter_tools_for_bundle,
    load_bundle_persisted_tools,
    set_active_bundle,
)
from extensions.sop_converter.bundle_skills import register_bundle_skills


def _pos_spec(name: str, *, bundle_id: str) -> AgentToolSpec:
    return AgentToolSpec(
        name=name,
        description=name,
        input_schema={"type": "object", "properties": {}},
        call_type="bash",
        call_impl='python3 -c "print(1)"',
        source="pos-converter",
        bundle_id=bundle_id,
    )


def _sop_spec(name: str, *, bundle_id: str) -> AgentToolSpec:
    return AgentToolSpec(
        name=name,
        description=name,
        input_schema={"type": "object", "properties": {}},
        call_type="bash",
        call_impl='python3 -c "print(1)"',
        source="sop-converter",
        bundle_id=bundle_id,
    )


class TestBundleSkillIsolation(unittest.TestCase):
    def tearDown(self) -> None:
        set_active_bundle(None)
        clear_dynamic_skills()

    def test_only_active_bundle_skills_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            bundle_a = ws / "JiuwenAgent_tool_test"
            bundle_b = ws / "JiuwenAgent_keyword_all"
            dir_a = ws / "skills" / "JiuwenAgent_tool_test"
            dir_b = ws / "skills" / "JiuwenAgent_keyword_all"
            bundle_a.mkdir(parents=True)
            bundle_b.mkdir(parents=True)
            dir_a.mkdir(parents=True)
            dir_b.mkdir(parents=True)

            (dir_a / "core_merged-skill.md").write_text(
                "---\nname: core_merged-skill\ndescription: bundle A\n---\n\n# A\n",
                encoding="utf-8",
            )
            (dir_b / "core_merged-skill.md").write_text(
                "---\nname: core_merged-skill\ndescription: bundle B\n---\n\n# B\n",
                encoding="utf-8",
            )

            load_a = register_bundle_skills(bundle_a, ws)
            bundle_ctx = build_bundle_context(
                bundle_path=bundle_a,
                skill_names=load_a.skill_names,
                skill_dirs=load_a.skill_dirs,
                tool_names=load_a.tool_names,
            )
            set_active_bundle(bundle_ctx)

            get_all_skills(project_root=ws)
            skill = get_registered_skill("core_merged-skill")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertIn("# A", skill.markdown_content or skill.content or "")
            self.assertNotIn("# B", skill.markdown_content or skill.content or "")

            missing = get_registered_skill("openjiuwen_merged-skill")
            self.assertIsNone(missing)


class TestBundleToolIsolation(unittest.TestCase):
    def tearDown(self) -> None:
        set_active_bundle(None)

    def test_filter_deferred_tools_to_bundle_allowlist(self) -> None:
        deferred_a = build_tool_from_spec(_pos_spec("openjiuwen-a-tool", bundle_id="bundle-a"))
        deferred_b = build_tool_from_spec(_pos_spec("openjiuwen-b-tool", bundle_id="bundle-b"))
        read_tool = build_tool(
            name="Read",
            input_schema={"type": "object", "properties": {}},
            call=lambda _i, _c: None,
            prompt="read",
        )
        bundle = build_bundle_context(
            bundle_path=Path("/tmp/bundle-a"),
            skill_names=["core_merged-skill"],
            skill_dirs=[],
            tool_names=["openjiuwen-a-tool"],
        )
        filtered = filter_tools_for_bundle(
            [read_tool, deferred_a, deferred_b],
            bundle,
        )
        names = {tool.name for tool in filtered}
        self.assertIn("Read", names)
        self.assertIn("openjiuwen-a-tool", names)
        self.assertNotIn("openjiuwen-b-tool", names)

    def test_persist_and_load_bundle_local_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "JiuwenAgent_tool_test"
            bundle_path.mkdir(parents=True)
            spec = _pos_spec("openjiuwen-agent-teams-cli-main", bundle_id=bundle_path.name)
            save_spec(spec, tool_dir=bundle_tool_dir(bundle_path))

            loaded_specs = list_persisted_specs(tool_dir=bundle_tool_dir(bundle_path))
            self.assertEqual(len(loaded_specs), 1)
            self.assertEqual(loaded_specs[0].bundle_id, "JiuwenAgent_tool_test")

            registry = ToolRegistry()
            bundle_ctx = build_bundle_context(
                bundle_path=bundle_path,
                skill_names=["harness_merged-skill"],
                skill_dirs=[],
                tool_names=[spec.name],
            )
            activate_bundle_isolation(registry, bundle_ctx)

            self.assertIsNotNone(registry.get(spec.name))
            foreign = build_tool_from_spec(_pos_spec("foreign-bundle-tool", bundle_id="other-bundle"))
            registry.register(foreign)
            removed = filter_tools_for_bundle(registry.list_tools(), bundle_ctx)
            names = {tool.name for tool in removed}
            self.assertIn(spec.name, names)
            self.assertNotIn("foreign-bundle-tool", names)


class TestBundleToolAllowlistFallback(unittest.TestCase):
    def test_register_bundle_skills_falls_back_to_persisted_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "JiuwenAgent_tool_test"
            bundle_path.mkdir(parents=True)
            ws = Path(tmp)
            skill_dir = ws / "skills" / "JiuwenAgent_tool_test"
            skill_dir.mkdir(parents=True)
            (skill_dir / "core_merged-skill.md").write_text(
                "---\nname: core_merged-skill\ndescription: test\n---\n\n# Core\n",
                encoding="utf-8",
            )
            save_spec(
                _pos_spec(
                    "openjiuwen-agent-teams-team-memory-dir",
                    bundle_id=bundle_path.name,
                ),
                tool_dir=bundle_tool_dir(bundle_path),
            )

            load = register_bundle_skills(bundle_path, ws)
            self.assertIn("openjiuwen-agent-teams-team-memory-dir", load.tool_names)

    def test_register_bundle_skills_accepts_sop_converter_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "JiuwenAgent_tool_test"
            bundle_path.mkdir(parents=True)
            ws = Path(tmp)
            skill_dir = ws / "skills" / "JiuwenAgent_tool_test"
            skill_dir.mkdir(parents=True)
            (skill_dir / "core_merged-skill.md").write_text(
                "---\nname: core_merged-skill\ndescription: test\n---\n\n# Core\n",
                encoding="utf-8",
            )
            save_spec(
                _sop_spec(
                    "openjiuwen-agent-teams-team-memory-dir",
                    bundle_id=bundle_path.name,
                ),
                tool_dir=bundle_tool_dir(bundle_path),
            )

            load = register_bundle_skills(bundle_path, ws)
            self.assertIn("openjiuwen-agent-teams-team-memory-dir", load.tool_names)

    def test_register_bundle_skills_accepts_composite_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "JiuwenAgent_tool_test"
            bundle_path.mkdir(parents=True)
            ws = Path(tmp)
            skill_dir = ws / "skills" / "JiuwenAgent_tool_test"
            skill_dir.mkdir(parents=True)
            (skill_dir / "core_merged-skill.md").write_text(
                "---\nname: core_merged-skill\ndescription: test\n---\n\n# Core\n",
                encoding="utf-8",
            )
            save_spec(
                AgentToolSpec(
                    name="invoke-existing-agent",
                    description="Invoke a persisted agent",
                    input_schema={"type": "object", "properties": {}},
                    call_type="bash",
                    call_impl='python3 -c "print(1)"',
                    source="composite-tool",
                    bundle_id=bundle_path.name,
                ),
                tool_dir=bundle_tool_dir(bundle_path),
            )

            load = register_bundle_skills(bundle_path, ws)
            self.assertIn("invoke-existing-agent", load.tool_names)

    def test_activate_keeps_bundle_tools_when_allowlist_from_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "JiuwenAgent_tool_test"
            bundle_path.mkdir(parents=True)
            spec_name = "openjiuwen-agent-teams-team-memory-dir"
            save_spec(
                _pos_spec(spec_name, bundle_id=bundle_path.name),
                tool_dir=bundle_tool_dir(bundle_path),
            )
            registry = ToolRegistry()
            bundle_ctx = build_bundle_context(
                bundle_path=bundle_path,
                skill_names=["core_merged-skill"],
                skill_dirs=[],
                tool_names=[spec_name],
            )
            activate_bundle_isolation(registry, bundle_ctx)
            self.assertIsNotNone(registry.get(spec_name))

    def test_load_skips_duplicate_alias_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "JiuwenAgent_tool_test"
            bundle_path.mkdir(parents=True)
            shared_alias = "openjiuwen.agent_teams.schema.register_tool_provider"
            first = AgentToolSpec(
                name="tool-a",
                description="first",
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl='python3 -c "print(1)"',
                aliases=(shared_alias,),
                source="pos-converter",
                bundle_id=bundle_path.name,
            )
            second = AgentToolSpec(
                name="tool-b",
                description="second",
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl='python3 -c "print(2)"',
                aliases=(shared_alias,),
                source="pos-converter",
                bundle_id=bundle_path.name,
            )
            save_spec(first, tool_dir=bundle_tool_dir(bundle_path))
            save_spec(second, tool_dir=bundle_tool_dir(bundle_path))

            registry = ToolRegistry()
            bundle_ctx = build_bundle_context(
                bundle_path=bundle_path,
                skill_names=["core_merged-skill"],
                skill_dirs=[],
                tool_names=["tool-a", "tool-b"],
            )
            set_active_bundle(bundle_ctx)
            loaded = load_bundle_persisted_tools(registry, bundle_ctx.bundle_path)
            self.assertEqual(loaded, 1)
            self.assertIsNotNone(registry.get("tool-a"))
            self.assertIsNone(registry.get("tool-b"))


class TestSdkSourceWorkingDirectory(unittest.TestCase):
    def test_apply_sdk_source_adds_to_allowed_roots(self):
        from clawcodex_ext.tool_system.context import ToolContext
        from extensions.sop_converter.bundle_context import (
            apply_sdk_source_working_directory,
        )

        with tempfile.TemporaryDirectory() as sdk_dir:
            sdk_path = Path(sdk_dir)
            bundle = build_bundle_context(
                bundle_path=Path("/tmp/bundle"),
                skill_names=[],
                skill_dirs=[],
                tool_names=[],
                sdk_source_dir=sdk_path,
            )
            ctx = ToolContext(workspace_root=Path("/tmp/workspace"))
            added = apply_sdk_source_working_directory(ctx, bundle)
            self.assertEqual(added, sdk_path.resolve())
            self.assertIn(sdk_path.resolve(), ctx.additional_working_directories)
            self.assertIn(sdk_path.resolve(), ctx.allowed_roots())

    def test_apply_sdk_source_is_idempotent(self):
        from clawcodex_ext.tool_system.context import ToolContext
        from extensions.sop_converter.bundle_context import (
            apply_sdk_source_working_directory,
        )

        with tempfile.TemporaryDirectory() as sdk_dir:
            sdk_path = Path(sdk_dir)
            bundle = build_bundle_context(
                bundle_path=Path("/tmp/bundle"),
                skill_names=[],
                skill_dirs=[],
                tool_names=[],
                sdk_source_dir=sdk_path,
            )
            ctx = ToolContext(workspace_root=Path("/tmp/workspace"))
            apply_sdk_source_working_directory(ctx, bundle)
            apply_sdk_source_working_directory(ctx, bundle)
            self.assertEqual(len(ctx.additional_working_directories), 1)


if __name__ == "__main__":
    unittest.main()
