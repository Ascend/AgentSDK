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

"""Tests for workspace POS bundle auto-discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clawcodex_ext.agent.tool_authoring.persistence import bundle_tool_dir, save_spec
from extensions.sop_converter.bundle_discovery import (
    discover_workspace_bundle,
    list_workspace_bundle_candidates,
    overview_has_sop_skills,
)
from extensions.sop_converter.bundle_manifest import write_bundle_manifest
from extensions.sop_converter.bundle_context import (
    build_bundle_context,
    filter_tools_for_bundle,
)
from clawcodex_ext.agent.tool_authoring.factory import build_tool_from_spec
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.build_tool import build_tool


def _pos_spec(name: str, *, bundle_id: str, aliases: tuple[str, ...] = ()) -> AgentToolSpec:
    return AgentToolSpec(
        name=name,
        description=name,
        input_schema={"type": "object", "properties": {}},
        call_type="bash",
        call_impl='python3 -c "print(1)"',
        source="pos-converter",
        bundle_id=bundle_id,
        aliases=aliases,
    )


class TestBundleDiscovery(unittest.TestCase):
    def test_overview_has_sop_skills(self) -> None:
        self.assertTrue(overview_has_sop_skills({"skills": ["openjiuwen_merged-skill", "Read"]}))
        self.assertFalse(overview_has_sop_skills({"skills": ["Read", "Bash"]}))

    def test_discovers_bundle_from_clawcodex_and_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            # Deliberately SDK-neutral name: discovery must not rely on a
            # "JiuwenAgent" (or any SDK) directory-name prefix.
            bundle_name = "demo_agent_tool_test"
            bundle_root = ws / ".clawcodex" / bundle_name
            bundle_root.mkdir(parents=True)
            save_spec(
                _pos_spec("openjiuwen-agent-teams-cli-run-team-cli", bundle_id=bundle_name),
                tool_dir=bundle_tool_dir(bundle_root),
            )
            skill_dir = ws / "skills" / bundle_name
            skill_dir.mkdir(parents=True)
            (skill_dir / "openjiuwen_merged-skill.md").write_text(
                "---\nname: openjiuwen_merged-skill\ndescription: test\n---\n\nbody\n",
                encoding="utf-8",
            )

            candidates = list_workspace_bundle_candidates(ws)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].name, bundle_name)

            found = discover_workspace_bundle(
                ws,
                agent_skills=["openjiuwen_merged-skill", "core_merged-skill"],
            )
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found.name, bundle_name)

    def test_discovers_bundle_from_manifest_only(self) -> None:
        """A bundle.json manifest alone marks a POS bundle root — no tools, no
        skill files, no SDK-name convention required.
        """
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            bundle_name = "manifest_only_bundle"
            bundle_root = ws / "skills" / bundle_name
            sdk_root = ws / "sdk-src"
            sdk_root.mkdir(parents=True)
            write_bundle_manifest(bundle_root, sdk_source_dir=sdk_root)

            candidates = list_workspace_bundle_candidates(ws)
            self.assertEqual([c.name for c in candidates], [bundle_name])

            found = discover_workspace_bundle(ws, agent_skills=[])
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found.name, bundle_name)

    def test_plain_skill_dir_is_not_a_bundle_candidate(self) -> None:
        """Ordinary project skill folders (``*-skill.md`` but no manifest /
        persisted tool specs) must NOT be treated as POS bundles — the
        regression the old ``JiuwenAgent``-prefix filter used to prevent.
        """
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            plain_dir = ws / "skills" / "plain_project_skill_zz"
            plain_dir.mkdir(parents=True)
            (plain_dir / "plain_project_skill_zz-skill.md").write_text(
                "---\nname: plain_project_skill_zz-skill\ndescription: test\n---\n\nbody\n",
                encoding="utf-8",
            )

            self.assertEqual(list_workspace_bundle_candidates(ws), [])
            self.assertIsNone(
                discover_workspace_bundle(
                    ws,
                    agent_skills=["plain_project_skill_zz-skill"],
                )
            )


class TestBundleAliasAllowlist(unittest.TestCase):
    def test_filter_matches_dot_alias_tool_names(self) -> None:
        deferred = build_tool_from_spec(
            _pos_spec(
                "openjiuwen-agent-teams-cli-run-team-cli",
                bundle_id="bundle-a",
                aliases=("openjiuwen.agent_teams.cli.run_team_cli", "cli.run_team_cli"),
            )
        )
        read_tool = build_tool(
            name="Read",
            input_schema={"type": "object", "properties": {}},
            call=lambda _i, _c: None,
            prompt="read",
        )
        bundle = build_bundle_context(
            bundle_path=Path("/tmp/bundle-a"),
            skill_names=["openjiuwen_merged-skill"],
            skill_dirs=[],
            tool_names=["cli.run_team_cli"],
        )
        filtered = filter_tools_for_bundle([read_tool, deferred], bundle)
        names = {tool.name for tool in filtered}
        self.assertIn("openjiuwen-agent-teams-cli-run-team-cli", names)


if __name__ == "__main__":
    unittest.main()
