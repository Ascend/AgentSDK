# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#

"""Tests for pos-converter search tag generation (Plan B)."""

from __future__ import annotations

import unittest

from extensions.sop_converter.core.search_tags import generate_search_tags
from extensions.sop_converter.core.source_parser import ParamSpec, SourceOperation


class TestGenerateSearchTags(unittest.TestCase):
    def test_snake_case_method(self) -> None:
        op = SourceOperation(
            name="run_team_cli",
            description="Bring up the Team CLI against a pre-built spec set.",
        )
        tags = generate_search_tags(op, comp_name="openjiuwen.agent_teams.cli")
        joined = " ".join(tags)
        self.assertIn("run_team_cli", tags)
        self.assertIn("run team cli", tags)
        self.assertIn("team", tags)
        self.assertIn("cli", tags)
        self.assertIn("bring", tags)
        self.assertIn("agent_teams", joined.replace(" ", ""))  # comp segments present
        self.assertIn("agent teams", joined)

    def test_camel_case_class_and_method(self) -> None:
        op = SourceOperation(
            name="should_continue",
            description="Return whether the task loop should continue.",
            class_name="LoopCoordinator",
        )
        tags = generate_search_tags(op, comp_name="openjiuwen.harness.task_loop")
        joined = " ".join(tags)
        self.assertIn("should_continue", tags)
        self.assertIn("should continue", tags)
        self.assertIn("loopcoordinator", tags)
        self.assertIn("loop coordinator", joined)
        self.assertIn("loop", tags)
        self.assertIn("coordinator", tags)
        self.assertIn("continue", tags)

    def test_deduplicates_tags(self) -> None:
        op = SourceOperation(
            name="team_home",
            description="Return the team home directory.",
        )
        tags = generate_search_tags(op, comp_name="agent_teams")
        self.assertEqual(len(tags), len(set(tags)))

    def test_snake_case_name_spaced_in_tags(self) -> None:
        op = SourceOperation(
            name="open_foo_panel",
            description="Open the Foo panel.",
        )
        tags = generate_search_tags(op, comp_name="example.foo")
        joined = " ".join(tags).lower()
        self.assertIn("open foo panel", joined)
        self.assertIn("open", joined)

    def test_includes_parameter_names(self) -> None:
        op = SourceOperation(
            name="run_team_cli",
            description="Run the team CLI.",
            parameters=[
                ParamSpec(name="yaml_paths", type_hint="list", description="YAML paths"),
                ParamSpec(name="*args", type_hint=None, required=False),
            ],
        )
        tags = generate_search_tags(op)
        self.assertIn("yaml_paths", tags)
        self.assertIn("yaml paths", tags)
        self.assertNotIn("*args", tags)


if __name__ == "__main__":
    unittest.main()
