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

"""Tests for SOP bundle source-exploration runtime guards."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from extensions.sop_converter.sop_exploration_guard import check_bundle_source_exploration

_SDK_ROOT = Path("/mnt/d/projects/JiuwenAgent")
_WIN_SDK_ROOT = Path("D:/projects/JiuwenAgent")

# Registered names include a neutral SDK tool to prove SDK-agnostic detection.
_DEFAULT_TOOL_NAMES = frozenset(
    {
        "openjiuwen-agent-teams-team-memory-dir",
        "acme-sdk-run-job",
    }
)
_DEFAULT_SKILL_NAMES = frozenset({"openjiuwen_merged-skill"})


def _mock_bundle(
    *,
    sdk_root: Path | None = _SDK_ROOT,
    tool_names=None,
    skill_names=None,
):
    return SimpleNamespace(
        bundle_name="JiuwenAgent_tool_test",
        sdk_source_dir=sdk_root,
        tool_names=frozenset(tool_names) if tool_names is not None else _DEFAULT_TOOL_NAMES,
        skill_names=frozenset(skill_names) if skill_names is not None else _DEFAULT_SKILL_NAMES,
    )


def _ctx(*, agent_type: str | None = None, messages=None):
    return SimpleNamespace(
        agent_type=agent_type,
        startup_agent=None,
        messages=messages or [],
        cwd="/tmp/ws",
        workspace_root="/tmp/ws",
        _agent_dir_override=None,
    )


class TestSopExplorationGuard(unittest.TestCase):
    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_no_bundle_is_noop(self, mock_bundle) -> None:
        mock_bundle.return_value = None
        err = check_bundle_source_exploration(
            "Grep",
            {"pattern": "team-memory-dir"},
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_glob_spec_yaml(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Glob",
            {"pattern": "**/spec.yaml", "path": "/tmp/ws"},
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_bash_ls(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Bash",
            {"command": "ls -la"},
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_sdk_source_ls(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Bash",
            {"command": "ls /mnt/d/projects/JiuwenAgent/openjiuwen/agent_teams/"},
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_sdk_source_read_wsl_path(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Read",
            {
                "file_path": ("/mnt/d/projects/JiuwenAgent/openjiuwen/agent_teams/cli/app.py"),
            },
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_sdk_source_read_windows_path(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Read",
            {
                "file_path": ("D:/projects/JiuwenAgent/openjiuwen/agent_teams/cli/app.py"),
            },
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_sdk_source_glob(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Glob",
            {
                "pattern": "**/*.py",
                "path": "/mnt/d/projects/JiuwenAgent/openjiuwen/agent_teams/cli",
            },
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_open_ended_sdk_grep(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Grep",
            {
                "pattern": "run.*team.*cli",
                "path": "/mnt/d/projects/JiuwenAgent/openjiuwen/agent_teams",
            },
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_blocks_grep_for_tool_discovery(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Grep",
            {"pattern": "team-memory-dir", "path": "/tmp/ws"},
            _ctx(agent_type="clawcodex-overview"),
            agent_definitions=[SimpleNamespace(agent_type="openjiuwen_merged-agent")],
        )
        self.assertIsNotNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_blocks_grep_tool_name_even_under_sdk_root(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Grep",
            {
                "pattern": "openjiuwen-agent-teams-team-memory-dir",
                "path": "/mnt/d/projects/JiuwenAgent/openjiuwen",
            },
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNotNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_blocks_grep_on_wrong_sdk_path(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Grep",
            {
                "pattern": "team.memory.dir",
                "path": "/tmp/ws/JiuwenAgent/openjiuwen",
            },
            _ctx(agent_type="clawcodex-overview"),
            agent_definitions=[SimpleNamespace(agent_type="memory-agent")],
        )
        self.assertIsNotNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_blocks_wrong_sdk_path_any_sdk(self, mock_bundle) -> None:
        """The wrong-workspace-SDK-path guard is SDK-agnostic: the checkout
        dir name is derived from the manifest ``sdk_source_dir``, not
        hardcoded to JiuwenAgent.
        """
        mock_bundle.return_value = _mock_bundle(sdk_root=Path("/mnt/d/projects/AcmeAgent"))
        err = check_bundle_source_exploration(
            "Grep",
            {"pattern": "team.memory.dir", "path": "/tmp/ws/AcmeAgent/sdk_src"},
            _ctx(agent_type="clawcodex-overview"),
            agent_definitions=[SimpleNamespace(agent_type="memory-agent")],
        )
        self.assertIsNotNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_sdk_source_read_any_sdk(self, mock_bundle) -> None:
        """The authorized manifest SDK root stays readable for any SDK name."""
        mock_bundle.return_value = _mock_bundle(sdk_root=Path("/mnt/d/projects/AcmeAgent"))
        err = check_bundle_source_exploration(
            "Read",
            {"file_path": "/mnt/d/projects/AcmeAgent/sdk_src/agent_teams/cli/app.py"},
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_blocks_grep_for_registered_tool_tail_any_sdk(self, mock_bundle) -> None:
        """A memorable tail of a registered tool name is hunting for any SDK."""
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Grep",
            {"pattern": "run-job", "path": "/tmp/ws"},
            _ctx(agent_type="clawcodex-overview"),
            agent_definitions=[SimpleNamespace(agent_type="memory-agent")],
        )
        self.assertIsNotNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_allows_grep_for_unregistered_sdk_name(self, mock_bundle) -> None:
        """Unregistered SDK names are not tool-hunting (fail-open, no brand prefix)."""
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Grep",
            {"pattern": "openjiuwen-fetch-document", "path": "/tmp/ws"},
            _ctx(agent_type="clawcodex-overview"),
            agent_definitions=[SimpleNamespace(agent_type="memory-agent")],
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_sdk_tool_call_failed_allows_diagnostics_any_sdk(self, mock_bundle) -> None:
        """A failed registered SDK tool call permits agent-tools diagnostics."""
        mock_bundle.return_value = _mock_bundle()
        messages = [
            SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="acme-sdk-run-job", id="t1")]),
            SimpleNamespace(content=[SimpleNamespace(type="tool_result", tool_use_id="t1", is_error=True)]),
        ]
        err = check_bundle_source_exploration(
            "Read",
            {"file_path": "/tmp/ws/agent-tools/scripts/run.sh"},
            _ctx(agent_type="memory-agent", messages=messages),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_allows_runtime_data_under_configured_home(self, mock_bundle) -> None:
        """Runtime data under SOP_SDK_RUNTIME_HOME is allowed for any SDK."""
        mock_bundle.return_value = _mock_bundle()
        with patch.dict(os.environ, {"SOP_SDK_RUNTIME_HOME": "/root/.acmesdk"}, clear=False):
            err = check_bundle_source_exploration(
                "Bash",
                {"command": "find /root/.acmesdk/team-data -type f 2>/dev/null"},
                _ctx(agent_type="clawcodex-overview"),
            )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_agent_loader_failure_is_logged_and_degrades(self, mock_bundle) -> None:
        """agent_loader failures are logged (not swallowed) and the overview
        guard degrades to an empty-agent message instead of crashing.
        """
        from extensions.sop_converter.adapters import DEFAULTS

        mock_bundle.return_value = _mock_bundle()
        with patch.object(DEFAULTS, "agent_loader", side_effect=AttributeError("agent_loader boom")):
            with self.assertLogs("extensions.sop_converter.runtime.sop_exploration_guard", level="ERROR") as logs:
                err = check_bundle_source_exploration(
                    "Grep",
                    {"pattern": "team-memory-dir", "path": "/tmp/ws"},
                    _ctx(agent_type="clawcodex-overview"),
                )
        self.assertIsNotNone(err)
        self.assertTrue(any("agent_loader failed" in line for line in logs.output))

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_agent_loader_importerror_degrades_quietly(self, mock_bundle) -> None:
        """ImportError (migration-period missing module) degrades to an empty
        agent list without error logs.
        """
        from extensions.sop_converter.adapters import DEFAULTS

        mock_bundle.return_value = _mock_bundle()
        with patch.object(DEFAULTS, "agent_loader", side_effect=ImportError("missing module")):
            err = check_bundle_source_exploration(
                "Grep",
                {"pattern": "team-memory-dir", "path": "/tmp/ws"},
                _ctx(agent_type="clawcodex-overview"),
            )
        self.assertIsNotNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_domain_agent_blocks_grep_for_kebab_tool_before_skill(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Grep",
            {"pattern": "openjiuwen-agent-teams-team-memory-dir"},
            _ctx(agent_type="memory-agent"),
        )
        self.assertIsNotNone(err)
        self.assertIn("Skill", err or "")

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_domain_agent_allows_read_spec_before_skill(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Read",
            {"file_path": "/tmp/ws/spec.yaml"},
            _ctx(agent_type="openjiuwen_merged-agent"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_domain_agent_allows_sdk_source_read_before_skill(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Read",
            {
                "file_path": ("D:/projects/JiuwenAgent/openjiuwen/agent_teams/cli/app.py"),
            },
            _ctx(agent_type="openjiuwen_merged-agent"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_allows_find_in_openjiuwen_runtime(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Bash",
            {"command": "find /root/.openjiuwen/.agent_teams/team/ -type f 2>/dev/null"},
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_overview_blocks_find_xargs_grep_for_tools(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Bash",
            {"command": "find .clawcodex -name '*.py' | xargs grep -l team-memory-dir"},
            _ctx(agent_type="clawcodex-overview"),
            agent_definitions=[SimpleNamespace(agent_type="memory-agent")],
        )
        self.assertIsNotNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_blocks_read_sdk_test_tree_config(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Read",
            {
                "file_path": ("/mnt/d/projects/JiuwenAgent/tests/system_tests/agent_swarm/config.yaml"),
            },
            _ctx(agent_type="openjiuwen_merged-agent"),
        )
        self.assertIsNotNone(err)
        self.assertIn("tests/fixtures", err or "")
        self.assertIn("交互式终端停损", err or "")

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_blocks_bash_find_yaml_in_tests(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Bash",
            {
                "command": ("find /mnt/d/projects/JiuwenAgent -name '*.yaml' -path '*/tests/*' 2>/dev/null"),
            },
            _ctx(agent_type="openjiuwen_merged-agent"),
        )
        self.assertIsNotNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_still_allows_read_sdk_cli_app_py(self, mock_bundle) -> None:
        mock_bundle.return_value = _mock_bundle()
        err = check_bundle_source_exploration(
            "Read",
            {
                "file_path": ("/mnt/d/projects/JiuwenAgent/openjiuwen/agent_teams/cli/app.py"),
            },
            _ctx(agent_type="openjiuwen_merged-agent"),
        )
        self.assertIsNone(err)


class TestSdkPathNormalization(unittest.TestCase):
    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_wsl_manifest_allows_windows_sdk_read(self, mock_bundle) -> None:
        if sys.platform != "win32":
            self.skipTest("Windows-only path normalization")
        mock_bundle.return_value = _mock_bundle(sdk_root=_SDK_ROOT)
        err = check_bundle_source_exploration(
            "Read",
            {"file_path": str(_WIN_SDK_ROOT / "openjiuwen/agent_teams/cli/app.py")},
            _ctx(agent_type="clawcodex-overview"),
        )
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
