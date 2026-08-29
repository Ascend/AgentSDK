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

"""P70-E: pyproject.toml plugin manifest format tests.

Verifies that plugin manifests can be declared inside a ``pyproject.toml``
file under the ``[tool.clawcodex.plugin]`` table, with parity to
``plugin.yaml`` / ``plugin.json`` for all supported fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.plugins.loader import (
    MANIFEST_FILES,
    clear_loaded_plugins,
    discover_plugins,
    load_plugin_from_directory,
    scan_plugin_directory,
)
from src.plugins.types import PluginError


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="tomllib requires Python 3.11+",
)


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded_plugins()
    yield
    clear_loaded_plugins()


def _write_pyproject(plugin_dir: Path, table_lines: list[str]) -> None:
    """Write a pyproject.toml with a [tool.clawcodex.plugin] table."""
    plugin_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "[build-system]",
            'requires = ["setuptools"]',
            "build-backend = 'setuptools.build_meta'",
            "",
        ]
        + ["[tool.clawcodex.plugin]"]
        + table_lines
        + [""]
    )
    (plugin_dir / "pyproject.toml").write_text(body, encoding="utf-8")


class TestPyprojectManifestLoad:
    def test_pyproject_in_manifest_files(self):
        assert "pyproject.toml" in MANIFEST_FILES

    def test_load_valid(self, tmp_path):
        plugin_dir = tmp_path / "toml-plugin"
        _write_pyproject(
            plugin_dir,
            ['name = "toml-plugin"', 'description = "A TOML plugin"', 'version = "1.2.3"'],
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.name == "toml-plugin"
        assert plugin.manifest.description == "A TOML plugin"
        assert plugin.manifest.version == "1.2.3"
        assert plugin.enabled is True

    def test_load_with_hooks(self, tmp_path):
        plugin_dir = tmp_path / "hooked-toml"
        _write_pyproject(
            plugin_dir,
            [
                'name = "hooked-toml"',
                "[tool.clawcodex.plugin.hooks.PreToolUse]",
                'command = "echo"',
            ],
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.hooks_config is not None
        assert "PreToolUse" in plugin.hooks_config

    def test_load_with_mcp_servers(self, tmp_path):
        plugin_dir = tmp_path / "mcp-toml"
        _write_pyproject(
            plugin_dir,
            [
                'name = "mcp-toml"',
                "[tool.clawcodex.plugin.mcp_servers.fetch]",
                'command = "uvx"',
                'args = ["mcp-fetch"]',
            ],
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.mcp_servers is not None
        assert "fetch" in plugin.mcp_servers

    def test_load_with_agents_paths_snake_case(self, tmp_path):
        plugin_dir = tmp_path / "agents-toml"
        _write_pyproject(
            plugin_dir,
            [
                'name = "agents-toml"',
                'agents_paths = ["agents", "skills"]',
            ],
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert len(plugin.agents_paths) == 2
        # Relative paths resolved against plugin dir
        assert plugin.agents_paths[0].endswith("agents")

    def test_load_disabled(self, tmp_path):
        plugin_dir = tmp_path / "disabled-toml"
        _write_pyproject(
            plugin_dir,
            ['name = "disabled-toml"', "enabled = false"],
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.enabled is False

    def test_load_with_repository_and_permissions(self, tmp_path):
        plugin_dir = tmp_path / "meta-toml"
        _write_pyproject(
            plugin_dir,
            [
                'name = "meta-toml"',
                'repository = "https://gitcode.com/u/meta-toml"',
                'permissions = ["read", "network"]',
            ],
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.repository == "https://gitcode.com/u/meta-toml"


class TestPyprojectManifestErrors:
    def test_missing_table_raises(self, tmp_path):
        plugin_dir = tmp_path / "no-table"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools"]\n', encoding="utf-8")
        with pytest.raises(PluginError) as exc:
            load_plugin_from_directory(plugin_dir)
        assert "tool.clawcodex.plugin" in str(exc.value)

    def test_table_not_mapping_raises(self, tmp_path):
        plugin_dir = tmp_path / "bad-table"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text("[tool.clawcodex]\nplugin = 42\n", encoding="utf-8")
        with pytest.raises(PluginError) as exc:
            load_plugin_from_directory(plugin_dir)
        assert "must be a table" in str(exc.value)

    def test_invalid_toml_raises(self, tmp_path):
        plugin_dir = tmp_path / "bad-toml"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text("[tool.clawcodex.plugin\nname = =\n", encoding="utf-8")
        with pytest.raises(PluginError) as exc:
            load_plugin_from_directory(plugin_dir)
        assert "Invalid TOML" in str(exc.value)

    def test_invalid_name_validation(self, tmp_path):
        plugin_dir = tmp_path / "bad-name"
        _write_pyproject(plugin_dir, ['name = "9bad"'])
        with pytest.raises(PluginError) as exc:
            load_plugin_from_directory(plugin_dir)
        assert "Invalid manifest" in str(exc.value)


class TestPyprojectDiscovery:
    def test_discover_finds_pyproject(self, tmp_path):
        base = tmp_path / "plugins"
        plugin_dir = base / "disc-toml"
        _write_pyproject(plugin_dir, ['name = "disc-toml"'])
        # An unrelated pyproject.toml-style file at base root shouldn't matter
        result = discover_plugins(base)
        assert len(result.plugins) == 1
        assert result.plugins[0].name == "disc-toml"

    def test_scan_recursive_finds_pyproject(self, tmp_path):
        root = tmp_path / "monorepo"
        deep = root / "team-a" / "plugin-x"
        _write_pyproject(deep, ['name = "plugin-x"', 'version = "0.1.0"'])
        result = scan_plugin_directory(root, recursive=True, max_depth=3)
        names = [p.name for p in result.plugins]
        assert "plugin-x" in names

    def test_pyproject_not_treated_as_plugin_when_no_table(self, tmp_path):
        base = tmp_path / "plugins"
        plugin_dir = base / "lib-pkg"
        plugin_dir.mkdir(parents=True)
        # A normal python package pyproject.toml without the plugin table
        (plugin_dir / "pyproject.toml").write_text('[project]\nname = "lib-pkg"\nversion = "1.0.0"\n', encoding="utf-8")
        result = discover_plugins(base)
        # Should surface as an error (table missing) but not as a plugin
        assert len(result.plugins) == 0
        assert any("lib-pkg" in e.plugin_name for e in result.errors)
