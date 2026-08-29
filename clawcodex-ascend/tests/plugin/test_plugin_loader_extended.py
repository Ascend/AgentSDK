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

"""Tests for YAML manifest support and extended loader functionality."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.plugins.loader import (
    MANIFEST_FILES,
    clear_loaded_plugins,
    discover_all_plugins,
    discover_entry_point_plugins,
    fire_lifecycle_event,
    get_default_plugin_dirs,
    get_enabled_plugins,
    get_loaded_plugin,
    load_plugin_from_directory,
    on_lifecycle,
    register_plugin,
    toggle_plugin_enabled,
)
from src.plugins.types import LoadedPlugin, PluginError, PluginManifest


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded_plugins()
    yield
    clear_loaded_plugins()


# ── YAML manifest tests ──────────────────────────────────────────────


class TestYamlManifestSupport:
    """Verify that plugin.yaml / plugin.yml manifests are accepted."""

    def _make_yaml_plugin(
        self,
        base: Path,
        name: str,
        extra: dict | None = None,
        ext: str = ".yaml",
    ) -> Path:
        plugin_dir = base / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": name,
            "description": f"Plugin {name}",
            "version": "1.0.0",
        }
        if extra:
            manifest.update(extra)
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")
        (plugin_dir / f"plugin{ext}").write_text(
            yaml.dump(manifest),
            encoding="utf-8",
        )
        return plugin_dir

    def test_load_yaml_manifest(self, tmp_path):
        plugin_dir = self._make_yaml_plugin(tmp_path, "yaml-plugin")
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.name == "yaml-plugin"
        assert plugin.manifest.version == "1.0.0"

    def test_load_yml_manifest(self, tmp_path):
        plugin_dir = self._make_yaml_plugin(tmp_path, "yml-plugin", ext=".yml")
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.name == "yml-plugin"

    def test_yaml_with_hooks(self, tmp_path):
        plugin_dir = self._make_yaml_plugin(
            tmp_path,
            "yaml-hooks",
            extra={"hooks": {"PreToolUse": [{"command": "echo"}]}},
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.hooks_config is not None
        assert "PreToolUse" in plugin.hooks_config

    def test_yaml_priority_over_json(self, tmp_path):
        """plugin.yaml should be preferred over plugin.json."""
        plugin_dir = tmp_path / "priority-plugin"
        plugin_dir.mkdir()
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")
        # Write both YAML and JSON
        yaml_data = {"name": "yaml-wins", "version": "2.0.0"}
        json_data = {"name": "json-loses", "version": "1.0.0"}
        (plugin_dir / "plugin.yaml").write_text(yaml.dump(yaml_data))
        (plugin_dir / "plugin.json").write_text(json.dumps(json_data))
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.name == "yaml-wins"
        assert plugin.manifest.version == "2.0.0"

    def test_yaml_with_mcp_servers(self, tmp_path):
        plugin_dir = self._make_yaml_plugin(
            tmp_path,
            "yaml-mcp",
            extra={"mcp_servers": {"server1": {"command": "node"}}},
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.mcp_servers is not None
        assert "server1" in plugin.mcp_servers

    def test_yaml_disabled(self, tmp_path):
        plugin_dir = self._make_yaml_plugin(
            tmp_path,
            "yaml-disabled",
            extra={"enabled": False},
        )
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.enabled is False

    def test_yaml_invalid(self, tmp_path):
        plugin_dir = tmp_path / "bad-yaml"
        plugin_dir.mkdir()
        import importlib.util

        if importlib.util.find_spec("yaml") is None:
            pytest.skip("PyYAML not available")
        # Write a non-dict YAML (invalid root)
        (plugin_dir / "plugin.yaml").write_text("- item1\n- item2")
        with pytest.raises(PluginError, match="YAML manifest must be a mapping"):
            load_plugin_from_directory(plugin_dir)

    def test_yaml_missing_pyyaml(self, tmp_path):
        plugin_dir = tmp_path / "no-pyyaml"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text("name: test")
        with patch.dict("sys.modules", {"yaml": None}):
            with pytest.raises(PluginError, match="PyYAML is required"):
                load_plugin_from_directory(plugin_dir)


# ── Manifest file priority tests ─────────────────────────────────────


class TestManifestFilePriority:
    def test_manifest_files_list(self):
        assert MANIFEST_FILES == (
            "plugin.yaml",
            "plugin.yml",
            "plugin.json",
            "pyproject.toml",
        )

    def test_json_only(self, tmp_path):
        plugin_dir = tmp_path / "json-only"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(json.dumps({"name": "j"}))
        plugin = load_plugin_from_directory(plugin_dir)
        assert plugin.name == "j"

    def test_no_manifest(self, tmp_path):
        plugin_dir = tmp_path / "no-manifest"
        plugin_dir.mkdir()
        with pytest.raises(PluginError, match="No manifest found"):
            load_plugin_from_directory(plugin_dir)


# ── Entry points discovery tests ─────────────────────────────────────


class TestEntryPointDiscovery:
    def test_discover_empty_when_no_entry_points(self):
        with patch("src.plugins.loader._get_entry_points", return_value=[]):
            plugins = discover_entry_point_plugins()
            assert plugins == []

    def test_discover_skips_failing_entry_points(self):
        """If an entry point loads but fails, it should be logged, not crash."""
        bad_ep = MagicMock()
        bad_ep.name = "bad-plugin"
        bad_ep.load.side_effect = ImportError("missing module")

        with patch("src.plugins.loader._get_entry_points", return_value=[bad_ep]):
            plugins = discover_entry_point_plugins()
            assert plugins == []

    def test_discover_succeeding_entry_point(self):
        good_ep = MagicMock()
        good_ep.name = "good-plugin"
        good_ep.load.return_value = lambda: LoadedPlugin(
            name="good-plugin",
            manifest=PluginManifest(name="good-plugin"),
        )

        with patch("src.plugins.loader._get_entry_points", return_value=[good_ep]):
            plugins = discover_entry_point_plugins()
            assert len(plugins) == 1
            assert plugins[0].name == "good-plugin"

    def test_discover_dict_entry_point(self):
        good_ep = MagicMock()
        good_ep.name = "dict-plugin"
        good_ep.load.return_value = lambda: {
            "name": "dict-plugin",
            "description": "From dict",
            "version": "3.0.0",
        }

        with patch("src.plugins.loader._get_entry_points", return_value=[good_ep]):
            plugins = discover_entry_point_plugins()
            assert len(plugins) == 1
            assert plugins[0].name == "dict-plugin"
            assert plugins[0].manifest.version == "3.0.0"


# ── Default plugin directories tests ─────────────────────────────────


class TestDefaultPluginDirs:
    def test_empty_when_no_dirs_exist(self, tmp_path):
        with (
            patch("src.plugins.loader._get_user_plugin_dir") as mock_user,
            patch("src.plugins.loader._get_project_plugin_dir") as mock_proj,
        ):
            mock_user.return_value = tmp_path / "nonexistent-user"
            mock_proj.return_value = tmp_path / "nonexistent-proj"
            dirs = get_default_plugin_dirs()
            assert dirs == []

    def test_user_dir_included_when_exists(self, tmp_path):
        user_dir = tmp_path / "user-plugins"
        user_dir.mkdir()
        with (
            patch("src.plugins.loader._get_user_plugin_dir", return_value=user_dir),
            patch("src.plugins.loader._get_project_plugin_dir", return_value=tmp_path / "no-proj"),
        ):
            dirs = get_default_plugin_dirs()
            assert len(dirs) == 1
            assert dirs[0] == user_dir

    def test_xdg_env_var(self, tmp_path):
        xdg_dir = tmp_path / "custom-plugins"
        xdg_dir.mkdir()
        with patch.dict("os.environ", {"CLAWCODEX_PLUGINS_DIR": str(xdg_dir)}):
            dirs = get_default_plugin_dirs()
            assert len(dirs) == 1
            assert dirs[0] == xdg_dir


# ── Discover all plugins tests ───────────────────────────────────────


class TestDiscoverAllPlugins:
    def test_discover_all_empty(self, tmp_path):
        with (
            patch("src.plugins.loader.get_default_plugin_dirs", return_value=[]),
            patch("src.plugins.loader.discover_entry_point_plugins", return_value=[]),
        ):
            result = discover_all_plugins()
            assert result.plugins == []
            assert result.errors == []

    def test_discover_all_combines(self, tmp_path):
        """discover_all should scan dirs AND entry points."""
        user_dir = tmp_path / "user-plugins"
        user_dir.mkdir()
        # Create a plugin in the user dir
        plugin_dir = user_dir / "discovered"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps({"name": "discovered", "version": "1.0.0"}),
        )

        ep_plugin = LoadedPlugin(
            name="ep-plugin",
            manifest=PluginManifest(name="ep-plugin"),
        )

        with (
            patch("src.plugins.loader.get_default_plugin_dirs", return_value=[user_dir]),
            patch("src.plugins.loader.discover_entry_point_plugins", return_value=[ep_plugin]),
        ):
            result = discover_all_plugins()
            names = {p.name for p in result.plugins}
            assert "discovered" in names
            assert "ep-plugin" in names


# ── Lifecycle management tests ───────────────────────────────────────


class TestTogglePluginEnabled:
    def test_toggle_on(self):
        plugin = LoadedPlugin(
            name="toggle-on",
            manifest=PluginManifest(name="toggle-on"),
            enabled=False,
        )
        register_plugin(plugin)
        assert toggle_plugin_enabled("toggle-on", True) is True
        assert get_loaded_plugin("toggle-on").enabled is True

    def test_toggle_off(self):
        plugin = LoadedPlugin(
            name="toggle-off",
            manifest=PluginManifest(name="toggle-off"),
            enabled=True,
        )
        register_plugin(plugin)
        assert toggle_plugin_enabled("toggle-off", False) is True
        assert get_loaded_plugin("toggle-off").enabled is False

    def test_toggle_nonexistent(self):
        assert toggle_plugin_enabled("nope", True) is False

    def test_toggle_no_change(self):
        plugin = LoadedPlugin(
            name="same",
            manifest=PluginManifest(name="same"),
            enabled=True,
        )
        register_plugin(plugin)
        # Toggle to same state — should return True (found) but not fire callbacks
        result = toggle_plugin_enabled("same", True)
        assert result is True
        assert get_loaded_plugin("same").enabled is True

    def test_toggle_affects_enabled_list(self):
        reg = LoadedPlugin(name="en", manifest=PluginManifest(name="en"), enabled=True)
        dis = LoadedPlugin(name="dis", manifest=PluginManifest(name="dis"), enabled=False)
        register_plugin(reg)
        register_plugin(dis)
        assert len(get_enabled_plugins()) == 1

        toggle_plugin_enabled("dis", True)
        assert len(get_enabled_plugins()) == 2

        toggle_plugin_enabled("en", False)
        assert len(get_enabled_plugins()) == 1
        assert get_enabled_plugins()[0].name == "dis"


class TestLifecycleCallbacks:
    def test_decorator_registers_callback(self):
        calls = []

        @on_lifecycle("decorator-plugin", "on_enable")
        def _cb(plugin):
            calls.append(plugin.name)

        plugin = LoadedPlugin(
            name="decorator-plugin",
            manifest=PluginManifest(name="decorator-plugin"),
            enabled=False,  # Start disabled so toggle enables it
        )
        register_plugin(plugin)
        toggle_plugin_enabled("decorator-plugin", True)
        assert calls == ["decorator-plugin"]

    def test_decorator_on_disable(self):
        calls = []

        @on_lifecycle("disable-cb", "on_disable")
        def _cb(plugin):
            calls.append("disabled")

        plugin = LoadedPlugin(
            name="disable-cb",
            manifest=PluginManifest(name="disable-cb"),
        )
        register_plugin(plugin)
        toggle_plugin_enabled("disable-cb", False)
        assert calls == ["disabled"]

    def test_callback_exception_handled(self):
        @on_lifecycle("failing-cb", "on_enable")
        def _cb(plugin):
            raise RuntimeError("boom!")

        plugin = LoadedPlugin(
            name="failing-cb",
            manifest=PluginManifest(name="failing-cb"),
        )
        register_plugin(plugin)
        # Should not raise — exceptions are caught and logged
        toggle_plugin_enabled("failing-cb", True)
        assert get_loaded_plugin("failing-cb").enabled is True

    def test_fire_lifecycle_event(self):
        def _cb1(plugin):
            return 1

        def _cb2(plugin):
            return 2

        on_lifecycle("fire-test", "on_enable")(_cb1)
        on_lifecycle("fire-test", "on_enable")(_cb2)

        plugin = LoadedPlugin(
            name="fire-test",
            manifest=PluginManifest(name="fire-test"),
        )
        register_plugin(plugin)
        fired = fire_lifecycle_event("fire-test", "on_enable", plugin)
        assert fired == [1, 2]

    def test_fire_nonexistent_event(self):
        assert fire_lifecycle_event("nope", "on_enable") == []

    def test_fire_nonexistent_plugin(self):
        assert fire_lifecycle_event("nope", "on_enable") == []


# ── Integration: YAML + lifecycle ────────────────────────────────────


class TestYamlLifecycleIntegration:
    def test_yaml_plugin_can_be_toggled(self, tmp_path):
        """YAML-loaded plugin should participate in lifecycle management."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")

        calls = []

        @on_lifecycle("yaml-lifecycle", "on_enable")
        def _on_enable(plugin):
            calls.append("enabled")

        plugin_dir = tmp_path / "yaml-lifecycle"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            yaml.dump(
                {
                    "name": "yaml-lifecycle",
                    "version": "1.0.0",
                    "enabled": False,
                }
            ),
        )
        plugin = load_plugin_from_directory(plugin_dir)
        register_plugin(plugin)
        assert plugin.enabled is False

        toggle_plugin_enabled("yaml-lifecycle", True)
        assert get_loaded_plugin("yaml-lifecycle").enabled is True
        assert calls == ["enabled"]
