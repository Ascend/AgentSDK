"""Tests for PluginManager (P70-C lifecycle management)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.plugins.base import BasePlugin, PluginContext
from src.plugins.loader import (
    clear_loaded_plugins,
    get_loaded_plugin,
    get_loaded_plugins,
    register_plugin,
)
from src.plugins.manager import PluginManager, create_manager
from src.plugins.sandbox import (
    SandboxConfig,
    SandboxMode,
    clear_sandboxes,
    get_all_sandboxes,
    get_sandbox,
)
from src.plugins.types import LoadedPlugin, PluginManifest


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded_plugins()
    clear_sandboxes()
    yield
    clear_sandboxes()
    clear_loaded_plugins()


def _make_plugin_dir(base: Path, name: str, **extra) -> Path:
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "description": f"Plugin {name}", "version": "1.0.0"}
    manifest.update(extra)
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest))
    return plugin_dir


def _make_plugin(name: str = "test-plugin", **kwargs) -> LoadedPlugin:
    return LoadedPlugin(
        name=name,
        manifest=PluginManifest(name=name),
        path=kwargs.get("path", "/tmp/test-plugin"),
        source=kwargs.get("source", "user"),
        enabled=kwargs.get("enabled", True),
    )


# ── PluginManager instantiation ────────────────────────────────────────


class TestPluginManagerInstantiation:
    def test_create_manager_defaults(self):
        mgr = create_manager()
        assert mgr.auto_discover is True
        assert mgr.sandbox_enabled is False
        assert mgr.extra_dirs == []

    def test_create_manager_with_options(self):
        mgr = create_manager(auto_discover=False, sandbox_enabled=True, extra_dirs=["/tmp"])
        assert mgr.auto_discover is False
        assert mgr.sandbox_enabled is True
        assert mgr.extra_dirs == ["/tmp"]

    def test_manager_no_auto_discover(self):
        mgr = PluginManager(auto_discover=False)
        assert mgr._instances == {}


# ── Discovery ────────────────────────────────────────────────────────


class TestPluginManagerDiscovery:
    def test_discover_directory(self, tmp_path):
        _make_plugin_dir(tmp_path, "plugin-a")
        _make_plugin_dir(tmp_path, "plugin-b")
        mgr = PluginManager(auto_discover=False)
        result = mgr.discover_directory(tmp_path)
        assert len(result.plugins) == 2
        names = {p.name for p in result.plugins}
        assert names == {"plugin-a", "plugin-b"}

    def test_discover_directory_recursive(self, tmp_path):
        nested = tmp_path / "nested" / "plugin-deep"
        nested.mkdir(parents=True)
        (nested / "plugin.json").write_text(json.dumps({"name": "deep", "version": "1.0.0"}))
        mgr = PluginManager(auto_discover=False)
        result = mgr.discover_directory(tmp_path, recursive=True)
        assert len(result.plugins) == 1
        assert result.plugins[0].name == "deep"

    def test_discover_entry_points(self):
        mgr = PluginManager(auto_discover=False)
        with patch("src.plugins.manager.discover_entry_point_plugins", return_value=[]):
            plugins = mgr.discover_entry_points()
            assert plugins == []

    def test_discover_all(self, tmp_path):
        user_dir = tmp_path / "user-plugins"
        user_dir.mkdir()
        _make_plugin_dir(user_dir, "user-plugin")
        mgr = PluginManager(auto_discover=False)
        with patch("src.plugins.manager.discover_all_plugins") as mock_discover:
            mock_discover.return_value = MagicMock(plugins=[], errors=[])
            result = mgr.discover()
            mock_discover.assert_called_once_with(extra_dirs=[], recursive=False)
            assert result is not None

    def test_discover_all_with_extra_dirs(self, tmp_path):
        extra = tmp_path / "extra"
        extra.mkdir()
        _make_plugin_dir(extra, "extra-plugin")
        mgr = PluginManager(auto_discover=False, extra_dirs=[str(extra)])
        result = mgr.discover()
        assert len(result.plugins) == 1
        assert result.plugins[0].name == "extra-plugin"


# ── Lifecycle: enable / disable ──────────────────────────────────────


class TestPluginManagerEnableDisable:
    def test_enable_plugin(self):
        plugin = _make_plugin("enable-me", enabled=False)
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False)
        assert mgr.enable_plugin("enable-me") is True
        assert get_loaded_plugin("enable-me").enabled is True

    def test_disable_plugin(self):
        plugin = _make_plugin("disable-me", enabled=True)
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False)
        assert mgr.disable_plugin("disable-me") is True
        assert get_loaded_plugin("disable-me").enabled is False

    def test_disable_unloads_first(self):
        """Disabling a loaded plugin should unload it first."""
        plugin = _make_plugin("loaded-disable")
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False)
        # Simulate instance tracking
        mgr._instances["loaded-disable"] = MagicMock()
        mgr.disable_plugin("loaded-disable")
        assert "loaded-disable" not in mgr._instances
        assert get_loaded_plugin("loaded-disable").enabled is False

    def test_enable_nonexistent(self):
        mgr = PluginManager(auto_discover=False)
        assert mgr.enable_plugin("nope") is False

    def test_disable_nonexistent(self):
        mgr = PluginManager(auto_discover=False)
        assert mgr.disable_plugin("nope") is False


# ── Install / Uninstall ──────────────────────────────────────────────


class TestPluginManagerInstallUninstall:
    def test_install_plugin(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        _make_plugin_dir(source, "my-plugin")
        target = tmp_path / "target"
        target.mkdir()

        mgr = PluginManager(auto_discover=False)
        with patch("src.plugins.manager.ensure_plugin_dirs", return_value=[target]):
            plugin = mgr.install_plugin(source, "my-plugin")
            assert plugin.name == "my-plugin"
            assert (target / "my-plugin" / "plugin.json").exists()

    def test_uninstall_plugin(self, tmp_path):
        target = tmp_path / "plugins"
        target.mkdir()
        _make_plugin_dir(target, "remove-me")
        register_plugin(_make_plugin("remove-me", path=str(target / "remove-me")))

        mgr = PluginManager(auto_discover=False)
        with patch("src.plugins.manager.ensure_plugin_dirs", return_value=[target]):
            result = mgr.uninstall_plugin("remove-me")
            assert result is True
            assert not (target / "remove-me").exists()
            assert get_loaded_plugin("remove-me") is None

    def test_uninstall_nonexistent(self, tmp_path):
        target = tmp_path / "plugins"
        target.mkdir()
        mgr = PluginManager(auto_discover=False)
        with patch("src.plugins.manager.ensure_plugin_dirs", return_value=[target]):
            result = mgr.uninstall_plugin("nope")
            assert result is False


# ── Status queries ───────────────────────────────────────────────────


class TestPluginManagerStatusQueries:
    def test_list_plugins(self):
        register_plugin(_make_plugin("p1", enabled=True))
        register_plugin(_make_plugin("p2", enabled=False))
        mgr = PluginManager(auto_discover=False)
        all_plugins = mgr.list_plugins()
        assert len(all_plugins) == 2
        enabled = mgr.list_plugins(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0].name == "p1"

    def test_get_plugin_status(self):
        plugin = _make_plugin("status-test", source="user")
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False)
        status = mgr.get_plugin_status("status-test")
        assert status is not None
        assert status["name"] == "status-test"
        assert status["enabled"] is True
        assert status["source"] == "user"
        assert status["loaded"] is False
        assert status["has_sandbox"] is False

    def test_get_plugin_status_nonexistent(self):
        mgr = PluginManager(auto_discover=False)
        assert mgr.get_plugin_status("nope") is None


# ── Sandbox integration ──────────────────────────────────────────────


class TestPluginManagerSandbox:
    def test_setup_sandbox_when_enabled(self):
        plugin = _make_plugin("sandboxed", source="marketplace")
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False, sandbox_enabled=True)
        mgr.enable_plugin("sandboxed")
        assert get_sandbox("sandboxed") is not None

    def test_no_sandbox_when_disabled(self):
        plugin = _make_plugin("no-sandbox", source="marketplace")
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False, sandbox_enabled=False)
        mgr.enable_plugin("no-sandbox")
        assert get_sandbox("no-sandbox") is None

    def test_sandbox_config_override(self):
        plugin = _make_plugin("custom-sandbox")
        register_plugin(plugin)
        custom_cfg = SandboxConfig(mode=SandboxMode.PROCESS, timeout_seconds=5.0)
        mgr = PluginManager(
            auto_discover=False,
            sandbox_enabled=True,
            sandbox_configs={"custom-sandbox": custom_cfg},
        )
        mgr.enable_plugin("custom-sandbox")
        sb = get_sandbox("custom-sandbox")
        assert sb is not None
        assert sb.config.timeout_seconds == 5.0


# ── Unload / shutdown ────────────────────────────────────────────────


class TestPluginManagerUnload:
    def test_unload_all(self):
        register_plugin(_make_plugin("p1"))
        register_plugin(_make_plugin("p2"))
        mgr = PluginManager(auto_discover=False)
        mgr._instances["p1"] = MagicMock()
        mgr._instances["p2"] = MagicMock()
        mgr.unload_all()
        assert mgr._instances == {}
        assert get_loaded_plugins() == []
        assert get_all_sandboxes() == []

    def test_shutdown(self):
        register_plugin(_make_plugin("p1"))
        mgr = PluginManager(auto_discover=False)
        mgr._instances["p1"] = MagicMock()
        mgr.shutdown()
        assert mgr._instances == {}


# ── BasePlugin lifecycle hooks integration ───────────────────────────


class _DummyPlugin(BasePlugin):
    name = "dummy"
    version = "1.0.0"
    description = "A dummy plugin for testing"

    def __init__(self):
        self.loaded = False
        self.unloaded = False

    async def on_load(self, context: PluginContext) -> None:
        self.loaded = True

    async def on_unload(self) -> None:
        self.unloaded = True


class TestPluginManagerBasePluginLifecycle:
    def test_load_plugin_with_baseplugin(self, tmp_path):
        """Test that a plugin with a BasePlugin subclass can be loaded."""
        plugin_dir = tmp_path / "baseplugin-test"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(
            "from src.plugins.base import BasePlugin, PluginContext\n"
            "class TestPlugin(BasePlugin):\n"
            "    name = 'baseplugin-test'\n"
            "    version = '1.0.0'\n"
            "    async def on_load(self, ctx: PluginContext) -> None:\n"
            "        pass\n"
            "    async def on_unload(self) -> None:\n"
            "        pass\n"
        )
        (plugin_dir / "plugin.json").write_text(json.dumps({"name": "baseplugin-test", "version": "1.0.0"}))

        plugin = _make_plugin("baseplugin-test", path=str(plugin_dir))
        register_plugin(plugin)

        mgr = PluginManager(auto_discover=False)
        instance = mgr.load_plugin(plugin)
        assert instance is not None
        assert instance.name == "baseplugin-test"
        assert "baseplugin-test" in mgr._instances

    def test_unload_plugin_calls_on_unload(self):
        plugin = _make_plugin("unload-test")
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False)
        dummy = _DummyPlugin()
        mgr._instances["unload-test"] = dummy
        mgr.unload_plugin("unload-test")
        assert dummy.unloaded is True

    def test_load_all_skips_disabled(self):
        register_plugin(_make_plugin("enabled-p", enabled=True))
        register_plugin(_make_plugin("disabled-p", enabled=False))
        mgr = PluginManager(auto_discover=False)
        loaded = mgr.load_all()
        # Only enabled plugins should be loaded
        assert "disabled-p" not in loaded

    def test_unload_nonexistent(self):
        mgr = PluginManager(auto_discover=False)
        assert mgr.unload_plugin("nope") is False
