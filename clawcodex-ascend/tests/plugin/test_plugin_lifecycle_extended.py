"""Tests for plugin lifecycle hooks (on_enable / on_disable / on_load / on_unload)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.plugins.base import BasePlugin, PluginContext
from src.plugins.loader import (
    clear_loaded_plugins,
    fire_lifecycle_event,
    get_loaded_plugin,
    on_lifecycle,
    register_plugin,
    toggle_plugin_enabled,
)
from src.plugins.manager import PluginManager
from src.plugins.sandbox import clear_sandboxes
from src.plugins.types import LoadedPlugin, PluginManifest


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded_plugins()
    clear_sandboxes()
    yield
    clear_sandboxes()
    clear_loaded_plugins()


def _make_plugin(name: str = "test-plugin", **kwargs) -> LoadedPlugin:
    return LoadedPlugin(
        name=name,
        manifest=PluginManifest(name=name),
        path=kwargs.get("path", "/tmp/test-plugin"),
        source=kwargs.get("source", "user"),
        enabled=kwargs.get("enabled", True),
    )


class _LifecycleTrackingPlugin(BasePlugin):
    name = "tracker"
    version = "1.0.0"

    def __init__(self):
        self.events: list[str] = []
        self.load_ctx: PluginContext | None = None

    async def on_load(self, context: PluginContext) -> None:
        self.events.append("on_load")
        self.load_ctx = context

    async def on_unload(self) -> None:
        self.events.append("on_unload")

    async def on_enable(self) -> None:
        self.events.append("on_enable")

    async def on_disable(self) -> None:
        self.events.append("on_disable")


# ── toggle_plugin_enabled lifecycle ──────────────────────────────────────


class TestToggleLifecycle:
    def test_enable_fires_on_enable(self):
        plugin = _make_plugin("p1", enabled=False)
        register_plugin(plugin)
        calls: list[str] = []

        @on_lifecycle("p1", "on_enable")
        def _cb(plugin):
            calls.append("enabled")

        result = toggle_plugin_enabled("p1", True)
        assert result is True
        assert calls == ["enabled"]
        assert get_loaded_plugin("p1").enabled is True

    def test_disable_fires_on_disable(self):
        plugin = _make_plugin("p1", enabled=True)
        register_plugin(plugin)
        calls: list[str] = []

        @on_lifecycle("p1", "on_disable")
        def _cb(plugin):
            calls.append("disabled")

        result = toggle_plugin_enabled("p1", False)
        assert result is True
        assert calls == ["disabled"]
        assert get_loaded_plugin("p1").enabled is False

    def test_no_event_when_state_unchanged(self):
        plugin = _make_plugin("p1", enabled=True)
        register_plugin(plugin)
        calls: list[str] = []

        @on_lifecycle("p1", "on_enable")
        def _cb(plugin):
            calls.append("enabled")

        # Already enabled — no transition
        result = toggle_plugin_enabled("p1", True)
        assert result is True
        assert calls == []

    def test_fire_lifecycle_event_manual(self):
        plugin = _make_plugin("p1")
        register_plugin(plugin)
        calls: list[str] = []

        @on_lifecycle("p1", "custom_event")
        def _cb(arg):
            calls.append(arg)
            return arg

        results = fire_lifecycle_event("p1", "custom_event", "hello")
        assert results == ["hello"]
        assert calls == ["hello"]

    def test_fire_lifecycle_event_exception_handled(self):
        plugin = _make_plugin("p1")
        register_plugin(plugin)

        @on_lifecycle("p1", "bad_event")
        def _cb(arg):
            raise ValueError("boom")

        # Should not raise; exception is logged
        results = fire_lifecycle_event("p1", "bad_event", "x")
        assert results == []


# ── BasePlugin lifecycle hooks via PluginManager ───────────────────────


class TestBasePluginLifecycleViaManager:
    def test_manager_enable_calls_on_enable(self):
        plugin = _make_plugin("enable-hook", enabled=False)
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False)

        calls: list[str] = []

        @on_lifecycle("enable-hook", "on_enable")
        def _cb(plugin):
            calls.append("on_enable")

        mgr.enable_plugin("enable-hook")
        assert calls == ["on_enable"]

    def test_manager_disable_calls_on_disable(self):
        plugin = _make_plugin("disable-hook", enabled=True)
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False)

        calls: list[str] = []

        @on_lifecycle("disable-hook", "on_disable")
        def _cb(plugin):
            calls.append("on_disable")

        mgr.disable_plugin("disable-hook")
        assert calls == ["on_disable"]

    def test_load_plugin_async_in_sync_context(self, tmp_path):
        """Verify that load_plugin can run async on_load in a sync context."""
        plugin_dir = tmp_path / "async-load"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(
            "from src.plugins.base import BasePlugin, PluginContext\n"
            "class AsyncPlugin(BasePlugin):\n"
            "    name = 'async-load'\n"
            "    version = '1.0.0'\n"
            "    async def on_load(self, ctx: PluginContext) -> None:\n"
            "        pass\n"
            "    async def on_unload(self) -> None:\n"
            "        pass\n"
        )
        (plugin_dir / "plugin.json").write_text('{"name": "async-load", "version": "1.0.0"}')

        plugin = _make_plugin("async-load", path=str(plugin_dir))
        register_plugin(plugin)

        mgr = PluginManager(auto_discover=False)
        instance = mgr.load_plugin(plugin)
        assert instance is not None
        assert instance.name == "async-load"

    def test_unload_plugin_async_in_sync_context(self):
        plugin = _make_plugin("async-unload")
        register_plugin(plugin)
        mgr = PluginManager(auto_discover=False)
        dummy = _LifecycleTrackingPlugin()
        mgr._instances["async-unload"] = dummy

        mgr.unload_plugin("async-unload")
        assert "on_unload" in dummy.events


# ── PluginContext data_dir helpers ─────────────────────────────────────


class TestPluginContext:
    def test_get_data_file(self, tmp_path):
        ctx = PluginContext(registry=None, data_dir=tmp_path)
        path = ctx.get_data_file("state.json")
        assert path == tmp_path / "state.json"

    def test_get_data_file_without_data_dir_raises(self):
        ctx = PluginContext(registry=None, data_dir=None)
        with pytest.raises(RuntimeError):
            ctx.get_data_file("state.json")

    def test_plugin_context_defaults(self):
        ctx = PluginContext(registry=None)
        assert ctx.config == {}
        assert ctx.data_dir is None
        assert ctx.command_system is None


# ── Sandbox health checks ────────────────────────────────────────────────


class TestSandboxHealthChecks:
    def test_health_check_none_mode(self):
        from src.plugins.sandbox import (
            SandboxedPlugin,
            SandboxConfig,
            SandboxMode,
            health_check_sandbox,
        )

        plugin = _make_plugin("health-none")
        sb = SandboxedPlugin(
            plugin=plugin,
            config=SandboxConfig(mode=SandboxMode.NONE),
        )
        status = health_check_sandbox(sb)
        assert status["alive"] is True
        assert status["pid"] is None
        assert status["uptime"] is None
        assert status["timed_out"] is False

    def test_ping_none_mode(self):
        from src.plugins.sandbox import SandboxedPlugin, SandboxConfig, SandboxMode, ping_sandbox

        plugin = _make_plugin("ping-none")
        sb = SandboxedPlugin(
            plugin=plugin,
            config=SandboxConfig(mode=SandboxMode.NONE),
        )
        assert ping_sandbox(sb) is True

    def test_ping_no_process(self):
        from src.plugins.sandbox import SandboxedPlugin, SandboxConfig, SandboxMode, ping_sandbox

        plugin = _make_plugin("ping-noproc")
        sb = SandboxedPlugin(
            plugin=plugin,
            config=SandboxConfig(mode=SandboxMode.PROCESS),
        )
        assert ping_sandbox(sb) is False


# ── Sandbox permission inference ─────────────────────────────────────────


class TestSandboxInferredConfigExtended:
    def test_builtin_source_trusted(self):
        from src.plugins.sandbox import _infer_sandbox_config, SandboxMode

        plugin = _make_plugin(source="builtin")
        cfg = _infer_sandbox_config(plugin)
        assert cfg.mode == SandboxMode.NONE
        assert "network" in cfg.allowed_permissions
        assert "mcp" in cfg.allowed_permissions

    def test_user_source_restricted(self):
        from src.plugins.sandbox import _infer_sandbox_config, SandboxMode

        plugin = _make_plugin(source="user")
        cfg = _infer_sandbox_config(plugin)
        # Unknown source defaults to PROCESS with minimal permissions
        assert cfg.mode == SandboxMode.PROCESS
        assert "network" not in cfg.allowed_permissions

    def test_marketplace_no_network(self):
        from src.plugins.sandbox import _infer_sandbox_config

        plugin = _make_plugin(source="marketplace")
        cfg = _infer_sandbox_config(plugin)
        assert cfg.network_allowed is False

    def test_entry_point_has_network(self):
        from src.plugins.sandbox import _infer_sandbox_config

        plugin = _make_plugin(source="entry_point")
        cfg = _infer_sandbox_config(plugin)
        assert cfg.network_allowed is True


# ── Sandbox execute_in_sandbox edge cases ────────────────────────────────


class TestSandboxExecuteEdgeCases:
    def test_execute_empty_command(self):
        from src.plugins.sandbox import SandboxedPlugin, SandboxConfig, execute_in_sandbox

        plugin = _make_plugin("empty-cmd")
        sb = SandboxedPlugin(plugin=plugin, config=SandboxConfig())
        result = execute_in_sandbox(sb, [])
        assert result.exit_code == 1
        assert "Permission denied" in result.error

    def test_execute_allowed_network(self):
        from src.plugins.sandbox import (
            SandboxedPlugin,
            SandboxConfig,
            SandboxMode,
            execute_in_sandbox,
        )

        plugin = _make_plugin("net-ok")
        cfg = SandboxConfig(
            mode=SandboxMode.PROCESS,
            allowed_permissions={"network"},
            network_allowed=True,
        )
        sb = SandboxedPlugin(plugin=plugin, config=cfg)
        result = execute_in_sandbox(sb, ["curl", "http://example.com"])
        # curl may not exist, but permission should be granted
        assert result.error is None or "Permission denied" not in result.error
        assert result.error is None or "Network" not in result.error

    def test_execute_network_denied(self):
        from src.plugins.sandbox import (
            SandboxedPlugin,
            SandboxConfig,
            SandboxMode,
            execute_in_sandbox,
        )

        plugin = _make_plugin("net-denied")
        cfg = SandboxConfig(
            mode=SandboxMode.PROCESS,
            allowed_permissions={"read"},
            network_allowed=False,
        )
        sb = SandboxedPlugin(plugin=plugin, config=cfg)
        result = execute_in_sandbox(sb, ["curl", "http://example.com"])
        assert result.exit_code == 1
        assert "Network access is disabled" in result.error

    def test_execute_read_operation(self):
        from src.plugins.sandbox import (
            SandboxedPlugin,
            SandboxConfig,
            SandboxMode,
            execute_in_sandbox,
        )

        plugin = _make_plugin("read-ok")
        cfg = SandboxConfig(
            mode=SandboxMode.PROCESS,
            allowed_permissions={"read"},
        )
        sb = SandboxedPlugin(plugin=plugin, config=cfg)
        result = execute_in_sandbox(sb, ["cat", "/etc/hostname"])
        # Permission granted; actual command may fail if cat unavailable
        assert result.error is None or "Permission denied" not in result.error


# ── PluginManager shutdown / cleanup ─────────────────────────────────────


class TestPluginManagerCleanup:
    def test_shutdown_unloads_and_clears(self):
        register_plugin(_make_plugin("p1"))
        register_plugin(_make_plugin("p2"))
        mgr = PluginManager(auto_discover=False)
        mgr._instances["p1"] = MagicMock()
        mgr._instances["p2"] = MagicMock()

        mgr.shutdown()
        assert mgr._instances == {}

    def test_unload_all_clears_registry_and_sandboxes(self):
        from src.plugins.sandbox import register_sandbox

        register_plugin(_make_plugin("p1"))
        register_sandbox(_make_plugin("p1"))
        mgr = PluginManager(auto_discover=False)
        mgr._instances["p1"] = MagicMock()

        mgr.unload_all()
        assert mgr._instances == {}
        from src.plugins.sandbox import get_all_sandboxes

        assert get_all_sandboxes() == []

    def test_manager_str_repr(self):
        mgr = PluginManager(auto_discover=False)
        assert "PluginManager" in repr(mgr)


# ── Loader lifecycle callback registration ───────────────────────────────


class TestLoaderLifecycleCallbacks:
    def test_register_plugin_creates_lifecycle_callbacks(self):
        plugin = _make_plugin("callback-test")
        register_plugin(plugin)
        # Internal: _lifecycle_callbacks should have entry after register
        from src.plugins.loader import _lifecycle_callbacks

        assert "callback-test" in _lifecycle_callbacks
        assert "on_load" in _lifecycle_callbacks["callback-test"]
        assert "on_unload" in _lifecycle_callbacks["callback-test"]
        assert "on_enable" in _lifecycle_callbacks["callback-test"]
        assert "on_disable" in _lifecycle_callbacks["callback-test"]

    def test_unregister_plugin_removes_lifecycle_callbacks(self):
        plugin = _make_plugin("callback-rm")
        register_plugin(plugin)
        from src.plugins.loader import unregister_plugin, _lifecycle_callbacks

        unregister_plugin("callback-rm")
        assert "callback-rm" not in _lifecycle_callbacks

    def test_clear_loaded_plugins_clears_callbacks(self):
        plugin = _make_plugin("callback-clear")
        register_plugin(plugin)
        from src.plugins.loader import clear_loaded_plugins, _lifecycle_callbacks

        clear_loaded_plugins()
        assert "callback-clear" not in _lifecycle_callbacks
