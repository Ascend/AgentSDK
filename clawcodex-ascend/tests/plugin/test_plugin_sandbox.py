"""Tests for plugin sandbox (subprocess isolation)."""

from __future__ import annotations


import pytest

from src.plugins.sandbox import (
    ResourceLimit,
    SandboxConfig,
    SandboxMode,
    clear_sandboxes,
    execute_in_sandbox,
    execute_rpc,
    get_all_sandboxes,
    get_sandbox,
    register_sandbox,
    remove_sandbox,
    start_sandbox,
    stop_sandbox,
)
from src.plugins.loader import clear_loaded_plugins, register_plugin
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
        enabled=True,
    )


# ── SandboxConfig / SandboxMode ──────────────────────────────────────


class TestSandboxConfig:
    def test_default_config(self):
        cfg = SandboxConfig()
        assert cfg.mode == SandboxMode.NONE
        assert cfg.network_allowed is True
        assert cfg.timeout_seconds == 30.0

    def test_process_config(self):
        cfg = SandboxConfig(mode=SandboxMode.PROCESS, network_allowed=False)
        assert cfg.mode == SandboxMode.PROCESS
        assert cfg.network_allowed is False

    def test_resource_limits(self):
        cfg = SandboxConfig(
            mode=SandboxMode.PROCESS,
            resource_limits={
                ResourceLimit.CPU_SECONDS: 10,
                ResourceLimit.MEMORY_BYTES: 1024 * 1024 * 100,  # 100 MB
            },
        )
        assert cfg.resource_limits[ResourceLimit.CPU_SECONDS] == 10


# ── Sandbox registry ─────────────────────────────────────────────────


class TestSandboxRegistry:
    def test_register_and_get(self):
        plugin = _make_plugin()
        sb = register_sandbox(plugin)
        assert get_sandbox("test-plugin") is sb
        assert sb.plugin.name == "test-plugin"

    def test_get_all(self):
        p1 = _make_plugin("p1")
        p2 = _make_plugin("p2")
        register_sandbox(p1)
        register_sandbox(p2)
        sandboxes = get_all_sandboxes()
        assert len(sandboxes) == 2
        names = {sb.plugin.name for sb in sandboxes}
        assert names == {"p1", "p2"}

    def test_remove(self):
        plugin = _make_plugin()
        register_sandbox(plugin)
        assert remove_sandbox("test-plugin") is True
        assert get_sandbox("test-plugin") is None

    def test_remove_nonexistent(self):
        assert remove_sandbox("nope") is False

    def test_clear(self):
        register_sandbox(_make_plugin("p1"))
        register_sandbox(_make_plugin("p2"))
        clear_sandboxes()
        assert get_all_sandboxes() == []


# ── SandboxMode.NONE ─────────────────────────────────────────────────


class TestNoopSandbox:
    def test_start_none_mode(self):
        plugin = _make_plugin()
        cfg = SandboxConfig(mode=SandboxMode.NONE)
        sb = register_sandbox(plugin, cfg)
        # SandboxMode.NONE should succeed without doing anything
        assert start_sandbox(sb) is True

    def test_execute_in_none_mode_allows_all(self):
        plugin = _make_plugin()
        sb = register_sandbox(plugin)
        result = execute_in_sandbox(sb, ["echo", "hello"])
        # In NONE mode, all permissions are allowed
        # The command will actually run in the host process
        assert result.exit_code == 0 or result.exit_code == 127  # echo exists or not


# ── Permission checks ────────────────────────────────────────────────


class TestPermissionChecks:
    def test_denied_network_operation(self):
        plugin = _make_plugin("restricted", source="marketplace")
        cfg = SandboxConfig(
            mode=SandboxMode.PROCESS,
            allowed_permissions={"read", "execute"},
            network_allowed=False,
        )
        sb = register_sandbox(plugin, cfg)
        result = execute_in_sandbox(sb, ["curl", "http://example.com"])
        assert result.exit_code == 1
        assert "Permission denied" in result.error or "Network" in result.error

    def test_denied_unknown_op(self):
        plugin = _make_plugin("limited", source="marketplace")
        cfg = SandboxConfig(
            mode=SandboxMode.PROCESS,
            allowed_permissions={"read"},
            network_allowed=False,
        )
        sb = register_sandbox(plugin, cfg)
        result = execute_in_sandbox(sb, ["unknown-cmd"])
        assert result.exit_code == 1


# ── Timeout handling ─────────────────────────────────────────────────


class TestTimeoutHandling:
    def test_timeout_returns_timed_out_flag(self):
        plugin = _make_plugin("slow")
        cfg = SandboxConfig(
            mode=SandboxMode.PROCESS,
            timeout_seconds=0.01,
            allowed_permissions={"read", "write", "execute", "network"},
        )
        sb = register_sandbox(plugin, cfg)
        # Sleep 1 second — should timeout with our 0.01s limit
        result = execute_in_sandbox(sb, ["sleep", "1"], timeout=0.01)
        assert result.timed_out is True
        assert result.exit_code == -1


# ── Error handling ───────────────────────────────────────────────────


class TestErrorHandling:
    def test_nonexistent_command(self):
        plugin = _make_plugin()
        sb = register_sandbox(plugin)
        result = execute_in_sandbox(sb, ["definitely-not-a-real-command-xyz"])
        assert result.exit_code != 0

    def test_execute_rpc_no_process(self):
        plugin = _make_plugin()
        sb = register_sandbox(plugin, SandboxConfig(mode=SandboxMode.NONE))
        # No process started
        result = execute_rpc(sb, "some_method")
        assert result is None

    def test_stop_no_process(self):
        plugin = _make_plugin()
        sb = register_sandbox(plugin)
        # Should not raise
        stop_sandbox(sb)


# ── Inferred sandbox config ──────────────────────────────────────────


class TestInferredConfig:
    def test_marketplace_source_restricted(self):
        plugin = _make_plugin(source="marketplace")
        register_plugin(plugin)
        from src.plugins.sandbox import _infer_sandbox_config

        cfg = _infer_sandbox_config(plugin)
        assert cfg.mode != SandboxMode.NONE  # marketplace gets PROCESS mode
        assert "network" not in cfg.allowed_permissions

    def test_entry_point_source_full_access(self):
        plugin = _make_plugin(source="entry_point")
        register_plugin(plugin)
        from src.plugins.sandbox import _infer_sandbox_config

        cfg = _infer_sandbox_config(plugin)
        assert cfg.mode != SandboxMode.NONE
        assert "read" in cfg.allowed_permissions
        assert "write" in cfg.allowed_permissions
        assert "execute" in cfg.allowed_permissions
