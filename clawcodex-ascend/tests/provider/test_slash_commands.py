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

from __future__ import annotations

from pathlib import Path
import threading
import time
from types import SimpleNamespace

from clawcodex_ext.cli.runtime_commands import register_runtime_commands
from src.command_system.builtins import execute_command_sync, register_builtin_commands
from src.command_system.engine import create_command_context
from src.command_system.registry import CommandRegistry


class Runtime:
    def __init__(self) -> None:
        self.provider_name = "anthropic"
        self.provider = SimpleNamespace(model="claude-sonnet-4-6")
        self.options = SimpleNamespace(model="claude-sonnet-4-6")
        self.tool_registry = object()
        self.tool_context = SimpleNamespace()
        self.swaps: list[tuple[str, str | None]] = []

    def swap_provider(self, provider: str, model: str | None = None) -> None:
        self.swaps.append((provider, model))
        self.provider_name = provider
        self.provider = SimpleNamespace(model=model or "zai/glm-5")
        self.options.model = self.provider.model


def _context(tmp_path: Path, runtime: Runtime):
    return create_command_context(
        workspace_root=tmp_path,
        conversation=SimpleNamespace(messages=[]),
        provider=runtime.provider,
        tool_registry=runtime.tool_registry,
        tool_context=runtime.tool_context,
        runtime_context=runtime,
    )


def _patch_store(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.cli.runtime_commands.ModelStore.set_default_provider",
        lambda self, provider: None,
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.runtime_commands.ModelStore.set_default_model",
        lambda self, provider, model: None,
    )


def test_runtime_provider_command_switches_runtime(monkeypatch, tmp_path: Path) -> None:
    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    _patch_store(monkeypatch)
    runtime = Runtime()

    success, text, error = execute_command_sync("provider", "glm", _context(tmp_path, runtime))

    assert success is True
    assert error is None
    assert runtime.swaps == [("zai", None)]
    assert "zai" in text


def test_runtime_model_command_switches_runtime(monkeypatch, tmp_path: Path) -> None:
    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    _patch_store(monkeypatch)
    runtime = Runtime()

    success, text, error = execute_command_sync(
        "model",
        "zai/glm-4 --provider glm",
        _context(tmp_path, runtime),
    )

    assert success is True
    assert error is None
    assert runtime.swaps == [("zai", "zai/glm-4")]
    assert "zai" in text
    assert "zai/glm-4" in text


def test_runtime_provider_no_args_shows_current_and_list(monkeypatch, tmp_path: Path) -> None:
    """``/provider`` (no args) shows current + available providers, no swap."""
    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    monkeypatch.setattr(
        "clawcodex_ext.cli.runtime_commands.format_provider_list",
        lambda: "Providers:\n  anthropic\tAnthropic\tconfigured=yes",
    )
    runtime = Runtime()

    success, text, error = execute_command_sync("provider", "", _context(tmp_path, runtime))

    assert success is True
    assert error is None
    assert runtime.swaps == []
    assert "anthropic" in text
    assert "Providers:" in text
    assert "anthropic" in text


def test_runtime_model_no_args_shows_current_and_list(monkeypatch, tmp_path: Path) -> None:
    """``/model`` lists models reported by the active runtime provider."""
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    runtime = Runtime()
    runtime.provider = SimpleNamespace(
        model="claude-sonnet-4-6",
        discover_available_models=lambda: ["claude-live-account-model"],
        get_available_models=lambda: ["claude-stale-configured-model"],
    )

    success, text, error = execute_command_sync("model", "", _context(tmp_path, runtime))

    assert success is True
    assert error is None
    assert runtime.swaps == []
    assert "anthropic" in text
    assert "claude-sonnet-4-6" in text
    assert "Models:" in text
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        _, refreshed, _ = execute_command_sync("model", "", _context(tmp_path, runtime))
        if "claude-live-account-model" in refreshed:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("background model refresh did not publish its result")


def test_runtime_model_shows_cached_fallback_without_waiting_for_refresh(monkeypatch, tmp_path: Path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    started = threading.Event()
    release = threading.Event()

    def discover():
        started.set()
        release.wait(timeout=2)
        return ["live-account-model"]

    runtime = Runtime()
    runtime.provider = SimpleNamespace(
        model="fallback-model",
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=discover,
    )

    before = time.perf_counter()
    success, text, error = execute_command_sync("model", "", _context(tmp_path, runtime))
    elapsed = time.perf_counter() - before

    assert success is True
    assert error is None
    assert elapsed < 0.25
    assert "fallback-model" in text
    assert "refresh" in text.lower()
    assert started.wait(timeout=0.5)

    release.set()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        _, refreshed, _ = execute_command_sync("model", "", _context(tmp_path, runtime))
        if "live-account-model" in refreshed:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("background model refresh did not publish its result")


def test_interactive_model_catalog_uses_cached_fallback_without_waiting(monkeypatch, tmp_path: Path) -> None:
    from clawcodex_ext.command_system.model_command import _list_models
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    started = threading.Event()
    release = threading.Event()

    def discover():
        started.set()
        release.wait(timeout=1)
        return ["live-account-model"]

    provider = SimpleNamespace(
        model="fallback-model",
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=discover,
    )

    before = time.perf_counter()
    models = _list_models(provider)
    elapsed = time.perf_counter() - before

    assert elapsed < 0.25
    assert models == ["fallback-model"]
    assert started.wait(timeout=0.5)
    release.set()


def test_runtime_model_switch_does_not_wait_for_registry_discovery_hook(monkeypatch, tmp_path: Path) -> None:
    import clawcodex_ext.cli.model_cmd.registry as registry_module

    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    _patch_store(monkeypatch)
    hook_called = threading.Event()

    def slow_hook():
        hook_called.set()
        time.sleep(0.3)
        return ["new-account-model"]

    monkeypatch.setitem(registry_module._DISCOVERY_HOOKS, "anthropic", [slow_hook])
    runtime = Runtime()

    before = time.perf_counter()
    success, text, error = execute_command_sync(
        "model",
        "new-account-model",
        _context(tmp_path, runtime),
    )
    elapsed = time.perf_counter() - before

    assert success is True
    assert error is None
    assert elapsed < 0.25
    assert runtime.swaps == [("anthropic", "new-account-model")]
    assert hook_called.is_set() is False


def test_runtime_model_discovery_failure_labels_fallback_and_keeps_current(monkeypatch, tmp_path: Path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    attempted = threading.Event()

    def fail_discovery():
        attempted.set()
        raise RuntimeError("TLS EOF")

    runtime = Runtime()
    runtime.provider_name = "openai-codex"
    runtime.provider = SimpleNamespace(
        model="gpt-current-live-model",
        get_available_models=lambda: ["gpt-fallback"],
        discover_available_models=fail_discovery,
    )

    success, _, error = execute_command_sync("model", "", _context(tmp_path, runtime))
    assert attempted.wait(timeout=0.5)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        _, text, _ = execute_command_sync("model", "", _context(tmp_path, runtime))
        if "Last model catalog refresh failed" in text:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("catalog refresh failure was not published")

    assert success is True
    assert error is None
    assert "TLS EOF" in text
    assert "showing configured fallback" in text.lower()
    assert "gpt-current-live-model" in text.split("Models:", 1)[1]


def test_runtime_model_discovery_fallback_supports_custom_provider(monkeypatch, tmp_path: Path) -> None:
    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    monkeypatch.setattr(
        "src.config.get_provider_config",
        lambda provider: (
            {
                "default_model": "custom-current",
                "models": ["custom-fallback"],
            }
            if provider == "custom-provider"
            else None
        ),
    )
    monkeypatch.setattr("src.config.get_global_config_path", lambda: None)
    runtime = Runtime()
    runtime.provider_name = "custom-provider"
    runtime.provider = SimpleNamespace(
        model="custom-current",
        discover_available_models=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    success, text, error = execute_command_sync("model", "", _context(tmp_path, runtime))

    assert success is True
    assert error is None
    assert "  custom-provider:" in text
    assert "custom-current *" in text
    assert "  anthropic:" not in text


def test_runtime_model_rejects_unknown_flag(monkeypatch, tmp_path: Path) -> None:
    """``/model <name> --bogus`` is rejected without swapping."""
    registry = CommandRegistry()
    register_builtin_commands(registry)
    register_runtime_commands(registry)
    monkeypatch.setattr("src.command_system.builtins.get_command_registry", lambda: registry)
    _patch_store(monkeypatch)
    runtime = Runtime()

    success, text, error = execute_command_sync(
        "model",
        "zai/glm-4 --bogus",
        _context(tmp_path, runtime),
    )

    assert success is True
    assert error is None
    assert runtime.swaps == []
    assert "usage:" in text
    assert "Unknown argument" in text
