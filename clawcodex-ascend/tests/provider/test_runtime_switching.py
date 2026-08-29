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

from clawcodex_ext.runtime.context import RuntimeContext, RuntimeOptions


class FakeProvider:
    def __init__(self, model: str | None = None) -> None:
        self.model = model


class FakeRegistry:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider
        self._tools = []
        self._by_name = {}

    def register(self, tool: object) -> None:
        name = getattr(tool, "name", tool.__class__.__name__.lower())
        self._tools.append(tool)
        self._by_name[name] = tool

    def unregister(self, name: str) -> None:
        self._tools = [t for t in self._tools if getattr(t, "name", "") != name]
        self._by_name.pop(name, None)

    def list_tools(self) -> list:
        return list(self._tools)

    def dispatch(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise NotImplementedError


def test_runtime_swap_does_not_run_model_discovery_hooks(monkeypatch, tmp_path: Path) -> None:
    import clawcodex_ext.cli.model_cmd.registry as registry_module

    hook_called = threading.Event()

    def slow_hook() -> list[str]:
        hook_called.set()
        time.sleep(0.3)
        return ["account-only-model"]

    monkeypatch.setitem(registry_module._DISCOVERY_HOOKS, "anthropic", [slow_hook])
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: FakeProvider(model),
    )
    monkeypatch.setattr(
        "src.tool_system.defaults.build_default_registry",
        lambda provider, **kwargs: FakeRegistry(provider),
    )
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.replace_cron_tools",
        lambda registry: None,
    )
    runtime = RuntimeContext(
        options=RuntimeOptions(
            provider_name="anthropic",
            model="claude-sonnet-4-6",
            workspace_root=tmp_path,
        ),
        provider_name="anthropic",
        provider=FakeProvider("claude-sonnet-4-6"),
        session=object(),
        tool_registry=FakeRegistry(FakeProvider()),
        tool_context=SimpleNamespace(),
        workspace_root=tmp_path,
    )

    before = time.perf_counter()
    runtime.swap_provider("anthropic", "account-only-model")
    elapsed = time.perf_counter() - before

    assert elapsed < 0.25
    assert hook_called.is_set() is False
    assert runtime.provider.model == "account-only-model"


def test_runtime_context_build_uses_model_resolver(monkeypatch, tmp_path: Path) -> None:
    built: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.resolve",
        lambda **kwargs: SimpleNamespace(provider="glm", model="zai/glm-4"),
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: (built.append((provider_name, model)) or FakeProvider(model)),
    )
    monkeypatch.setattr(
        "src.tool_system.defaults.build_default_registry",
        lambda provider, **kwargs: FakeRegistry(provider),
    )
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.attach_cron_runtime",
        lambda runtime, **kwargs: None,
    )

    options = RuntimeOptions(
        provider_name=None,
        model=None,
        max_turns=20,
        stream=True,
        allowed_tools=(),
        disallowed_tools=(),
        workspace_root=tmp_path,
        permission_mode="default",
        is_bypass_permissions_mode_available=False,
        resume_session_id=None,
        resume_browse=False,
    )

    runtime = RuntimeContext.build(options)

    assert built == [("glm", "zai/glm-4")]
    assert runtime.provider_name == "glm"
    assert runtime.provider.model == "zai/glm-4"
    assert options.provider_name == "glm"
    assert options.model == "zai/glm-4"


def test_runtime_context_swap_provider_replaces_provider_and_registry(monkeypatch, tmp_path: Path) -> None:
    registries: list[FakeRegistry] = []

    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: FakeProvider(model),
    )

    def fake_registry(provider: FakeProvider, **kwargs) -> FakeRegistry:
        registry = FakeRegistry(provider)
        registries.append(registry)
        return registry

    monkeypatch.setattr("src.tool_system.defaults.build_default_registry", fake_registry)
    monkeypatch.setattr(
        "clawcodex_ext.cron_system.runtime.replace_cron_tools",
        lambda registry: None,
    )

    tool_context = SimpleNamespace(provider=None, provider_name=None, tool_registry=None)
    runtime = RuntimeContext(
        options=SimpleNamespace(
            allowed_tools=(),
            disallowed_tools=(),
            provider_name="anthropic",
            model="claude-sonnet-4-6",
        ),
        provider_name="anthropic",
        provider=FakeProvider("claude-sonnet-4-6"),
        session=object(),
        tool_registry=FakeRegistry(FakeProvider()),
        tool_context=tool_context,
        workspace_root=tmp_path,
    )

    runtime.swap_provider("glm", "zai/glm-4")

    assert runtime.provider_name == "zai"
    assert runtime.provider.model == "zai/glm-4"
    assert runtime.tool_registry is registries[-1]
    assert runtime.options.provider_name == "zai"
    assert runtime.options.model == "zai/glm-4"
    assert tool_context.provider is runtime.provider
    assert tool_context.provider_name == "zai"
    assert tool_context.tool_registry is runtime.tool_registry


class _CountingObserver:
    """Test double that records every swap notification."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def on_runtime_swap(self, runtime) -> None:
        self.calls.append(runtime)


def test_swap_provider_notifies_attached_observers(monkeypatch, tmp_path: Path) -> None:
    """``swap_provider`` must fan out to all attached observers."""
    from clawcodex_ext.runtime.observer import attach_observer

    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: FakeProvider(model),
    )
    monkeypatch.setattr(
        "src.tool_system.defaults.build_default_registry",
        lambda provider, **kwargs: FakeRegistry(provider),
    )
    monkeypatch.setattr(
        "clawcodex_ext.cron_system.runtime.replace_cron_tools",
        lambda registry: None,
    )

    runtime = RuntimeContext(
        options=SimpleNamespace(
            allowed_tools=(),
            disallowed_tools=(),
            provider_name="anthropic",
            model="claude-sonnet-4-6",
        ),
        provider_name="anthropic",
        provider=FakeProvider("claude-sonnet-4-6"),
        session=object(),
        tool_registry=FakeRegistry(FakeProvider()),
        tool_context=SimpleNamespace(),
        workspace_root=tmp_path,
    )

    observer_a = _CountingObserver()
    observer_b = _CountingObserver()
    attach_observer(runtime, observer_a)
    attach_observer(runtime, observer_b)

    runtime.swap_provider("glm", "zai/glm-4")

    assert observer_a.calls == [runtime]
    assert observer_b.calls == [runtime]


def test_swap_provider_observer_errors_do_not_break_swap(monkeypatch, tmp_path: Path) -> None:
    """A faulty observer must not abort the swap."""
    from clawcodex_ext.runtime.observer import attach_observer

    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: FakeProvider(model),
    )
    monkeypatch.setattr(
        "src.tool_system.defaults.build_default_registry",
        lambda provider, **kwargs: FakeRegistry(provider),
    )
    monkeypatch.setattr(
        "clawcodex_ext.cron_system.runtime.replace_cron_tools",
        lambda registry: None,
    )

    runtime = RuntimeContext(
        options=SimpleNamespace(
            allowed_tools=(),
            disallowed_tools=(),
            provider_name="anthropic",
            model="claude-sonnet-4-6",
        ),
        provider_name="anthropic",
        provider=FakeProvider("claude-sonnet-4-6"),
        session=object(),
        tool_registry=FakeRegistry(FakeProvider()),
        tool_context=SimpleNamespace(),
        workspace_root=tmp_path,
    )

    class _Boom:
        def on_runtime_swap(self, runtime) -> None:
            raise RuntimeError("boom")

    healthy = _CountingObserver()
    attach_observer(runtime, _Boom())
    attach_observer(runtime, healthy)

    runtime.swap_provider("glm", "zai/glm-4")

    assert runtime.provider_name == "zai"
    assert healthy.calls == [runtime]


def test_install_repl_extensions_attaches_observer(monkeypatch) -> None:
    """``install_repl_extensions`` must register runtime commands + observer."""
    from clawcodex_ext.frontend.repl_extensions import install_repl_extensions

    runtime = SimpleNamespace(
        provider=FakeProvider("claude-sonnet-4-6"),
        provider_name="anthropic",
        tool_registry=FakeRegistry(FakeProvider()),
        tool_context=SimpleNamespace(),
        options=SimpleNamespace(model="claude-sonnet-4-6"),
        _observers=[],
    )
    prewarmed = []
    monkeypatch.setattr(
        "clawcodex_ext.frontend.repl_extensions.prewarm_model_catalog",
        lambda provider_name, provider: prewarmed.append((provider_name, provider)),
        raising=False,
    )

    class _Repl:
        provider = None
        provider_name = None
        tool_registry = None
        tool_context = None
        command_context = SimpleNamespace(
            provider=None,
            tool_registry=None,
            tool_context=None,
        )
        command_registry = SimpleNamespace(register=lambda cmd: None)

    repl = _Repl()
    install_repl_extensions(repl, runtime)

    assert runtime._observers
    observer = runtime._observers[0]
    assert hasattr(observer, "on_runtime_swap")

    fake_new = SimpleNamespace(
        provider=FakeProvider("zai/glm-4"),
        provider_name="glm",
        tool_registry=FakeRegistry(FakeProvider("zai/glm-4")),
        tool_context=SimpleNamespace(),
    )
    observer.on_runtime_swap(fake_new)

    assert repl.provider is fake_new.provider
    assert repl.provider_name == "glm"
    assert repl.tool_registry is fake_new.tool_registry
    assert repl.tool_context is fake_new.tool_context
    assert repl.command_context.provider is fake_new.provider
    assert repl.command_context.tool_registry is fake_new.tool_registry
    assert repl.command_context.tool_context is fake_new.tool_context
    assert prewarmed == [
        ("anthropic", runtime.provider),
        ("glm", fake_new.provider),
    ]


def test_repl_multimodel_renderer_shows_each_slot_result() -> None:
    """The interactive transcript exposes both candidates, not just the winner."""
    from clawcodex_ext.frontend.repl_extensions import _ReplMultiModelRenderer

    class _Console:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, value, **_kwargs) -> None:
            self.lines.append(str(value))

    console = _Console()
    renderer = _ReplMultiModelRenderer(SimpleNamespace(console=console))
    results = [
        SimpleNamespace(
            slot_name="agnes",
            duration_ms=9300,
            error=None,
            response=SimpleNamespace(model="agnes-2.0-flash", content="AGNES_OK"),
        ),
        SimpleNamespace(
            slot_name="minimax",
            duration_ms=1400,
            error=None,
            response=SimpleNamespace(model="MiniMax-M3", content="MINIMAX_OK"),
        ),
    ]
    renderer._on_router_event(
        "aggregated",
        {"results": results, "output": SimpleNamespace(chosen=results[0].response)},
    )

    rendered = "\n".join(console.lines)
    assert "agnes-2.0-flash · 9.30s" in rendered
    assert "MiniMax-M3 · 1.40s" in rendered
    assert "AGNES_OK" in rendered
    assert "MINIMAX_OK" in rendered
    assert "最终采用: agnes-2.0-flash" in rendered


def test_install_tui_extensions_attaches_observer(monkeypatch) -> None:
    """``install_tui_extensions`` must wire the TUI observer to the runtime."""
    from clawcodex_ext.frontend.tui_extensions import install_tui_extensions

    runtime = SimpleNamespace(
        provider=FakeProvider("claude-sonnet-4-6"),
        provider_name="anthropic",
        tool_registry=FakeRegistry(FakeProvider()),
        tool_context=SimpleNamespace(),
        options=SimpleNamespace(model="claude-sonnet-4-6"),
        _observers=[],
    )
    prewarmed = []
    monkeypatch.setattr(
        "clawcodex_ext.frontend.tui_extensions.prewarm_model_catalog",
        lambda provider_name, provider: prewarmed.append((provider_name, provider)),
        raising=False,
    )

    class _StatusBar:
        def __init__(self) -> None:
            self.refreshes: list[tuple[str, str | None]] = []

        def refresh_identity(self, provider, model) -> None:
            self.refreshes.append((provider, model))

    class _ReplScreen:
        status_bar: _StatusBar

    class _App:
        provider = None
        provider_name = None
        model = None
        tool_registry = None
        tool_context = None
        app_state = SimpleNamespace(provider=None, model=None)
        _command_context = SimpleNamespace(
            provider=None,
            tool_registry=None,
            tool_context=None,
        )
        _repl_screen = _ReplScreen()
        _repl_screen.status_bar = _StatusBar()

        class _Bridge:
            def replace_runtime(self, **kwargs) -> None:
                self.replaced = kwargs

        _agent_bridge = _Bridge()

    app = _App()
    install_tui_extensions(app, runtime)

    assert runtime._observers
    observer = runtime._observers[0]
    assert hasattr(observer, "on_runtime_swap")

    fake_new = SimpleNamespace(
        provider=FakeProvider("zai/glm-4"),
        provider_name="glm",
        tool_registry=FakeRegistry(FakeProvider("zai/glm-4")),
        tool_context=SimpleNamespace(),
        options=SimpleNamespace(model="zai/glm-4"),
    )
    observer.on_runtime_swap(fake_new)

    assert app.provider is fake_new.provider
    assert app.provider_name == "glm"
    assert app.model == "zai/glm-4"
    assert app.tool_registry is fake_new.tool_registry
    assert app.app_state.provider == "glm"
    assert app.app_state.model == "zai/glm-4"
    assert app._repl_screen.status_bar.refreshes == [("glm", "zai/glm-4")]
    assert prewarmed == [
        ("anthropic", runtime.provider),
        ("glm", fake_new.provider),
    ]
