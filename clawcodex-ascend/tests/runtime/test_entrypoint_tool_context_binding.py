from __future__ import annotations

import io
from types import SimpleNamespace

from src.tool_system.context import ToolContext
from src.tool_system.registry import ToolRegistry


class _MemoryHistory:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def load(self):
        if False:  # pragma: no cover - keeps this an async generator
            yield ""


def test_core_repl_constructor_binds_live_runtime(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.repl import core

    core._load_heavy_runtime()
    provider = SimpleNamespace(model="test-model")
    session = SimpleNamespace(session_id="core-session")
    registry = ToolRegistry()

    monkeypatch.setattr(
        core,
        "get_provider_config",
        lambda _name: {
            "api_key": "test-key",
            "base_url": "https://example.invalid",
            "default_model": "test-model",
        },
    )
    monkeypatch.setattr(core, "get_provider_class", lambda _name: lambda **_kw: provider)
    monkeypatch.setattr(core.Session, "create", lambda *_args: session)
    monkeypatch.setattr(core, "build_default_registry", lambda **_kw: registry)
    monkeypatch.setattr(core, "FileHistory", _MemoryHistory)
    monkeypatch.setattr(core, "PromptSession", lambda **_kw: SimpleNamespace())
    monkeypatch.setattr(core.ClawcodexREPL, "_init_command_system", lambda _self: None)
    monkeypatch.setattr(
        core.ClawcodexREPL,
        "_warm_slash_suggestions_cache",
        lambda _self: None,
    )
    monkeypatch.setattr(core, "_patch_accept_suggestion_bindings", lambda *_a, **_kw: None)
    monkeypatch.setattr(core.Path, "home", lambda: tmp_path)

    repl = core.ClawcodexREPL(provider_name="fake")

    assert repl.tool_context.tool_registry is registry
    assert repl.tool_context.session_id == "core-session"
    assert repl.tool_context._active_provider is provider


def test_extended_repl_overwrites_injected_context(monkeypatch, tmp_path) -> None:
    import prompt_toolkit
    import prompt_toolkit.history
    from clawcodex_ext.repl import core
    from clawcodex_ext.repl.app import ClawCodexExtREPL

    provider = SimpleNamespace(model="test-model")
    session = SimpleNamespace(
        session_id="extended-session",
        conversation=SimpleNamespace(messages=[]),
    )
    registry = ToolRegistry()
    context = ToolContext(
        workspace_root=tmp_path,
        tool_registry=object(),
        session_id="stale-session",
        _active_provider=object(),
    )

    monkeypatch.setattr(ClawCodexExtREPL, "_init_command_system", lambda _self: None)
    monkeypatch.setattr(
        ClawCodexExtREPL,
        "_warm_slash_suggestions_cache",
        lambda _self: None,
    )
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda **_kw: SimpleNamespace())
    monkeypatch.setattr(prompt_toolkit.history, "FileHistory", _MemoryHistory)
    monkeypatch.setattr(core, "_patch_accept_suggestion_bindings", lambda *_a, **_kw: None)
    monkeypatch.setattr(core.Path, "home", lambda: tmp_path)

    repl = ClawCodexExtREPL(
        provider_name="fake",
        provider=provider,
        session=session,
        tool_registry=registry,
        tool_context=context,
        workspace_root=tmp_path,
    )

    assert repl.tool_context is context
    assert context.tool_registry is registry
    assert context.session_id == "extended-session"
    assert context._active_provider is provider
    assert context.allow_session_macro_registration is True
    assert context.confirm_session_macro_plan is not None
    assert callable(context.confirm_session_macro_plan)


def test_headless_entrypoint_binds_query_tool_context(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.entrypoints import headless

    provider = SimpleNamespace(model="test-model")
    registry = ToolRegistry()

    class _Conversation:
        def __init__(self) -> None:
            self.messages = []

        def add_user_message(self, text: str) -> None:
            self.messages.append(SimpleNamespace(role="user", content=text))

        def add_message(self, role, content) -> None:
            self.messages.append(SimpleNamespace(role=role, content=content))

    session = SimpleNamespace(
        session_id="headless-session",
        conversation=_Conversation(),
    )
    captured = {}

    class _Provider:
        def __new__(cls, **_kwargs):
            return provider

    async def _run_query(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            response_text="ok",
            usage={"input_tokens": 1, "output_tokens": 1},
            num_turns=1,
        )

    monkeypatch.setattr(headless, "get_default_provider", lambda: "fake")
    monkeypatch.setattr(
        headless,
        "get_provider_config",
        lambda _name: {
            "api_key": "test-key",
            "base_url": "https://example.invalid",
            "default_model": "test-model",
        },
    )
    monkeypatch.setattr(headless, "get_provider_class", lambda _name: _Provider)
    monkeypatch.setattr(headless.Session, "create", lambda *_args: session)
    monkeypatch.setattr(headless, "build_default_registry", lambda **_kw: registry)
    monkeypatch.setattr(headless, "build_effective_system_prompt", lambda *_a: "")
    monkeypatch.setattr(headless, "run_query_as_agent_loop", _run_query)
    monkeypatch.setattr(
        headless,
        "_install_sigint_handler",
        lambda *_a: lambda: None,
    )
    monkeypatch.setattr(
        "src.outputStyles.resolve_output_style",
        lambda *_a, **_kw: SimpleNamespace(prompt=""),
    )

    result = headless.run_headless(
        headless.HeadlessOptions(
            prompt="hello",
            output_format="text",
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            workspace_root=tmp_path,
        )
    )

    assert result == 0
    context = captured["tool_context"]
    assert context.tool_registry is registry
    assert context.session_id == "headless-session"
    assert context._active_provider is provider
