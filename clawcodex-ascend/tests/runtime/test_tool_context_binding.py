from __future__ import annotations

from types import SimpleNamespace

from clawcodex_ext.runtime.tool_context_binding import bind_tool_context_runtime
from src.tool_system.context import ToolContext


def test_binding_overwrites_injected_runtime_references(tmp_path) -> None:
    stale_registry = object()
    stale_provider = object()
    context = ToolContext(
        workspace_root=tmp_path,
        tool_registry=stale_registry,
        session_id="stale-session",
        _active_provider=stale_provider,
    )
    registry = object()
    provider = object()
    session = SimpleNamespace(session_id="live-session")

    returned = bind_tool_context_runtime(
        context,
        tool_registry=registry,
        session=session,
        provider=provider,
    )

    assert returned is context
    assert context.tool_registry is registry
    assert context.session_id == "live-session"
    assert context._active_provider is provider


def test_binding_leaves_fields_untouched_until_runtime_object_is_available(
    tmp_path,
) -> None:
    registry = object()
    provider = object()
    context = ToolContext(
        workspace_root=tmp_path,
        tool_registry=registry,
        session_id="session-1",
        _active_provider=provider,
    )

    bind_tool_context_runtime(context)

    assert context.tool_registry is registry
    assert context.session_id == "session-1"
    assert context._active_provider is provider


def test_binding_accepts_absent_context() -> None:
    assert bind_tool_context_runtime(None, provider=object()) is None
