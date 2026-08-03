"""Per-agent tool exposure is also enforced at execution time."""

from __future__ import annotations

from clawcodex_ext.query.query import _dispatch_single_tool
from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.protocol import ToolResult
from clawcodex_ext.tool_system.registry import ToolRegistry
from clawcodex_ext.types.content_blocks import ToolResultBlock, ToolUseBlock


def _tool(name: str, calls: list[str]):
    return build_tool(
        name=name,
        input_schema={"type": "object"},
        call=lambda _input, _context: (calls.append(name) or ToolResult(name=name, output={"ok": True})),
    )


def test_unadvertised_tool_cannot_dispatch_through_shared_registry(tmp_path) -> None:
    calls: list[str] = []
    allowed = _tool("Read", calls)
    hidden = _tool("Config", calls)
    registry = ToolRegistry([allowed, hidden])
    context = ToolContext(workspace_root=tmp_path)

    message, extras = _dispatch_single_tool(
        ToolUseBlock(id="hidden-call", name="Config", input={}),
        registry,
        context,
        tools=[allowed],
    )

    assert calls == []
    assert extras == []
    assert isinstance(message.content, list)
    result = message.content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.tool_use_id == "hidden-call"
    assert result.is_error is True
    assert "not available in this agent context" in str(result.content)


def test_advertised_tool_still_dispatches(tmp_path) -> None:
    calls: list[str] = []
    allowed = _tool("Read", calls)
    registry = ToolRegistry([allowed])
    context = ToolContext(workspace_root=tmp_path)

    message, extras = _dispatch_single_tool(
        ToolUseBlock(id="allowed-call", name="Read", input={}),
        registry,
        context,
        tools=[allowed],
    )

    assert calls == ["Read"]
    assert extras == []
    result = message.content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error is False
