from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.query.query import QueryParams, query
from clawcodex_ext.query.task_reminder import (
    TASK_REMINDER_MARKER,
    build_task_reminder,
)
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.registry import ToolRegistry
from clawcodex_ext.tool_system.tools.tasks_v2 import TaskListTool, TaskUpdateTool
from clawcodex_ext.tool_system.tools.todo_write import TodoWriteTool
from clawcodex_ext.types.content_blocks import (
    RedactedThinkingBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from clawcodex_ext.types.messages import AssistantMessage, UserMessage
from clawcodex_ext.utils.abort_controller import AbortController


def _assistant(
    *,
    tool_name: str | None = None,
    api_error: bool = False,
    thinking_only: bool = False,
) -> AssistantMessage:
    if thinking_only:
        content = [RedactedThinkingBlock(data="hidden")]
    elif tool_name is not None:
        content = [ToolUseBlock(id=f"use-{tool_name}", name=tool_name, input={})]
    else:
        content = [TextBlock(text="working")]
    return AssistantMessage(content=content, isApiErrorMessage=api_error)


def _history(turns: int) -> list[UserMessage | AssistantMessage]:
    messages: list[UserMessage | AssistantMessage] = []
    for index in range(turns):
        messages.extend(
            [
                UserMessage(content=f"user-{index}"),
                _assistant(),
            ]
        )
    return messages


def _task_context(tmp_path: Path) -> ToolContext:
    context = ToolContext(workspace_root=tmp_path)
    context.tasks = {
        "1": {
            "id": "1",
            "subject": "Implement reminder",
            "status": "in_progress",
            "lkb": {"derivedStatus": "running"},
        },
        "2": {
            "id": "2",
            "subject": "Recheck downstream docs",
            "status": "completed",
            "lkb": {"derivedStatus": "needs_recheck"},
        },
    }
    return context


def test_task_v2_reminder_triggers_at_ten_turns_with_lkb_statuses(tmp_path: Path) -> None:
    context = _task_context(tmp_path)

    with patch(
        "clawcodex_ext.query.task_reminder.is_todo_v2_enabled",
        return_value=True,
    ):
        reminder = build_task_reminder(
            _history(10),
            context,
            [TaskUpdateTool],
        )

    assert reminder is not None
    assert reminder.isMeta is True
    assert reminder.origin == "system_injection"
    assert isinstance(reminder.content, str)
    assert TASK_REMINDER_MARKER in reminder.content
    assert "#1. [running] Implement reminder" in reminder.content
    assert "#2. [needs_recheck] Recheck downstream docs" in reminder.content
    assert "Never mention this reminder to the user" in reminder.content


def test_task_write_resets_cadence_but_read_only_task_list_does_not(
    tmp_path: Path,
) -> None:
    context = _task_context(tmp_path)
    read_only_history = [
        *_history(9),
        UserMessage(content="check"),
        _assistant(tool_name="TaskList"),
    ]
    write_history = [
        *_history(10),
        UserMessage(content="update"),
        _assistant(tool_name="TaskUpdate"),
    ]

    with patch(
        "clawcodex_ext.query.task_reminder.is_todo_v2_enabled",
        return_value=True,
    ):
        read_only_reminder = build_task_reminder(
            read_only_history,
            context,
            [TaskUpdateTool],
        )
        write_reminder = build_task_reminder(
            write_history,
            context,
            [TaskUpdateTool],
        )

    assert read_only_reminder is not None
    assert write_reminder is None


def test_previous_reminder_is_throttled_for_ten_assistant_turns(tmp_path: Path) -> None:
    context = _task_context(tmp_path)

    with patch(
        "clawcodex_ext.query.task_reminder.is_todo_v2_enabled",
        return_value=True,
    ):
        first = build_task_reminder(_history(10), context, [TaskUpdateTool])
        assert first is not None
        nine_more = [first, *_history(9)]
        assert build_task_reminder(nine_more, context, [TaskUpdateTool]) is None
        assert (
            build_task_reminder(
                [*nine_more, UserMessage(content="continue"), _assistant()],
                context,
                [TaskUpdateTool],
            )
            is not None
        )


def test_api_errors_and_thinking_only_messages_do_not_advance_cadence(
    tmp_path: Path,
) -> None:
    context = _task_context(tmp_path)
    history = [
        *_history(9),
        _assistant(api_error=True),
        _assistant(thinking_only=True),
    ]

    with patch(
        "clawcodex_ext.query.task_reminder.is_todo_v2_enabled",
        return_value=True,
    ):
        assert build_task_reminder(history, context, [TaskUpdateTool]) is None
        assert (
            build_task_reminder(
                [*history, UserMessage(content="continue"), _assistant()],
                context,
                [TaskUpdateTool],
            )
            is not None
        )


def test_todo_write_mode_and_scope_gates(tmp_path: Path) -> None:
    context = ToolContext(workspace_root=tmp_path)
    context.todos = [{"content": "Ship release", "status": "in_progress", "activeForm": "Shipping"}]

    with patch(
        "clawcodex_ext.query.task_reminder.is_todo_v2_enabled",
        return_value=False,
    ):
        reminder = build_task_reminder(_history(10), context, [TodoWriteTool])
        subagent = build_task_reminder(
            _history(10),
            context,
            [TodoWriteTool],
            query_source="agent:builtin:general-purpose",
        )
        unavailable = build_task_reminder(_history(10), context, [])

    assert reminder is not None
    assert isinstance(reminder.content, str)
    assert "1. [in_progress] Ship release" in reminder.content
    assert subagent is None
    assert unavailable is None


def test_query_injects_and_persists_reminder_at_tool_round_boundary(
    tmp_path: Path,
) -> None:
    context = _task_context(tmp_path)
    registry = ToolRegistry([TaskListTool, TaskUpdateTool])
    provider = MagicMock()
    provider.chat_stream_response.side_effect = NotImplementedError()
    provider.chat.side_effect = [
        ChatResponse(
            content="Checking tasks.",
            model="test",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="tool_use",
            tool_uses=[
                {
                    "id": "task-list-1",
                    "name": "TaskList",
                    "input": {},
                }
            ],
        ),
        ChatResponse(
            content="Done.",
            model="test",
            usage={"input_tokens": 20, "output_tokens": 5},
            finish_reason="end_turn",
            tool_uses=None,
        ),
    ]
    persisted: list[UserMessage] = []
    collected: list[object] = []
    params = QueryParams(
        messages=[*_history(9), UserMessage(content="continue")],
        system_prompt="You are helpful.",
        tools=registry.list_tools(),
        tool_registry=registry,
        tool_use_context=context,
        provider=provider,
        abort_controller=AbortController(),
        max_turns=2,
        on_attachment=persisted.append,
    )

    async def run() -> None:
        async for message in query(params):
            collected.append(message)

    with patch(
        "clawcodex_ext.query.task_reminder.is_todo_v2_enabled",
        return_value=True,
    ):
        asyncio.run(run())

    reminders = [
        message
        for message in collected
        if isinstance(message, UserMessage)
        and message.isMeta
        and isinstance(message.content, str)
        and TASK_REMINDER_MARKER in message.content
    ]
    assert len(reminders) == 1
    assert persisted == reminders

    tool_result_index = next(
        index
        for index, message in enumerate(collected)
        if isinstance(message, UserMessage)
        and isinstance(message.content, list)
        and any(isinstance(block, ToolResultBlock) for block in message.content)
    )
    reminder_index = collected.index(reminders[0])
    final_assistant_index = max(
        index for index, message in enumerate(collected) if isinstance(message, AssistantMessage)
    )
    assert tool_result_index < reminder_index < final_assistant_index
    assert provider.chat.call_count == 2
