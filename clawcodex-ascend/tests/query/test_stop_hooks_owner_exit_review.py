"""Regression tests for the subagent owner exit review."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from clawcodex_ext.query.stop_hooks import handle_stop_hooks
from clawcodex_ext.utils.abort_controller import AbortController


def _context(*, agent_id: str | None, tasks: dict[str, dict]):
    return SimpleNamespace(
        agent_id=agent_id,
        teammate_name=None,
        team_name=None,
        agent_type="general-purpose",
        tasks=tasks,
        abort_controller=AbortController(),
        permission_context=None,
        hook_config_manager=None,
        options=None,
        skill_hooks={},
        workspace_trusted=True,
    )


def _run(context, *, query_source: str = "agent:builtin:general-purpose"):
    return asyncio.run(
        handle_stop_hooks(
            [],
            [],
            "",
            context,
            query_source,
        )
    )


def test_plain_subagent_must_review_owned_in_progress_task_before_exit() -> None:
    result = _run(
        _context(
            agent_id="agent-a",
            tasks={
                "T-1": {
                    "id": "T-1",
                    "subject": "Collect references",
                    "status": "in_progress",
                    "owner": "agent-a",
                }
            },
        )
    )

    assert result.prevent_continuation is False
    assert len(result.blocking_errors) == 1
    message = str(result.blocking_errors[0].content)
    assert "T-1" in message
    assert "still owning in-progress tasks" in message
    assert 'status="pending", owner=""' in message


def test_owner_exit_review_ignores_completed_and_other_agents_tasks() -> None:
    result = _run(
        _context(
            agent_id="agent-a",
            tasks={
                "T-1": {
                    "id": "T-1",
                    "subject": "Completed work",
                    "status": "completed",
                    "owner": "agent-a",
                },
                "T-2": {
                    "id": "T-2",
                    "subject": "Someone else's work",
                    "status": "in_progress",
                    "owner": "agent-b",
                },
            },
        )
    )

    assert result.blocking_errors == []
    assert result.prevent_continuation is False


def test_long_lived_agent_loop_is_not_blocked_by_resumable_owned_work() -> None:
    result = _run(
        _context(
            agent_id="persistent-agent",
            tasks={
                "T-1": {
                    "id": "T-1",
                    "subject": "Continue next session",
                    "status": "in_progress",
                    "owner": "persistent-agent",
                }
            },
        ),
        query_source="repl_main_thread",
    )

    assert result.blocking_errors == []
    assert result.prevent_continuation is False
