"""F-22-F: agent ownership model + scheduler + tool-layer filtering.

Pins the contracts from
``docs/feature_plan/05-cron-system/f-22-cron-execution.md`` §Phase F:

- F-1: ``CronTask.agent_id`` / ``team_id``, ``CronRun.owner_agent_id``,
  ``CronTaskDetail.agent_id`` round-trip through snake_case and camelCase
  serialisation. ``format_cron_task_detail`` renders the real ``agent_id``
  rather than the hard-coded ``"Agent: —"`` placeholder.
- F-2: ``CronScheduler(agent_id=...)`` filters due tasks so global
  (``agent_id=None``) tasks fire for every agent, owned tasks fire only for
  their owner.
- F-3: ``CronCreate`` auto-fills ``agent_id`` when supplied; ``CronList``
  returns only own + global tasks by default; ``agent_id="*"`` exposes
  every task; ``CronDelete`` rejects deletion of tasks owned by another
  agent unless the caller is an admin.
- F-5: ``cleanup_orphaned_tasks`` flags owned-but-inactive tasks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clawcodex_ext.cron_system import tools as tools_module
from clawcodex_ext.cron_system.models import CronTask
from clawcodex_ext.cron_system.runs import CronRun
from clawcodex_ext.cron_system.schedule import (
    format_cron_task_detail,
    get_cron_task_detail,
)
from clawcodex_ext.cron_system.scheduler import CronScheduler
from clawcodex_ext.cron_system.tasks import (
    _MtimeCache,
    add_cron_task,
    cleanup_orphaned_tasks,
    read_all_cron_tasks,
    read_cron_tasks,
)
from clawcodex_ext.tool_system.errors import ToolInputError


@pytest.fixture(autouse=True)
def _clear_mtime_cache():
    _MtimeCache.clear()
    yield
    _MtimeCache.clear()


def _stub_tool_context(workspace: Path, agent_id: str | None = None):
    """Build a minimal ``ToolContext`` for direct tool invocations.

    Cron tools only read ``workspace_root``, ``crons``, and ``agent_id``;
    permission checks are bypassed in unit tests because we never declare a
    permission context.
    """
    from src.tool_system.context import ToolContext

    return ToolContext(workspace_root=workspace, crons={}, agent_id=agent_id)


# ---- F-1: model fields --------------------------------------------------


def test_cron_task_agent_id_round_trip_snake_case() -> None:
    """``CronTask.to_dict()`` / ``from_dict()`` preserves ``agent_id``."""
    task = CronTask(
        id="t-001",
        cron="*/5 * * * *",
        prompt="ping",
        agent_id="agent-A",
        team_id="team-1",
        created_at=1_000,
        updated_at=1_000,
    )
    data = task.to_dict()
    assert data["agent_id"] == "agent-A"
    assert data["team_id"] == "team-1"
    restored = CronTask.from_dict(data)
    assert restored is not None
    assert restored.agent_id == "agent-A"
    assert restored.team_id == "team-1"


def test_cron_task_agent_id_round_trip_camel_case() -> None:
    """Backward compat: reading legacy ``agentId`` / ``teamId`` payloads."""
    data = {
        "id": "t-002",
        "cron": "*/5 * * * *",
        "prompt": "ping",
        "agentId": "agent-B",
        "teamId": "team-2",
        "recurring": True,
        "durable": True,
    }
    restored = CronTask.from_dict(data)
    assert restored is not None
    assert restored.agent_id == "agent-B"
    assert restored.team_id == "team-2"


def test_cron_run_owner_agent_id_round_trip() -> None:
    """``CronRun.to_dict()`` / ``from_dict()`` preserves ``owner_agent_id``."""
    run = CronRun(
        id="r-001",
        task_id="t-001",
        prompt="hi",
        status="completed",
        queued_at=1_000,
        owner_agent_id="agent-A",
    )
    data = run.to_dict()
    assert data["owner_agent_id"] == "agent-A"
    restored = CronRun.from_dict(data)
    assert restored is not None
    assert restored.owner_agent_id == "agent-A"


def test_cron_run_owner_agent_id_backward_compat_camel_case() -> None:
    """Older ``ownerAgentId`` payloads decode without loss."""
    data = {
        "id": "r-002",
        "task_id": "t-002",
        "prompt": "hi",
        "status": "running",
        "queued_at": 1_000,
        "ownerAgentId": "agent-LEGACY",
    }
    restored = CronRun.from_dict(data)
    assert restored is not None
    assert restored.owner_agent_id == "agent-LEGACY"


def test_cron_task_detail_renders_real_agent_id(tmp_path) -> None:
    """``format_cron_task_detail`` shows the real ``agent_id``, not ``"Agent: —"``."""
    task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="ping",
        durable=True,
        created_at=1_000,
        agent_id="agent-A",
    )

    detail = get_cron_task_detail(tmp_path, task.id)
    assert detail is not None
    output = format_cron_task_detail(detail)

    assert "Agent: agent-A" in output
    # Sanity: the hard-coded "Agent: —" must NOT appear when the agent_id is set.
    assert "Agent: —" not in output


def test_cron_task_detail_renders_dash_when_unset(tmp_path) -> None:
    """Unset ``agent_id`` keeps the legacy ``"Agent: —"`` placeholder."""
    task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="ping",
        durable=True,
        created_at=1_000,
    )

    detail = get_cron_task_detail(tmp_path, task.id)
    assert detail is not None
    output = format_cron_task_detail(detail)

    assert "Agent: —" in output


# ---- F-2: scheduler filtering ------------------------------------------


def _silent_scheduler(workspace: Path, **kwargs) -> CronScheduler:
    return CronScheduler(workspace, on_fire=lambda prompt: None, **kwargs)


def test_scheduler_with_no_agent_id_sees_all_tasks(tmp_path) -> None:
    """Single-agent mode (``agent_id=None``) does not apply any ownership filter.

    Global-vs-owned distinction only kicks in when the scheduler is
    constructed with an explicit ``agent_id``; the ``None`` case treats
    every task as fireable.
    """
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="global", durable=True, created_at=1_000)
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="owned",
        durable=True,
        created_at=1_000,
        agent_id="agent-A",
    )

    scheduler = _silent_scheduler(tmp_path)  # no agent_id
    tasks = read_all_cron_tasks(tmp_path)
    filtered = scheduler._agent_owned_only(tasks)
    assert {t.prompt for t in filtered} == {"global", "owned"}


def test_scheduler_filters_to_own_agent(tmp_path) -> None:
    """``agent_id="agent-A"`` keeps only global + own tasks."""
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="global", durable=True, created_at=1_000)
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="owned-A",
        durable=True,
        created_at=1_000,
        agent_id="agent-A",
    )
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="owned-B",
        durable=True,
        created_at=1_000,
        agent_id="agent-B",
    )

    scheduler = _silent_scheduler(tmp_path, agent_id="agent-A")
    tasks = read_all_cron_tasks(tmp_path)
    filtered = scheduler._agent_owned_only(tasks)
    assert {t.prompt for t in filtered} == {"global", "owned-A"}


# ---- F-3: tool-layer visibility ---------------------------------------


def test_cron_create_stamps_agent_id_when_provided(tmp_path) -> None:
    """CronCreate tool writes ``agent_id`` from the tool context onto the task."""
    ctx = _stub_tool_context(tmp_path, agent_id="agent-A")
    result = tools_module.CronCreateTool.call(
        {
            "cron": "*/5 * * * *",
            "prompt": "hi",
            "durable": True,
        },
        ctx,
    )
    tasks = read_cron_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].agent_id == "agent-A"
    # ToolResult.output reflects the call outcome.
    assert result.output.get("id") == tasks[0].id
    assert result.output.get("agentId") == "agent-A"


def test_cron_list_filters_by_viewer_agent_id(tmp_path) -> None:
    """``CronList`` with ``agent_id="agent-A"`` returns own + global only."""
    # Two global + two owned-by-A + one owned-by-B
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="g1", created_at=1_000)
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="g2", created_at=1_000)
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="oA1",
        created_at=1_000,
        agent_id="agent-A",
    )
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="oA2",
        created_at=1_000,
        agent_id="agent-A",
    )
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="oB1",
        created_at=1_000,
        agent_id="agent-B",
    )

    ctx = _stub_tool_context(tmp_path, agent_id="agent-A")

    # Agent-A sees own + global tasks only.
    viewer_result = tools_module.CronListTool.call({}, ctx)
    prompts = {job["prompt"] for job in viewer_result.output["jobs"]}
    assert prompts == {"g1", "g2", "oA1", "oA2"}

    # Admin view ("*") sees all tasks.
    admin_result = tools_module.CronListTool.call({"agent_id": "*"}, ctx)
    admin_prompts = {job["prompt"] for job in admin_result.output["jobs"]}
    assert admin_prompts == {"g1", "g2", "oA1", "oA2", "oB1"}


def test_cron_delete_rejects_foreign_owned_task(tmp_path) -> None:
    """``CronDelete`` refuses to remove another agent's task by default."""
    task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="owned-by-B",
        created_at=1_000,
        agent_id="agent-B",
    )

    ctx = _stub_tool_context(tmp_path, agent_id="agent-A")

    # viewer agent-A cannot delete B's task.
    with pytest.raises(ToolInputError):
        tools_module.CronDeleteTool.call({"id": task.id}, ctx)
    # Task is still on disk.
    assert any(t.id == task.id for t in read_cron_tasks(tmp_path))

    # Admin (``*``) can.
    admin_result = tools_module.CronDeleteTool.call({"id": task.id, "agent_id": "*"}, ctx)
    assert admin_result.output.get("success") is True
    assert admin_result.output.get("id") == task.id
    assert not any(t.id == task.id for t in read_cron_tasks(tmp_path))


def test_cron_delete_allows_owner_to_remove_own_task(tmp_path) -> None:
    """Owner agent can delete their own task without escalation."""
    task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="mine",
        created_at=1_000,
        agent_id="agent-A",
    )

    ctx = _stub_tool_context(tmp_path, agent_id="agent-A")
    result = tools_module.CronDeleteTool.call({"id": task.id}, ctx)
    assert result.output.get("success") is True
    assert not any(t.id == task.id for t in read_cron_tasks(tmp_path))


# ---- F-5: cleanup_orphaned_tasks --------------------------------------


def test_cleanup_orphaned_tasks_returns_inactive_owned_tasks(tmp_path) -> None:
    """Owned tasks whose agent is not in the active set are returned as orphaned."""
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="g", created_at=1_000)
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="alive",
        created_at=1_000,
        agent_id="alive-agent",
    )
    add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="ghost",
        created_at=1_000,
        agent_id="ghost-agent",
    )

    orphaned = cleanup_orphaned_tasks(tmp_path, active_agents={"alive-agent"})
    assert len(orphaned) == 1
    assert orphaned[0].agent_id == "ghost-agent"


def test_cleanup_orphaned_tasks_skips_global_tasks(tmp_path) -> None:
    """Global (``agent_id=None``) tasks are never flagged as orphaned."""
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="g", created_at=1_000)
    assert cleanup_orphaned_tasks(tmp_path, active_agents={"agent-X"}) == []
