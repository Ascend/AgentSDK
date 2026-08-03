from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace

from clawcodex_ext.cron_system.runs import (
    CreateCronRunParams,
    create_queued_run,
    read_cron_runs,
)
from clawcodex_ext.repl.core import ClawcodexREPL  # pylint: disable=no-name-in-module


def _make_repl(tmp_path):
    repl = ClawcodexREPL.__new__(ClawcodexREPL)
    repl.tool_context = SimpleNamespace(workspace_root=tmp_path, outbox=[])
    repl._cron_active_tasks = {}
    repl._queued_prompts = deque()
    repl._cron_queued_prompts = deque()
    repl._queued_prompts_lock = threading.Lock()
    return repl


def test_cron_prompt_claims_and_completes_run(tmp_path) -> None:
    run = create_queued_run(
        tmp_path,
        CreateCronRunParams(task_id="task1", prompt="ping", queued_at=1000),
    )
    assert run is not None
    repl = _make_repl(tmp_path)
    repl.tool_context.outbox.append({"type": "cron_prompt", "prompt": "ping", "task_id": "task1", "run_id": run.id})

    repl._drain_cron_outbox()
    result = repl._pop_queued_prompt()
    queued = result[0] if result else ""
    task_id = repl._extract_cron_task_id(queued)

    assert task_id == "task1"
    repl._claim_cron_task(task_id)
    running = read_cron_runs(tmp_path)[0]
    assert running.status == "running"
    assert running.started_at is not None

    repl._finalize_cron_task(task_id)
    completed = read_cron_runs(tmp_path)[0]
    assert completed.status == "completed"
    assert completed.completed_at is not None
    assert completed.ended_at == completed.completed_at
    assert repl._cron_active_tasks == {}


def test_cron_prompt_failure_finalizes_run_failed(tmp_path) -> None:
    run = create_queued_run(
        tmp_path,
        CreateCronRunParams(task_id="task1", prompt="ping", queued_at=1000),
    )
    assert run is not None
    repl = _make_repl(tmp_path)
    repl._cron_active_tasks["task1"] = run.id

    repl._claim_cron_task("task1")
    repl._finalize_cron_task("task1", "failed", error="RuntimeError: boom")

    failed = read_cron_runs(tmp_path)[0]
    assert failed.status == "failed"
    assert failed.error == "RuntimeError: boom"
    assert failed.ended_at == failed.completed_at
    assert repl._cron_active_tasks == {}


def test_duplicate_active_cron_prompt_cancels_new_run(tmp_path) -> None:
    first = create_queued_run(
        tmp_path,
        CreateCronRunParams(task_id="task1", prompt="ping", queued_at=1000),
    )
    assert first is not None
    repl = _make_repl(tmp_path)
    repl._cron_active_tasks["task1"] = first.id

    second = create_queued_run(
        tmp_path,
        CreateCronRunParams(task_id="task2", prompt="ping", queued_at=1000),
    )
    assert second is not None
    repl.tool_context.outbox.append(
        {
            "type": "cron_prompt",
            "prompt": "ping again",
            "task_id": "task1",
            "run_id": second.id,
        }
    )

    repl._drain_cron_outbox()

    runs_by_id = {run.id: run for run in read_cron_runs(tmp_path)}
    assert runs_by_id[first.id].status == "queued"
    assert runs_by_id[second.id].status == "cancelled"
    assert repl._pop_queued_prompt() is None
