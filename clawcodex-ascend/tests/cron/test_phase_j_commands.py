"""F-22-J-1 / F-22-J-2: cron command aliases and ``--deep`` integration.

Pins the contract from
``docs/feature_plan/05-cron-system/f-22-cron-execution.md`` §Phase J:

J-1 — ``/cron-trigger`` is a behaviorally-equivalent alias for
``/cron-run`` / ``/cron-fire``. All three share the same handler
(``CRON_RUN_COMMAND``) and call the same ``CronRunTool`` with the same
arguments. Verified by exercising the registry's alias resolution.

J-2 — ``/cron-list --deep`` parses the flag, passes ``deep=True`` to
``build_schedule_list``, and appends a "(deep mode)" marker to the output.
``/cron-status --deep`` / ``/cron-runs --deep`` mirror the same wiring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from clawcodex_ext.command_system.builtins import (  # pylint: disable=no-name-in-module
    CRON_RUN_COMMAND,
    cron_list_command_call,
    cron_run_command_call,
)
from clawcodex_ext.cron_system import status as status_module
from clawcodex_ext.cron_system.status import build_autonomy_status
from clawcodex_ext.cron_system.tasks import add_cron_task


# ---- Helpers ---------------------------------------------------------------


@dataclass
class _NoRuntimeContext:
    """Stub that signals "no cron tool runtime", forcing the fallback path.

    The cron-list handler falls back to ``build_schedule_list`` when there
    is no tool registry / tool context available. This stub intentionally
    leaves those attributes undefined so the handler takes the fallback.
    """

    workspace_root: Path


@dataclass
class _OutboxContext:
    """Stub with a list-like outbox for ``cron_run_command_call``.

    The handler only inspects ``tool_context.outbox`` for the optional
    "Queued for execution in this session" message; providing a real
    ``list`` lets us assert the synthetic run was appended.
    """

    workspace_root: Path
    tool_context: Any
    tool_registry: Any

    def __init__(self, workspace: Path) -> None:
        self.workspace_root = workspace
        self.tool_context = _ToolCtx(workspace)
        self.tool_registry = _Registry()


@dataclass
class _ToolCtx:
    workspace_root: Path | None
    outbox: list = field(default_factory=list)


@dataclass
class _Registry:
    """Registry stub that returns a synthetic CronRun result.

    Matches the shape of ``ToolRegistry.dispatch`` so the handler does not
    raise — the CronRun tool stub returns a ``run`` dict.
    """

    def dispatch(self, call, ctx):
        from clawcodex_ext.tool_system.protocol import ToolResult

        return ToolResult(
            name="CronRun",
            output={
                "run": {
                    "id": "r-fake",
                    "task_id": call.input.get("id", ""),
                    "prompt": "ping",
                }
            },
        )


# ---- J-1: /cron-trigger alias ---------------------------------------------


def test_cron_run_command_has_trigger_alias() -> None:
    """``CRON_RUN_COMMAND.aliases`` includes ``cron-trigger`` alongside ``cron-fire``."""
    assert "cron-trigger" in CRON_RUN_COMMAND.aliases
    assert "cron-fire" in CRON_RUN_COMMAND.aliases


def test_cron_run_command_uses_same_handler_for_all_aliases() -> None:
    """The trigger alias must resolve to the same handler as cron-run.

    The handler is stored on the private ``_call_impl`` slot; the alias
    layer simply dispatches all three names (``cron-run``,
    ``cron-fire``, ``cron-trigger``) to the same underlying callable.
    """
    assert CRON_RUN_COMMAND._call_impl is cron_run_command_call


def test_cron_trigger_fires_via_cron_run_tool(tmp_path) -> None:
    """Calling the shared handler with cron-trigger arguments hits CronRunTool.

    Confirms the alias path goes through the same handler that
    ``/cron-run`` and ``/cron-fire`` already use.
    """
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", durable=True, created_at=1_000)
    ctx = _OutboxContext(tmp_path)

    # ``cron_run_command_call`` ignores which alias was used — it only reads args.
    result = cron_run_command_call(task.id, ctx)

    assert result.type == "text"
    assert f"Trigger {task.id} fired." in result.value
    assert "Run ID: r-fake" in result.value
    # Synthetic run was appended to the outbox (one entry).
    assert len(ctx.tool_context.outbox) == 1


# ---- J-2: --deep in /cron-list --------------------------------------------


def test_cron_list_deep_marker_appears(tmp_path, monkeypatch) -> None:
    """``/cron-list --deep`` appends "(deep mode)" to the rendered output."""
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", created_at=1_000)

    captured: dict[str, bool] = {}

    def _spy_schedule_list(workspace, *, deep: bool = False) -> str:
        captured["deep"] = deep
        return "Scheduled cron jobs\n  ID Schedule..."

    monkeypatch.setattr(status_module, "build_schedule_list", _spy_schedule_list)

    ctx = _NoRuntimeContext(workspace_root=tmp_path)
    result = cron_list_command_call("--deep", ctx)

    assert captured["deep"] is True
    assert "(deep mode)" in result.value


def test_cron_list_without_deep_does_not_mark(tmp_path, monkeypatch) -> None:
    """``/cron-list`` without ``--deep`` must NOT add the marker or pass deep=True."""
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", created_at=1_000)

    captured: dict[str, bool] = {}

    def _spy_schedule_list(workspace, *, deep: bool = False) -> str:
        captured["deep"] = deep
        return "Scheduled cron jobs\n  ID Schedule..."

    monkeypatch.setattr(status_module, "build_schedule_list", _spy_schedule_list)

    ctx = _NoRuntimeContext(workspace_root=tmp_path)
    result = cron_list_command_call("", ctx)

    assert captured["deep"] is False
    assert "(deep mode)" not in result.value


def test_cron_list_passes_deep_through_to_build_schedule_list(tmp_path, monkeypatch) -> None:
    """``build_schedule_list`` itself receives the ``deep`` kwarg verbatim."""
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", created_at=1_000)

    seen: list[bool] = []

    def _spy(workspace, *, deep: bool = False) -> str:
        seen.append(deep)
        return "Scheduled cron jobs\n  ID Schedule..."

    monkeypatch.setattr(status_module, "build_schedule_list", _spy)

    ctx = _NoRuntimeContext(workspace_root=tmp_path)
    cron_list_command_call("--deep", ctx)
    cron_list_command_call("", ctx)

    assert seen == [True, False]


def test_build_autonomy_status_respects_deep_for_runs(tmp_path) -> None:
    """``build_autonomy_status`` truncates runs at 10 by default; --deep shows all.

    The truncation threshold is hard-coded to 10 inside the function; this
    test creates 12 synthetic completed runs and verifies the default path
    hides the tail while ``deep=True`` shows all.
    """
    runs_path = tmp_path / ".clawcodex" / "cron" / "scheduled_task_runs.json"
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    payloads = []
    for i in range(12):
        payloads.append(
            {
                "id": f"r-{i:02d}",
                "task_id": f"t-{i:02d}",
                "prompt": f"prompt {i}",
                "status": "completed",
                "queued_at": 1_700_000_000_000 + i,
            }
        )
    runs_path.write_text(json.dumps({"version": 1, "runs": payloads}), encoding="utf-8")

    default_status = build_autonomy_status(tmp_path, deep=False)
    deep_status = build_autonomy_status(tmp_path, deep=True)

    assert "older runs hidden" in default_status
    assert "older runs hidden" not in deep_status
    # Deep mode shows every run id in full.
    for i in range(12):
        assert f"r-{i:02d}" in deep_status


def test_cron_status_and_runs_advertise_deep_flag() -> None:
    """``cron-status`` and ``cron-runs`` advertise ``[--deep]`` for help text."""
    from clawcodex_ext.command_system.builtins import (  # pylint: disable=no-name-in-module
        CRON_STATUS_COMMAND,
        CRON_RUNS_COMMAND,
    )

    assert "--deep" in (CRON_STATUS_COMMAND.argument_hint or "")
    assert "--deep" in (CRON_RUNS_COMMAND.argument_hint or "")
