"""F-88 ``/monitor`` slash command family.

Subcommands:
  /monitor <cmd>            Start a background monitor task.
  /monitor list             List active monitor tasks.
  /monitor stop <task_id>   Stop a monitor task.
  /monitor tail <task_id>   Show the latest output of a monitor task.
"""

from __future__ import annotations

from pathlib import Path

from .engine import CommandContext
from .types import LocalCommand, LocalCommandResult
from clawcodex_ext.feature_gate import get_registry as _get_feature_registry  # pylint: disable=no-name-in-module
from clawcodex_ext.services.monitor.controller import MonitorController


def _is_monitor_enabled() -> bool:
    return _get_feature_registry().is_enabled("MONITOR_TOOL")


def _get_controller(context: CommandContext) -> MonitorController:
    tool_context = getattr(context, "tool_context", None)
    if tool_context is None:
        raise RuntimeError("/monitor requires an active ToolContext")
    return MonitorController(tool_context)


def _handle_monitor_command(args: str, context: CommandContext) -> LocalCommandResult:
    """Dispatch ``/monitor`` subcommands."""
    parts = args.strip().split(None, 1)
    action = parts[0].lower() if parts else "start"
    rest = parts[1] if len(parts) > 1 else ""

    if action == "list":
        return _monitor_list(context)
    if action == "stop":
        return _monitor_stop(rest, context)
    if action == "tail":
        return _monitor_tail(rest, context)
    if action in ("start", ""):
        command = rest or args.strip()
        if not command:
            return LocalCommandResult(
                type="text",
                value="Usage: /monitor <cmd> | /monitor list | /monitor stop <task_id> | /monitor tail <task_id>",
            )
        return _monitor_start(command, context)

    # Unknown action — treat the whole argument string as a command to start.
    command = args.strip()
    return _monitor_start(command, context)


def _monitor_start(command: str, context: CommandContext) -> LocalCommandResult:
    ctrl = _get_controller(context)
    cwd = context.cwd if context.cwd else context.workspace_root
    result = ctrl.start(command=command, cwd=Path(cwd))
    return LocalCommandResult(
        type="text",
        value=(
            f"Monitor started: {result.task_id}\n"
            f"Command: {command}\n"
            f"Output: {result.output_path}\n"
            "Use /monitor list, /monitor tail, or /monitor stop to interact."
        ),
    )


def _monitor_list(context: CommandContext) -> LocalCommandResult:
    ctrl = _get_controller(context)
    active = ctrl.list_active()
    if not active:
        return LocalCommandResult(type="text", value="No active monitor tasks.")

    lines = ["Active monitor tasks:"]
    for i, state in enumerate(active, 1):
        desc = state.description or state.command
        interval = getattr(state, "interval_sec", None)
        interval_str = f" (interval={interval}s)" if interval else ""
        lines.append(f"  {i}. {state.id}: {desc}{interval_str}\n     output: {state.output_path}")
    return LocalCommandResult(type="text", value="\n".join(lines))


def _monitor_stop(args: str, context: CommandContext) -> LocalCommandResult:
    task_id = args.strip()
    if not task_id:
        return LocalCommandResult(type="text", value="Usage: /monitor stop <task_id>")
    ctrl = _get_controller(context)
    ok = ctrl.stop(task_id)
    if ok:
        return LocalCommandResult(type="text", value=f"Monitor {task_id} stopped.")
    return LocalCommandResult(
        type="text",
        value=f"Could not stop {task_id} (not found or already finished).",
    )


def _monitor_tail(args: str, context: CommandContext) -> LocalCommandResult:
    task_id = args.strip()
    if not task_id:
        return LocalCommandResult(type="text", value="Usage: /monitor tail <task_id>")
    ctrl = _get_controller(context)
    snapshot = ctrl.read(task_id, max_bytes=200_000)
    if snapshot is None:
        return LocalCommandResult(type="text", value=f"Monitor {task_id} not found.")

    output = snapshot.get("output", "")
    status = snapshot.get("status", "unknown")
    return LocalCommandResult(
        type="text",
        value=f"[{task_id}] status={status}\n{output or '(no output yet)'}",
    )


MONITOR_COMMAND = LocalCommand(
    name="monitor",
    description="Start a background monitor task (Shift+Down to view in TUI)",
    argument_hint="[start|list|stop|tail] <cmd|task_id>",
    supports_non_interactive=True,
    is_enabled=_is_monitor_enabled,
)
MONITOR_COMMAND.set_call(_handle_monitor_command)


__all__ = ["MONITOR_COMMAND"]
