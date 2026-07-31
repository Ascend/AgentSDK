"""F-94 P94-E2 — ``/bg`` 命令族。

面向用户的后台会话命令（f-94-bg-sessions.md §1.8）::

    /bg                       — 等价 /bg list
    /bg list [--all]          — 列出后台会话（--all 含终态）
    /bg inspect <id>          — 详细状态
    /bg attach <id> [--all-ws]— tail transcript + 恢复命令
    /bg stop <id> [--force]   — 停止（graceful-first）
    /bg cleanup [--failed]    — 清理终态/orphaned 记录
    /bg logs <id> [--tail N]  — 仅查看 transcript 尾部

注册方式（黄金法则 5）：通过 ``register_bg_commands(registry)`` 注册到
全局 ``CommandRegistry``，由 ``builtins.register_builtin_commands`` 调用，
不修改上游命令枚举。
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from clawcodex_ext.tasks.bg_session import (  # pylint: disable=no-name-in-module
    BgSessionAttachError,
    BgSessionNotFoundError,
    BgSessionPermissionError,
    BgSessionStopError,
    BgSessionsDisabledError,
    is_bg_sessions_enabled,
)
from clawcodex_ext.tasks.bg_session_manager import BgSessionManager  # pylint: disable=no-name-in-module
from clawcodex_ext.tasks.bg_session_registry import BgSessionRegistry  # pylint: disable=no-name-in-module

from .types import CommandAvailability, LocalCommand, LocalCommandResult


def _get_manager(context: Any) -> BgSessionManager:
    """从 CommandContext 构造 manager。

    ``CommandContext`` 不像 ``ToolContext`` 有私有属性缓存位，这里每次
    命令调用新建（命令频率低，可接受）。``runtime_tasks`` 从 context
    的 tool_context 取（若存在）。
    """
    registry = BgSessionRegistry()
    runtime_tasks = None
    tc = getattr(context, "tool_context", None)
    if tc is not None:
        runtime_tasks = getattr(tc, "runtime_tasks", None)
    return BgSessionManager(registry=registry, runtime_tasks=runtime_tasks)


def _cmd_list(mgr: BgSessionManager, args: list[str]) -> str:
    include_completed = "--all" in args or "-a" in args
    ws_only = "--workspace" in args or "-w" in args
    ws = _workspace_arg(mgr) if ws_only else None
    sessions = mgr.list_sessions(include_completed=include_completed, workspace_root=ws)
    if not sessions:
        return "No background sessions."
    lines = ["Background sessions:"]
    for s in sessions:
        lines.append(f"  • {s.id}  status={s.status}  pid={s.pid}  ws={s.workspace_root}  started={s.started_at}")
    return "\n".join(lines)


def _workspace_arg(_mgr: BgSessionManager) -> Any:
    return None  # 占位；命令侧用 _workspace(context)


def _workspace(context: Any) -> Path | None:
    for attr in ("workspace_root", "cwd", "working_dir"):
        val = getattr(context, attr, None)
        if val is not None:
            return Path(val) if not isinstance(val, Path) else val
    return None


def _cmd_inspect(mgr: BgSessionManager, args: list[str]) -> str:
    sid = _require_id(args)
    try:
        s = mgr.inspect(sid)
    except BgSessionNotFoundError as exc:
        return f"Not found: {exc}"
    return (
        f"BG session {s.id}\n"
        f"  status:       {s.status}\n"
        f"  pid:          {s.pid}\n"
        f"  workspace:    {s.workspace_root}\n"
        f"  task_id:      {s.task_id}\n"
        f"  team_id:      {s.team_id}\n"
        f"  agent_name:   {s.agent_name}\n"
        f"  transcript:   {s.transcript_path}\n"
        f"  marker:       {s.marker_path}\n"
        f"  started_at:   {s.started_at}\n"
        f"  updated_at:   {s.updated_at}\n"
        f"  completed_at: {s.completed_at}\n"
        f"  error:        {s.error}"
    )


def _cmd_attach(mgr: BgSessionManager, args: list[str], context: Any) -> str:
    sid = _require_id(args)
    all_ws = "--all-ws" in args or "--all" in args
    ws = _workspace(context)
    try:
        result = mgr.attach(sid, follow=True, current_workspace=ws, allow_cross_workspace=all_ws)
    except BgSessionNotFoundError as exc:
        return f"Not found: {exc}"
    except BgSessionPermissionError as exc:
        return f"Permission denied: {exc}\n(retry with --all-ws to allow cross-workspace)"
    except BgSessionAttachError as exc:
        return f"Attach error: {exc}"
    s = result.session
    header = f"Attached to {s.id} (status={s.status}, pid={s.pid})\n{result.resume_hint}\n--- transcript tail ---"
    tail = result.transcript_tail or "(empty or unreadable transcript)"
    return f"{header}\n{tail}"


def _cmd_stop(mgr: BgSessionManager, args: list[str]) -> str:
    sid = _require_id(args)
    force = "--force" in args or "-f" in args
    try:
        s = mgr.stop(sid, force=force)
    except BgSessionNotFoundError as exc:
        return f"Not found: {exc}"
    except BgSessionStopError as exc:
        return f"Stop failed: {exc}\n(retry with --force for SIGKILL)"
    return f"Stopped {s.id} (final status={s.status})."


def _cmd_cleanup(mgr: BgSessionManager, args: list[str]) -> str:
    include_failed = "--failed" in args
    removed = mgr.cleanup(include_failed=include_failed)
    if not removed:
        return "Nothing to clean up."
    lines = [f"Cleaned {len(removed)} session(s) from index:"]
    for s in removed:
        lines.append(f"  • {s.id}  (was {s.status})")
    return "\n".join(lines)


def _cmd_logs(mgr: BgSessionManager, args: list[str]) -> str:
    sid = _require_id(args)
    tail_n = _tail_count(args, default=100)
    try:
        s = mgr.inspect(sid)
    except BgSessionNotFoundError as exc:
        return f"Not found: {exc}"
    from clawcodex_ext.tasks.bg_session_manager import _read_tail  # pylint: disable=no-name-in-module  # 复用

    tail = _read_tail(s.transcript_path, max_lines=tail_n)
    return f"--- {s.id} transcript (tail {tail_n}) ---\n{tail or '(empty)'}"


# ---------------------------------------------------------------------------
# 参数解析辅助
# ---------------------------------------------------------------------------


def _require_id(args: list[str]) -> str:
    for a in args:
        if not a.startswith("-"):
            return a
    raise ValueError("missing <session-id> argument")


def _tail_count(args: list[str], *, default: int) -> int:
    for i, a in enumerate(args):
        if a == "--tail" and i + 1 < len(args):
            try:
                return max(1, int(args[i + 1]))
            except ValueError:
                return default
    return default


# ---------------------------------------------------------------------------
# 主分发
# ---------------------------------------------------------------------------


_SUBCOMMANDS = {
    "list": "list",
    "ls": "list",
    "inspect": "inspect",
    "info": "inspect",
    "attach": "attach",
    "stop": "stop",
    "kill": "stop",
    "cleanup": "cleanup",
    "clean": "cleanup",
    "logs": "logs",
    "log": "logs",
}


def _bg_run(args: str, context: Any) -> LocalCommandResult:
    """``/bg <sub> [args...]`` 主分发。"""
    if not is_bg_sessions_enabled():
        return LocalCommandResult(
            type="text",
            value=(
                "BG_SESSIONS is disabled (CLAWCODEX_BG_SESSIONS=off). "
                "Per-session marker behavior is unchanged; use /tasks for runtime tasks."
            ),
        )
    raw = (args or "").strip()
    if not raw:
        sub, sub_args = "list", []
    else:
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = raw.split()
        if not parts:
            sub, sub_args = "list", []
        else:
            sub, sub_args = parts[0], parts[1:]
    canonical = _SUBCOMMANDS.get(sub)
    if canonical is None:
        return LocalCommandResult(
            type="text",
            value=(f"Unknown /bg subcommand {sub!r}. Valid: list, inspect, attach, stop, cleanup, logs"),
        )
    mgr = _get_manager(context)
    try:
        mgr.registry.scan()
        if canonical == "list":
            text = _cmd_list(mgr, sub_args)
        elif canonical == "inspect":
            text = _cmd_inspect(mgr, sub_args)
        elif canonical == "attach":
            text = _cmd_attach(mgr, sub_args, context)
        elif canonical == "stop":
            text = _cmd_stop(mgr, sub_args)
        elif canonical == "cleanup":
            text = _cmd_cleanup(mgr, sub_args)
        elif canonical == "logs":
            text = _cmd_logs(mgr, sub_args)
        else:  # pragma: no cover
            text = f"Unhandled /bg subcommand: {canonical}"
    except ValueError as exc:
        text = f"/bg {sub}: {exc}"
    except BgSessionsDisabledError as exc:
        text = str(exc)
    except Exception as exc:  # noqa: BLE001 — CLI defensive
        text = f"/bg error: {exc}"
    return LocalCommandResult(type="text", value=text)


# ---------------------------------------------------------------------------
# 命令对象
# ---------------------------------------------------------------------------

BG_COMMAND: LocalCommand = LocalCommand(
    name="bg",
    description="List, inspect, attach, stop, or clean up background agent sessions.",
    aliases=["bgsession", "bgsessions"],
    availability=[CommandAvailability.CONSOLE],
    argument_hint="[list|inspect|attach|stop|cleanup|logs] [session-id] [flags]",
    when_to_use=(
        "Manage background agent sessions created via Ctrl+B or BgSession tool. "
        "Subcommands: list, inspect, attach, stop, cleanup, logs."
    ),
    supports_non_interactive=True,
    user_invocable=True,
)
# LocalCommand 契约：必须 set_call 以通过 test_all_builtins_have_call_impl。
BG_COMMAND.set_call(_bg_run)


# ---------------------------------------------------------------------------
# 注册入口
# ---------------------------------------------------------------------------


def register_bg_commands(registry: Any) -> None:
    """将 ``/bg`` 注册到命令 registry。

    由 ``builtins.register_builtin_commands`` 调用。失败不阻断其他命令注册。
    """
    try:
        registry.register(BG_COMMAND)
    except Exception:  # noqa: BLE001 # nosec B110 — defensive, never break command registration
        pass


__all__ = [
    "BG_COMMAND",
    "register_bg_commands",
]
