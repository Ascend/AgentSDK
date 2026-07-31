"""F-87 /ultraplan command family."""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path

from clawcodex_ext.command_system.engine import CommandContext
from clawcodex_ext.command_system.registry import CommandRegistry
from clawcodex_ext.command_system.types import LocalCommand, LocalCommandResult
from clawcodex_ext.services.ultraplan import PlanStatus
from clawcodex_ext.services.ultraplan.audit import AuditLogger
from clawcodex_ext.services.ultraplan.ccr_session import CCRClient
from clawcodex_ext.services.ultraplan.controller import UltraplanController
from clawcodex_ext.services.ultraplan.exceptions import UltraplanError
from clawcodex_ext.services.ultraplan.executor import PlanExecutor
from clawcodex_ext.services.ultraplan.feature_gates import (
    is_ccr_endpoint_allowed,
    is_ultraplan_llm_enabled,
    is_ultraplan_remote_enabled,
)
from clawcodex_ext.services.ultraplan.llm_planner import LLMPlanner, PlannerContext
from clawcodex_ext.services.ultraplan.planner_recovery import recovery_hint
from clawcodex_ext.services.ultraplan.store import PlanStore
from clawcodex_ext.services.ultraplan.templates import TemplateLibrary


def _usage() -> str:
    return "\n".join(
        [
            "Usage:",
            "  /ultraplan <goal...>",
            "  /ultraplan create [--template id] <goal...>",
            "  /ultraplan run [--remote [endpoint]|--local] [plan_id]",
            "  /ultraplan pause <plan_id>",
            "  /ultraplan resume <plan_id>",
            "  /ultraplan status [plan_id]",
            "  /ultraplan ls [--status status] [--limit n]",
            "  /ultraplan show <plan_id>",
            "  /ultraplan rm <plan_id>",
            "  /ultraplan template list|apply <id> <goal...>",
        ]
    )


def _split(args: str) -> list[str]:
    try:
        return shlex.split(args or "")
    except ValueError as exc:
        raise ValueError(f"invalid arguments: {exc}") from exc


def _take_option(parts: list[str], name: str) -> str | None:
    if name not in parts:
        return None
    index = parts.index(name)
    if index + 1 >= len(parts):
        raise ValueError(f"{name} requires a value")
    value = parts[index + 1]
    del parts[index : index + 2]
    return value


def _flag(parts: list[str], name: str) -> bool:
    if name not in parts:
        return False
    parts.remove(name)
    return True


def _data_dir(context: CommandContext) -> Path:
    configured = context.config.get("ultraplan_dir") if isinstance(context.config, dict) else None
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("CLAWCODEX_ULTRAPLAN_DIR", "~/.clawcodex/ultraplan")).expanduser()


def _controller(
    context: CommandContext, *, need_planner: bool = False, endpoint: str | None = None
) -> UltraplanController:
    root = _data_dir(context)
    store = PlanStore(root / "plans")
    audit = AuditLogger(root / "audit")
    planner = None
    if need_planner:
        if not is_ultraplan_llm_enabled():
            raise RuntimeError("ULTRAPLAN_LLM_PLANNER is disabled")
        provider = getattr(context, "provider", None)
        planner = LLMPlanner(provider, templates=TemplateLibrary(root / "templates"))
    ccr_endpoint = endpoint or os.environ.get("CCR", "")
    if ccr_endpoint and not is_ccr_endpoint_allowed(ccr_endpoint):
        raise RuntimeError("CCR endpoint is not allowed by CCR_ALLOWLIST")
    ccr = CCRClient(ccr_endpoint) if ccr_endpoint else None
    return UltraplanController(planner=planner, store=store, audit=audit, ccr=ccr)


def _latest_plan_id(controller: UltraplanController, plans) -> str:
    if not plans:
        raise ValueError("no ultraplan plans found")
    return sorted(plans, key=lambda p: p.updated_at or p.created_at or p.id)[-1].id


def _format_progress(progress) -> str:
    pct = int(progress.ratio * 100)
    return (
        f"{pct}% complete "
        f"({progress.done}/{progress.total}; pending={progress.pending}, "
        f"in_progress={progress.in_progress}, blocked={progress.blocked}, failed={progress.failed})"
    )


def _format_plan(plan) -> str:
    lines = [f"{plan.id} - {plan.title}", f"status: {plan.status.value}", f"goal: {plan.goal}", ""]
    for sp in plan.sub_plans:
        lines.append(f"[{sp.id}] {sp.title}")
        for step in sp.steps:
            lines.append(f"  - {step.id} [{step.status.value}] {step.title}")
    return "\n".join(lines).rstrip()


class UltraplanCommand(LocalCommand):
    async def call(self, args: str, context: CommandContext) -> LocalCommandResult:
        try:
            value = await _handle(args, context)
            return LocalCommandResult(type="text", value=value)
        except UltraplanError as exc:
            return LocalCommandResult(type="text", value=recovery_hint(exc).message)
        except Exception as exc:  # noqa: BLE001
            return LocalCommandResult(type="text", value=f"ultraplan error: {exc}\n\n{_usage()}")


async def _handle(args: str, context: CommandContext) -> str:
    parts = _split(args)
    if not parts or parts[0] not in {
        "create",
        "run",
        "pause",
        "resume",
        "status",
        "ls",
        "show",
        "rm",
        "template",
        "help",
    }:
        parts.insert(0, "create")
    command = parts.pop(0)
    if command == "help":
        return _usage()
    if command == "create":
        return await _create(parts, context)
    if command == "run":
        return await _run(parts, context)
    if command == "pause":
        return await _pause_resume(parts, context, pause=True)
    if command == "resume":
        return await _pause_resume(parts, context, pause=False)
    if command == "status":
        return await _status(parts, context)
    if command == "ls":
        return await _ls(parts, context)
    if command == "show":
        return await _show(parts, context)
    if command == "rm":
        return await _rm(parts, context)
    if command == "template":
        return _template(parts, context)
    raise ValueError(f"unknown subcommand: {command}")


async def _create(parts: list[str], context: CommandContext) -> str:
    template = _take_option(parts, "--template")
    prompt = " ".join(parts).strip()
    if not prompt:
        raise ValueError("create requires a goal")
    controller = _controller(context, need_planner=True)
    result = await controller.create_plan(
        PlannerContext(
            user_prompt=prompt,
            cwd=str(context.cwd or context.workspace_root),
            template=template,
        )
    )
    return (
        f"Created ultraplan {result.plan.id}: {result.plan.title}\n"
        f"sub_plans={len(result.plan.sub_plans)}, retry_count={result.retry_count}\n"
        f"Run it with /ultraplan run {result.plan.id}"
    )


async def _run(parts: list[str], context: CommandContext) -> str:
    local = _flag(parts, "--local")
    endpoint = None
    remote = False
    if "--remote" in parts:
        index = parts.index("--remote")
        remote = True
        del parts[index]
        if index < len(parts) and not parts[index].startswith("--"):
            endpoint = parts.pop(index)
    if local:
        remote = False
    if remote and not is_ultraplan_remote_enabled():
        raise RuntimeError("ULTRAPLAN_REMOTE is disabled")
    controller = _controller(context, endpoint=endpoint)
    plans = await controller.list_plans()
    plan_id = parts[0] if parts else _latest_plan_id(controller, plans)
    progress = await controller.run_plan(plan_id, remote=remote, cwd=str(context.cwd))
    return f"Ran ultraplan {plan_id}: {_format_progress(progress)}"


async def _pause_resume(parts: list[str], context: CommandContext, *, pause: bool) -> str:
    if len(parts) != 1:
        raise ValueError("pause/resume requires exactly one plan_id")
    controller = _controller(context)
    plan = await (controller.pause_plan(parts[0]) if pause else controller.resume_plan(parts[0]))
    action = "Paused" if pause else "Resumed"
    return f"{action} ultraplan {plan.id}: {plan.status.value}"


async def _status(parts: list[str], context: CommandContext) -> str:
    controller = _controller(context)
    plans = await controller.list_plans()
    plan_id = parts[0] if parts else _latest_plan_id(controller, plans)
    plan = await controller.show_plan(plan_id)
    progress = PlanExecutor(plan).progress()
    return f"{plan.id} - {plan.title}\nstatus: {plan.status.value}\n{_format_progress(progress)}"


async def _ls(parts: list[str], context: CommandContext) -> str:
    status_raw = _take_option(parts, "--status")
    limit_raw = _take_option(parts, "--limit")
    if parts:
        raise ValueError(f"unexpected arguments: {' '.join(parts)}")
    status = PlanStatus(status_raw) if status_raw else None
    limit = int(limit_raw) if limit_raw else 20
    controller = _controller(context)
    plans = (await controller.list_plans(status=status))[:limit]
    if not plans:
        return "No ultraplan plans found."
    lines = ["Ultraplan plans:", ""]
    lines.extend(f"{p.id:18} {p.status.value:10} {p.title}" for p in plans)
    return "\n".join(lines)


async def _show(parts: list[str], context: CommandContext) -> str:
    if len(parts) != 1:
        raise ValueError("show requires exactly one plan_id")
    plan = await _controller(context).show_plan(parts[0])
    return _format_plan(plan)


async def _rm(parts: list[str], context: CommandContext) -> str:
    if len(parts) != 1:
        raise ValueError("rm requires exactly one plan_id")
    await _controller(context).remove_plan(parts[0])
    return f"Removed ultraplan {parts[0]}"


def _template(parts: list[str], context: CommandContext) -> str:
    if not parts:
        raise ValueError("template requires list or apply")
    root = _data_dir(context)
    library = TemplateLibrary(root / "templates")
    action = parts.pop(0)
    if action == "list":
        rows = library.list_templates()
        return "\n".join(["Ultraplan templates:", ""] + [f"{t.id:16} {t.title} - {t.description}" for t in rows])
    if action == "apply":
        if len(parts) < 2:
            raise ValueError("template apply requires <id> <goal...>")
        template_id = parts.pop(0)
        return library.apply(template_id, " ".join(parts))
    raise ValueError(f"unknown template action: {action}")


ULTRAPLAN_COMMAND = UltraplanCommand(
    name="ultraplan",
    description="LLM-driven planning and multi-step execution",
    argument_hint="[create|run|pause|resume|status|ls|show|rm|template] ...",
    supports_non_interactive=True,
)
# LocalCommand consumers inspect this implementation slot when validating the
# registry. Ultraplan overrides ``call`` for its richer error handling, but it
# still publishes the same executable contract as every other local command.
ULTRAPLAN_COMMAND.set_call(ULTRAPLAN_COMMAND.call)


def run_ultraplan_command_sync(args: str, context: CommandContext) -> LocalCommandResult:
    return asyncio.run(ULTRAPLAN_COMMAND.call(args, context))


def register_ultraplan_command(registry: CommandRegistry) -> None:
    registry.register(ULTRAPLAN_COMMAND)
