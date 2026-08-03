"""Runtime glue for downstream Cron tools and scheduler (F-22-G1 + G4)."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import CronTask, is_cron_disabled, load_jitter_config
from .runs import CronRun
from .scheduler import CronScheduler
from .tools import CronCreateTool, CronDeleteTool, CronListTool, CronRunTool
from clawcodex_ext.query.outbox_types import CronMissedEvent, CronPromptEvent  # pylint: disable=no-name-in-module

_log = logging.getLogger(__name__)

_CRON_TOOL_NAMES = {"croncreate", "crondelete", "cronlist", "cronrun"}


def replace_cron_tools(registry: Any) -> None:
    """Replace any existing cron tools in the registry with the extension versions.

    This function directly reads and writes the registry's private ``_tools``
    and ``_by_name`` attributes because ``ToolRegistry`` does not expose a
    public "remove by name" API. The ``getattr`` guards tolerate test fakes
    that lack these attributes. If ``ToolRegistry``'s internal structure
    changes (e.g. switches to a dict or adds a secondary index), this
    function must be updated in tandem.
    """
    tools = getattr(registry, "_tools", None)
    by_name = getattr(registry, "_by_name", None)
    if tools is not None:
        registry._tools = [tool for tool in tools if tool.name.lower() not in _CRON_TOOL_NAMES]
    if by_name is not None:
        for name in list(by_name.keys()):
            tool = by_name[name]
            if tool.name.lower() in _CRON_TOOL_NAMES:
                del by_name[name]
    for tool in (CronCreateTool, CronListTool, CronDeleteTool, CronRunTool):
        try:
            registry.register(tool)
        except Exception as exc:  # nosec B110 - ignore duplicate tool registration
            _log.warning("failed to register cron tool %s: %s", tool.__name__, exc)


def attach_cron_runtime(
    ctx: Any,
    *,
    autostart: bool = False,
    is_killed: Any | None = None,
    is_loading: Callable[[], bool] | None = None,
    assistant_mode: bool = False,
    asciicast_observer: Any | None = None,
    agent_id: str | None = None,  # F-22-F: agent ownership
) -> CronScheduler:
    """Wire Cron tools + scheduler to a session context.

    ``is_killed`` is the F-22-G1 kill switch. When None, falls back to
    ``is_cron_disabled`` (reads ``CLAWCODEX_DISABLE_CRON``). When provided,
    it takes precedence — daemon callers can pass a GrowthBook-style flag.

    ``is_loading`` is polled before each scheduler tick. When it returns
    True (and ``assistant_mode`` is False), the entire tick is skipped —
    cron fires are deferred until the agent is idle. Mirrors TS
    ``isLoading`` gate in ``cronScheduler.ts``.

    ``assistant_mode`` bypasses the ``is_loading`` gate so cron fires
    proceed even while the agent is busy. Used by assistant/daemon
    sub-modes where cron must not be starved.

    ``agent_id`` (F-22-F) sets the owning agent identity. When provided,
    the scheduler only fires tasks belonging to this agent or global tasks
    (agent_id=None). Pass None for single-agent mode (no filtering).
    """
    if is_killed is None:
        is_killed = is_cron_disabled

    outbox = getattr(ctx, "outbox", None)
    if outbox is None:
        outbox = []
        setattr(ctx, "outbox", outbox)

    def on_fire(prompt: str) -> None:
        if is_killed():
            return
        outbox.append(CronPromptEvent(prompt=prompt))

    def on_fire_task(task: CronTask, run: CronRun) -> None:
        if is_killed():
            return
        outbox.append(
            CronPromptEvent(
                prompt=task.prompt,
                task_id=task.id,
                run_id=run.id,
            )
        )

    def on_missed(tasks: list[CronTask], notification: str) -> None:
        if is_killed():
            return
        outbox.append(
            CronMissedEvent(
                tasks=[task.id for task in tasks],
                notification=notification,
            )
        )

    # F-22-G7: opt-in observability sink — by default just logs at debug.
    def _log_event(payload: dict) -> None:
        _log.debug("cron event: %s", payload)

    # F-REC: when an asciicast observer is wired, mirror its four
    # callbacks into the scheduler in addition to the debug logger so
    # cron fires land in the recording's .cast file. When the observer
    # is None (the common case), the existing debug logger is the only
    # sink.
    if asciicast_observer is not None:
        on_fire_event = getattr(asciicast_observer, "on_fire_event", _log_event)
        on_missed_event = getattr(asciicast_observer, "on_missed_event", _log_event)
        on_expired_event = getattr(asciicast_observer, "on_expired_event", _log_event)
    else:
        on_fire_event = _log_event
        on_missed_event = _log_event
        on_expired_event = _log_event

    # F-22-G2: the scheduler hot-loads the jitter config on every
    # ``check_once`` tick. Threading the loader through ctx.cron_jitter_config
    # (if present) lets REPL callers inject a GrowthBook-style remote source.
    config_loader = getattr(ctx, "cron_jitter_config", None)

    # Phase A-3: pass the in-memory session store to the scheduler so
    # durable=False tasks are also discoverable. ``ctx.crons`` is set
    # upstream by ToolContext; it is None for contexts that never hold
    # session-only tasks (e.g. headless ``-p`` one-shot runs).
    session_store = getattr(ctx, "crons", None)

    scheduler = CronScheduler(
        ctx.workspace_root,
        on_fire=on_fire,
        on_fire_task=on_fire_task,
        on_missed=on_missed,
        is_killed=is_killed,
        load_jitter_config=config_loader,
        on_fire_event=on_fire_event,
        on_missed_event=on_missed_event,
        on_expired_event=on_expired_event,
        session_store=session_store,
        is_loading=is_loading,
        assistant_mode=assistant_mode,
        agent_id=agent_id,  # F-22-F
    )
    setattr(ctx, "cron_scheduler", scheduler)
    setattr(ctx, "cron_jitter_config", lambda: load_jitter_config(ctx.workspace_root))
    if autostart:
        scheduler.start()
    return scheduler


def install_permanent_cron_tasks(
    workspace_root: Any,
    tasks: list[dict],
) -> list[tuple[Any, bool]]:
    """F-22-G4 installer entry point.

    ``tasks`` is a list of dicts with keys: ``cron``, ``prompt``,
    optional ``recurring`` (default True), ``jitter`` (CronJitterConfig),
    ``created_at`` (epoch ms), ``task_id`` (8-hex string).

    Returns a list of ``(task, created)`` tuples — same shape as
    :func:`clawcodex_ext.cron_system.tasks.write_permanent_task_if_missing`.
    Used by the assistant-mode installer to seed catch-up / morning-checkin
    / dream tasks.
    """
    # Local import to avoid circular import: tasks.py imports from .jitter
    # which imports from .models.
    from .tasks import write_permanent_task_if_missing

    results: list[tuple[Any, bool]] = []
    for spec in tasks:
        try:
            result = write_permanent_task_if_missing(
                workspace_root,
                cron=spec["cron"],
                prompt=spec["prompt"],
                recurring=spec.get("recurring", True),
                jitter=spec.get("jitter"),
                created_at=spec.get("created_at"),
                task_id=spec.get("task_id"),
            )
        except PermissionError as exc:
            _log.warning("skipping permanent install: %s", exc)
            continue
        results.append(result)
    return results
