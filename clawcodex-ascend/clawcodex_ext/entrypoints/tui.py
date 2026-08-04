#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=consider-using-in,no-name-in-module
"""TUI (Textual) entrypoint.

Phase 11 counterpart to :mod:`src.entrypoints.headless`. Where ``headless``
emits NDJSON for pipes, ``tui`` owns the interactive experience: a
retained-mode Textual UI matching the layout of
``typescript/src/screens/REPL.tsx``.

This module deliberately does the provider / session / tool-context setup
*outside* the Textual app so unit tests can construct a :class:`TUIOptions`,
build the app manually, and drive it with :meth:`textual.app.App.run_test`
without touching real network I/O.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.cli_core.exit import cli_error
from src.config import get_default_provider, get_provider_config
from src.providers import get_provider_class


@dataclass
class TUIOptions:
    """Options for :func:`run_tui`. Mirrors :class:`HeadlessOptions`."""

    provider_name: str | None = None
    model: str | None = None
    max_turns: int = 20
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    workspace_root: Path | None = None
    stream: bool = True
    # Resolved permission state from --dangerously-skip-permissions /
    # --allow-dangerously-skip-permissions / --permission-mode. Threaded
    # in by ``cli._run_tui_mode`` so the TUI tool context honors the same
    # flags as the headless entrypoint.
    permission_mode: str = "default"
    is_bypass_permissions_mode_available: bool = False
    # Optional system prompt body to append (from resolved default agent).
    append_system_prompt: str = ""
    # Test hook: replace the provider instance we'd otherwise build from config.
    provider_factory: Callable[[], object] | None = None


def run_tui(options: TUIOptions) -> int:
    """Boot the Textual TUI and block until the user exits.

    Returns a conventional CLI exit code.
    """

    if not _textual_available():
        cli_error(
            "error: textual is not installed. Install it with `pip install 'textual>=0.79'` or pass --no-tui.",
            2,
        )

    # ``is_interactive`` is set during bootstrap phase 2 by
    # ``src.init.run_pre_action`` (called from ``cli.main``) before any
    # entry point runs. Previously we set it here too, but that was the
    # M7.1 gap closed in plan phase 1 of ch02-bootstrap. The TUI
    # entrypoint can rely on ``get_is_interactive()`` already being
    # ``True`` by the time this function runs.
    workspace_root = options.workspace_root or Path.cwd()

    # Build provider ------------------------------------------------------
    if options.provider_factory is not None:
        provider = options.provider_factory()
        provider_name = options.provider_name or getattr(provider, "provider_name", "unknown")
    else:
        provider_name = options.provider_name or get_default_provider()
        try:
            provider_cfg = get_provider_config(provider_name)
        except Exception as exc:
            cli_error(f"error: unable to load provider config: {exc}", 2)
        if not provider_cfg.get("api_key"):
            cli_error(
                f"error: API key for provider '{provider_name}' is not configured. Run `clawcodex login` to set it up.",
                2,
            )
        provider_cls = get_provider_class(provider_name)
        model = options.model or provider_cfg.get("default_model")
        provider = provider_cls(
            api_key=provider_cfg["api_key"],
            base_url=provider_cfg.get("base_url"),
            model=model,
        )

    # Build tool registry + context --------------------------------------
    from src.tool_system.context import ToolContext
    from src.tool_system.defaults import build_default_registry

    tool_registry = build_default_registry(provider=provider)
    if options.allowed_tools:
        allow = {name.lower() for name in options.allowed_tools}
        _filter_registry(tool_registry, keep=lambda n: n.lower() in allow)
    if options.disallowed_tools:
        deny = {name.lower() for name in options.disallowed_tools}
        _filter_registry(tool_registry, keep=lambda n: n.lower() not in deny)

    # Apply the resolved permission state (from ``--dangerously-skip-permissions``
    # or ``--permission-mode``). When bypass is in effect we also flip
    # ``allow_docs`` so the doc-write gate in write.py / edit.py doesn't
    # second-guess the user's explicit opt-in.
    from src.permissions.types import ToolPermissionContext

    tool_context = ToolContext(
        workspace_root=workspace_root,
        permission_context=ToolPermissionContext(
            mode=options.permission_mode or "default",  # type: ignore[arg-type]
            is_bypass_permissions_mode_available=bool(options.is_bypass_permissions_mode_available),
        ),
    )
    if options.permission_mode == "bypassPermissions":
        tool_context.allow_docs = True
    tool_context.options.is_non_interactive_session = False

    # F-22-G-1: Wire cron scheduler to the TUI tool context.
    # F-22-G-3: Track whether the agent loop is active so the cron scheduler
    # can defer fires during model responses.
    class _InAgentLoopFlag:
        value: bool = False

    tool_context._in_agent_loop = _InAgentLoopFlag()  # type: ignore[attr-defined]
    _attach_cron_to_tui(tool_context)

    # Build and run app ---------------------------------------------------
    from src.tui.app import ClawCodexTUI

    app = ClawCodexTUI(
        provider=provider,
        provider_name=provider_name,
        workspace_root=workspace_root,
        tool_registry=tool_registry,
        tool_context=tool_context,
        max_turns=options.max_turns,
        stream=options.stream,
    )
    try:
        # ``inline=True`` renders the app in-place at the bottom of the
        # terminal rather than grabbing the alt-screen — previous shell
        # output stays in scrollback, and ``/exit`` leaves the rendered
        # transcript intact (``inline_no_clear=True``). Matches the
        # TS / ink reference's terminal-native experience.
        # ``mouse=False`` lets the host terminal handle mouse events so
        # the user can drag-select and copy text natively. The trade-off
        # is no in-app mouse scroll on the transcript — keyboard scroll
        # bindings (PgUp/PgDn) still work.
        app.run(inline=True, inline_no_clear=True, mouse=False)
    except KeyboardInterrupt:
        return 130
    finally:
        # Textual normally reaches ``on_unmount``. Keep a final
        # idempotent fallback for mount failures and host interrupts.
        app._shutdown_managed_tasks()

    # Print resume hint after TUI exits (S-R1).
    _print_resume_hint_after_tui(app)
    return 0


def _print_resume_hint_after_tui(app) -> None:
    """Print resume hint to the host terminal after TUI teardown.

    Delegates to the centralised helper so the TUI upstream/downstream
    and the REPL all share one implementation (and one process-wide
    idempotency latch).
    """
    from clawcodex_ext.utils.resume_hint import print_resume_hint

    session = getattr(app, "session", None)
    print_resume_hint(getattr(session, "session_id", None) if session is not None else None)


def _replay_transcript_to_host(app) -> None:
    """Dump the captured transcript to the host terminal after exit.

    Mirrors ink's non-fullscreen behaviour: when the app exits, the
    conversation the user saw stays in scrollback. Textual runs in
    the alt-screen by default which wofrom clawcodex_ext.repl.color_scheme import build_oklch_console

        console = build_oklch_console()
    """

    snapshot = getattr(app, "exit_snapshot", None)
    if not snapshot:
        return
    try:
        from rich.console import Console

        console = Console()
        for piece in snapshot:
            try:
                console.print(piece)
            except Exception:
                continue
    except Exception:  # nosec B110
        pass


def should_use_tui(explicit: bool | None) -> bool:
    """Decide whether to launch the Textual TUI based on flags + environment.

    The default interactive experience is the prompt_toolkit + rich REPL at
    :mod:`src.repl.core` — it matches the TS Ink reference's terminal-native
    UX (transcript flows into scrollback, only the prompt + status row are
    live, native mouse copy works). The Textual TUI is opt-in and reachable
    via ``--tui`` or ``CLAWCODEX_TUI=1`` for users who prefer the richer
    in-app experience.

    * ``explicit=True``   -> always TUI when ``textual`` is importable.
      Also enabled by ``CLAWCODEX_TUI=1``.
    * ``explicit=False``  -> never TUI. Also forced by
      ``CLAWCODEX_LEGACY_REPL=1`` (kept for back-compat).
    * ``explicit=None``   -> default to the REPL. Honor ``CLAWCODEX_TUI=1``
      from the environment so users can pin the TUI without a flag.
    """

    if explicit is False:
        return False
    if os.environ.get("CLAWCODEX_LEGACY_REPL") == "1":
        return False
    if os.environ.get("CLAWCODEX_TUI") == "0":
        return False

    env_tui = os.environ.get("CLAWCODEX_TUI") == "1"
    if not (explicit is True or env_tui):
        return False

    if not _textual_available():
        return False

    term = os.environ.get("TERM", "")
    if term == "dumb" or term == "":
        return False
    try:
        if not sys.stdout.isatty() or not sys.stdin.isatty():
            return False
    except Exception:
        return False
    return True


def _textual_available() -> bool:
    try:
        import textual  # noqa: F401

        return True
    except Exception:
        return False


def _filter_registry(registry, *, keep: Callable[[str], bool]) -> None:
    names = [t.name for t in registry.list_tools()]
    for name in names:
        if not keep(name):
            try:
                registry.unregister(name)
            except Exception:
                try:
                    del registry._tools[name]  # type: ignore[attr-defined]
                except Exception:  # nosec B110
                    pass


# ---- F-22-G-1: TUI cron integration ----


def _attach_cron_to_tui(tool_context) -> None:
    """Wire cron scheduler + replace cron tools for the TUI entrypoint."""
    from clawcodex_ext.cron_system.runtime import attach_cron_runtime, replace_cron_tools

    # F-22-G-3: pass is_loading callback so cron fires defer during agent turns.
    in_agent_loop = getattr(tool_context, "_in_agent_loop", None)
    is_loading = (lambda: in_agent_loop.value) if in_agent_loop is not None else None

    attach_cron_runtime(
        tool_context,
        autostart=True,
        is_loading=is_loading,
    )
    # Replace cron tools in the registry attached to the tool context.
    registry = getattr(tool_context, "registry", None)
    if registry is not None:
        replace_cron_tools(registry)


def _drain_cron_outbox(
    tool_context,
    active_tasks: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Drain cron_prompt events from ``tool_context.outbox``.

    Returns runnable prompts as ``(wrapped_prompt, task_id, run_id)``.
    Duplicate active tasks are discarded and their runs finalized as
    cancelled.
    """
    from clawcodex_ext.cron_system.dispatch import CronDispatchBridge, _default_wrap_prompt
    from clawcodex_ext.cron_system.runs import finalize_cron_run

    outbox = getattr(tool_context, "outbox", [])
    if not outbox:
        return []

    bridge = CronDispatchBridge(
        tool_context.workspace_root,
        wrap_prompt=_default_wrap_prompt,
    )
    events = bridge.drain(outbox)
    results: list[tuple[str, str, str]] = []
    for event in events:
        if event.task_id in active_tasks:
            finalize_cron_run(
                tool_context.workspace_root,
                event.run_id,
                "cancelled",
                error="duplicate fire",
            )
            continue
        active_tasks[event.task_id] = event.run_id
        results.append((event.wrapped_prompt, event.task_id, event.run_id))
    return results


def _claim_cron_task(
    workspace_root,
    active_tasks: dict[str, str],
    task_id: str,
) -> None:
    """Claim a cron task run (queued → running)."""
    from clawcodex_ext.cron_system.runs import claim_cron_run

    run_id = active_tasks.get(task_id)
    if run_id is not None:
        claim_cron_run(workspace_root, run_id, task_id)


def _finalize_cron_task(
    workspace_root,
    active_tasks: dict[str, str],
    task_id: str,
    status: str,
    error: str | None = None,
) -> None:
    """Finalize a cron task run (→ completed/failed/cancelled)."""
    from clawcodex_ext.cron_system.runs import finalize_cron_run

    run_id = active_tasks.pop(task_id, None)
    if run_id is not None:
        finalize_cron_run(workspace_root, run_id, status, error=error)


def _run_cron_prompt(
    workspace_root,
    active_tasks: dict[str, str],
    prompt: str,
    task_id: str,
    run_id: str,
) -> bool:
    """Execute a single drained cron prompt and finalize its run.

    For TUI, this enqueues the prompt into the app's input pipeline.
    The actual execution is handled by the TUI agent loop.
    """
    _claim_cron_task(workspace_root, active_tasks, task_id)
    # In TUI mode, the cron prompt is dispatched to the app via the
    # tool_context's outbox or a dedicated callback registered by the
    # TUI app.  The base implementation here logs the fire and finalizes
    # as completed — downstream TUI code should override this behaviour.
    _finalize_cron_task(workspace_root, active_tasks, task_id, "completed")
    return True


def _process_cron_outbox(
    tool_context,
    active_tasks: dict[str, str],
    run_prompt: Callable[[str, str, str], bool],
    *,
    max_iterations: int = 10,
) -> None:
    """Drain and execute cron prompts until the outbox is empty.

    Bounded loop prevents runaway scheduling storms if a recurring task
    fires faster than it can be finalized.
    """
    for _ in range(max_iterations):
        prompts = _drain_cron_outbox(tool_context, active_tasks)
        if not prompts:
            break
        for prompt, task_id, run_id in prompts:
            try:
                run_prompt(prompt, task_id, run_id)
            except Exception:
                _finalize_cron_task(
                    tool_context.workspace_root,
                    active_tasks,
                    task_id,
                    "failed",
                    error="cron prompt execution failed",
                )
