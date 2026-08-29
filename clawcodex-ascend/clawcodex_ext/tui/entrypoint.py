#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


"""Downstream TUI entrypoint — extended version with session resume support.

Called by :class:`clawcodex_ext.frontend.tui.TUIFrontend.run` which passes
the full runtime context (provider, session, tool registry, etc.) so the
TUI can resume an existing session and print a resume hint on exit.
"""

from __future__ import annotations
# pylint: disable=E0611

import logging
import os
import time
from pathlib import Path
from typing import Any

from src.agent import Session
from src.entrypoints.tui import (
    TUIOptions,
    textual_available as _textual_available,
)
from src.tool_system.context import ToolContext
from src.tool_system.registry import ToolRegistry

from clawcodex_ext.tui.app import ClawCodexExtTUI

_LOGGER = logging.getLogger(__name__)


def run_tui(
    options: TUIOptions,
    *,
    provider: Any | None = None,
    session: Session | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContext | None = None,
    runtime_context: Any | None = None,
    resume_session_id: str | None = None,
    resume_browse: bool = False,
    append_system_prompt: str = "",
) -> int:
    return _run_tui_with_app(
        options,
        app_cls=ClawCodexExtTUI,
        provider=provider,
        session=session,
        tool_registry=tool_registry,
        tool_context=tool_context,
        runtime_context=runtime_context,
        resume_session_id=resume_session_id,
        resume_browse=resume_browse,
        append_system_prompt=append_system_prompt,
    )


def _run_tui_with_app(
    options: TUIOptions,
    *,
    app_cls: type[ClawCodexExtTUI] = ClawCodexExtTUI,
    provider: Any | None = None,
    session: Session | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContext | None = None,
    runtime_context: Any | None = None,
    resume_session_id: str | None = None,
    resume_browse: bool = False,
    append_system_prompt: str = "",
) -> int:
    """Boot the Textual TUI with extended options and print resume hint on exit.

    Args:
        options: Base TUI options (provider name, model, permissions, …).
        provider: Pre-built provider instance (optional).
        session: Pre-built session to use (optional).
        tool_registry: Pre-built tool registry (optional).
        tool_context: Pre-built tool context (optional).
        runtime_context: RuntimeContext from CLI dispatch (optional).
        resume_session_id: Session ID to resume (optional).
        resume_browse: If True, show the session browser on startup (default False).
        append_system_prompt: Extra system prompt text to append.

    Returns:
        CLI exit code (0 on success, 130 on KeyboardInterrupt).
    """
    if not _textual_available():
        from src.cli_core.exit import cli_error

        cli_error(
            "error: textual is not installed. Install it with `pip install 'textual>=0.79'` or pass --no-tui.",
            2,
        )

    workspace_root = options.workspace_root or Path.cwd()

    # Build provider if not injected
    if provider is None:
        from src.config import get_default_provider, get_provider_config
        from src.providers import get_provider_class

        provider_name = options.provider_name or get_default_provider()
        try:
            provider_cfg = get_provider_config(provider_name)
        except Exception as exc:
            from src.cli_core.exit import cli_error

            cli_error(f"error: unable to load provider config: {exc}", 2)
        if not provider_cfg.get("api_key"):
            from src.cli_core.exit import cli_error

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
    else:
        provider_name = options.provider_name or getattr(provider, "provider_name", "unknown")

    # Build tool registry + context if not injected
    if tool_registry is None:
        from src.tool_system.defaults import build_default_registry

        tool_registry = build_default_registry(provider=provider)
        if options.allowed_tools:
            allow = {name.lower() for name in options.allowed_tools}
            _filter_registry(tool_registry, keep=lambda n: n.lower() in allow)
        if options.disallowed_tools:
            deny = {name.lower() for name in options.disallowed_tools}
            _filter_registry(tool_registry, keep=lambda n: n.lower() not in deny)

    if tool_context is None:
        from src.permissions.types import ToolPermissionContext

        tool_context = ToolContext(
            workspace_root=workspace_root,
            permission_context=ToolPermissionContext(
                mode=options.permission_mode or "default",
                is_bypass_permissions_mode_available=bool(options.is_bypass_permissions_mode_available),
            ),
        )
        if options.permission_mode == "bypassPermissions":
            tool_context.allow_docs = True
        tool_context.options.is_non_interactive_session = False

    model_label = getattr(provider, "model", "")

    # Build session: resume or create
    resumed_existing = False
    if session is not None:
        used_session = session
        resumed_existing = bool(resume_session_id and str(getattr(session, "session_id", "")) == str(resume_session_id))
    elif resume_session_id:
        loaded = Session.resume(resume_session_id)
        if loaded is not None:
            used_session = loaded
            resumed_existing = True
        else:
            from clawcodex_ext.repl.color_scheme import build_oklch_console

            console = build_oklch_console()
            console.print(f"[warning]Session not found: {resume_session_id}. Starting new session.[/warning]")
            used_session = Session.create(provider_name, model_label)
    else:
        used_session = Session.create(provider_name, model_label)

    # Build and run app
    app = app_cls(
        provider=provider,
        provider_name=provider_name,
        workspace_root=workspace_root,
        tool_registry=tool_registry,
        tool_context=tool_context,
        session=used_session,
        max_turns=options.max_turns,
        stream=options.stream,
        resume_browse=resume_browse,
        runtime_context=runtime_context,
        append_system_prompt=append_system_prompt,
        session_was_resumed=resumed_existing,
    )

    # ---- SIGTERM/SIGINT: save session + print resume hint via graceful shutdown (S-R1) ----
    _register_tui_signal_save(used_session)

    # best-effort session_start. The session id is the same one
    # the conversation persists under so the per-day aggregator can
    # cross-link events to a known session. Failures are swallowed.
    try:
        from telemetry import record_session_start

        record_session_start(
            session_id=used_session.session_id,
            entrypoint="tui",
            client_type=os.environ.get("CLAUDE_CODE_ENTRYPOINT", "cli"),
            is_non_interactive=False,
        )
    except Exception:  # nosec B110
        pass  # Telemetry is optional and must not affect command execution.

    tui_start = time.monotonic()
    exit_code = 0
    try:
        app.run()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        # best-effort error event with stable fingerprint.
        # Failures are swallowed.
        try:
            from telemetry import record_error

            record_error(session_id=used_session.session_id, exc=exc)
        except Exception:  # nosec B110
            pass  # Telemetry is optional and must not affect command execution.
        raise
    finally:
        # best-effort session_end + command_run. Telemetry
        # must never block the user's exit.
        try:
            from telemetry import (
                record_command_run,
                record_session_end,
            )

            duration_s = time.monotonic() - tui_start
            record_session_end(
                session_id=used_session.session_id,
                duration_s=duration_s,
                exit_status=exit_code,
            )
            record_command_run(
                session_id=used_session.session_id,
                command_name="tui",
                mode="interactive",
                success=(exit_code == 0),
                duration_s=duration_s,
                exit_status=exit_code,
            )
        except Exception:  # nosec B110
            pass  # Telemetry is optional and must not affect command execution.

    # --- After TUI exits, print resume hint ---
    _print_resume_hint(used_session)
    return exit_code


def _register_tui_signal_save(session: Session) -> None:
    """Register graceful-shutdown cleanup that saves the TUI session and
    prints a resume hint when the process receives SIGTERM/SIGINT.

    The print delegates to :func:`clawcodex_ext.utils.resume_hint.print_resume_hint`
    so the helper's process-wide latch keeps the hint to a single
    emission even if the inline TUI teardown path also fires one.
    """
    try:
        from src.utils.graceful_shutdown import register_cleanup
    except ImportError:
        return

    def _cleanup() -> None:
        try:
            session.save()
        except Exception:  # nosec B110
            pass  # Session persistence is best-effort at this recovery or shutdown boundary.
        try:
            from clawcodex_ext.utils.resume_hint import print_resume_hint

            print_resume_hint(getattr(session, "session_id", None))
        except Exception:  # nosec B110
            pass  # Resume-hint output is best-effort during teardown.

    register_cleanup(_cleanup)


def _print_resume_hint(session: Session) -> None:
    """Print resume hint to the host terminal after TUI teardown.

    Delegates to the centralised helper so the TUI upstream/downstream
    and the REPL all share one implementation (including the
    process-wide idempotency latch that suppresses atexit double-fires).
    """
    from clawcodex_ext.utils.resume_hint import print_resume_hint

    print_resume_hint(getattr(session, "session_id", None))


def _filter_registry(registry, *, keep) -> None:
    """Filter tools in a registry by a predicate."""
    names = [t.name for t in registry.list_tools()]
    for name in names:
        if not keep(name):
            try:
                registry.unregister(name)
            except Exception:
                try:
                    del registry._tools[name]
                except Exception as exc:  # nosec B110
                    _LOGGER.warning(
                        "Failed to remove filtered tool %s (%s)",
                        name,
                        type(exc).__name__,
                    )


__all__ = ["run_tui", "_run_tui_with_app"]
