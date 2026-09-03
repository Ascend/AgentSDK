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

# pylint: disable=function-redefined,no-name-in-module,too-many-lines,too-many-nested-blocks,ungrouped-imports,unnecessary-dunder-call
"""Headless (non-interactive) entrypoint.

Port of ``typescript/src/cli/print.ts``, scoped to the slice that matters for
Phase 1: run a single prompt (or a stream of prompts via stream-json stdin)
through the agent loop and emit the response in the requested output format.

The heavy lifting lives in :mod:`src.query.query` (the canonical agent
loop), driven via the sync wrapper
:func:`src.query.agent_loop_compat.run_query_as_agent_loop`. That loop
already understands Anthropic + OpenAI-compatible providers and emits
structured tool events; this module adapts those events to the CLI
protocol in :mod:`src.cli_core`.

Design notes
------------
* No Rich / prompt_toolkit imports — headless mode must run on plain pipes
  (CI, SDK clients, tests) without a TTY.
* Tool permission handling is driven by ``--dangerously-skip-permissions``:
  when set, tools run without gating; otherwise the default ``ToolContext``
  mode (``bypassPermissions``) still applies but *interactive* permission
  prompts auto-deny — we never ``input()`` in headless mode.
* The agent loop is synchronous; we call it inside ``run_headless`` and
  translate events to NDJSON on the fly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal as _signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Iterable

from src.agent import Session
from src.cli_core import (
    AssistantEvent,
    PartialTextEvent,
    ResultEvent,
    StreamJsonReader,
    StreamJsonWriter,
    SystemEvent,
    ToolResultEvent,
    ToolUseEvent,
    UserInputMessage,
    cli_error,
    ndjson_safe_dumps,
)
from src.config import get_default_provider, get_provider_config
from src.providers import (
    get_provider_class,
    provider_requires_api_key,
    resolve_api_key,
)
from src.tool_system.renderers import AgentLoopResult, ToolEvent
from clawcodex_ext.command_system.builtins import (
    execute_command_sync,
    get_builtin_commands,
)
from clawcodex_ext.command_system.engine import CommandEngine, create_command_context
from clawcodex_ext.command_system.input_processing import parse_user_input
from clawcodex_ext.command_system.registry import CommandRegistry, get_command_registry
from clawcodex_ext.command_system.types import LocalCommand, PromptCommand
from clawcodex_ext.cron_system.runtime import attach_cron_runtime, replace_cron_tools
from clawcodex_ext.cron_system.runs import claim_cron_run, finalize_cron_run
from clawcodex_ext.query.agent_loop_compat import (
    build_effective_system_prompt,
    run_query_as_agent_loop,
)
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.utils.abort_controller import AbortController, AbortError

_LOGGER = logging.getLogger(__name__)

OUTPUT_FORMATS = ("text", "json", "stream-json")
INPUT_FORMATS = ("text", "stream-json")


@dataclass
class HeadlessOptions:
    """Options accepted by :func:`run_headless`.

    Kept as a plain dataclass (no Click/argparse coupling) so the CLI layer
    and tests can construct it independently.
    """

    prompt: str | None = None
    output_format: str = "text"
    input_format: str = "text"
    provider_name: str | None = None
    provider_instance: Any | None = None
    model: str | None = None
    fallback_model: str | None = None
    effort: str | None = None
    max_turns: int = 20
    # Goal mode is intentionally unbounded unless the CLI user supplied
    # --max-turns.  A finite default can stop an unmet goal and then report
    # success, which violates /goal's keep-going contract.
    max_turns_explicit: bool = False
    # ``skip_permissions`` is a backward-compat alias for the boolean form
    # of ``--dangerously-skip-permissions``. ``permission_mode`` and
    # ``is_bypass_permissions_mode_available`` were added in round 5 to
    # mirror the TS reference's resolved state. When ``skip_permissions``
    # is True we treat it as ``permission_mode='bypassPermissions'`` and
    # ``is_bypass_permissions_mode_available=True``.
    skip_permissions: bool = False
    permission_mode: str = "default"
    is_bypass_permissions_mode_available: bool = False
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    include_partial_messages: bool = False
    verbose: bool = False

    # Mostly for tests: override streams so we can capture output.
    stdin: IO[str] | None = None
    stdout: IO[str] | None = None
    stderr: IO[str] | None = None

    # Workspace root override (default: cwd).
    workspace_root: Path | None = None

    # Optional system prompt body to append (from resolved default agent).
    append_system_prompt: str = ""
    startup_agent: Any | None = None
    bundle_context: Any | None = None

    # MCP runtime state forwarded from RuntimeContext so headless can call
    # connected MCP servers without re-bootstrapping them.
    mcp_clients: dict[str, Any] = field(default_factory=dict)
    mcp_manager_loop: Any | None = None

    # Environment variables merged into every Bash subprocess env.
    # Values override inherited daemon env.
    env: dict[str, str] = field(default_factory=dict)

    # External tool-event callback (orchestrator's QueryRunner wires this).
    # Called alongside the internal NDJSON writer when set.
    on_event: Callable[[Any], None] | None = None

    # External abort controller. When provided, cooperative cancellation is
    # driven by the CALLER (e.g. QueryRunner's wall-clock budget) in
    # addition to SIGINT. Without it a timed-out orchestrator run keeps
    # executing on its executor thread — spawning workers and burning
    # tokens — because nothing outside can reach the internal controller.
    abort_controller: Any | None = None

    # resume / fork support. Three pieces:
    # * ``resume_session_id`` — load history from this session and
    #   reuse its session_id (the next ``--resume <sid>`` invocation
    #   sees the new messages too).
    # * ``fork_session_id`` — load history but mint a fresh session_id
    #   so the new conversation branches off.
    # * ``resume_session_at`` — truncate the loaded conversation to
    #   the given message index before resuming (0-based, inclusive).
    # * ``external_session`` — when ``RuntimeContext.build()`` has
    #   already produced a session (the canonical headless path),
    #   pass it in here so we skip the ``Session.create()`` /
    #   ``Session.resume()`` branch. Mutually exclusive with the
    #   three IDs above; the frontend layer wires it for us.
    resume_session_id: str | None = None
    fork_session_id: str | None = None
    resume_session_at: int | None = None
    external_session: Any | None = None
    # ``persist_on_exit`` defaults to True so headless writes the
    # accumulated transcript back to disk at the end of the run — the
    # minimum bar for ``--resume`` to actually accumulate history
    # across runs. Set to False in tests that
    # want to exercise the in-memory path only.
    persist_on_exit: bool = True

    # F-REC-H: headless asciicast recording support.
    record: str | None = None
    record_width: int | None = None
    record_height: int | None = None
    capture: Any | None = None

    # agent identity for pending_messages drain.
    # When set, ToolContext gets this agent_id + runtime_tasks so
    # _drain_pending_user_messages fires at ToolResult boundaries.
    agent_id: str | None = None
    runtime_tasks: Any | None = None


def run_headless(options: HeadlessOptions) -> int:
    """Run one or more prompts in headless mode. Returns the exit code."""

    # F-REC-H: open the side-channel recorder before anything else. The
    # recorder is responsible for closing itself; ``options.capture`` is
    # injected so _run_one_agent_loop can bridge tool/text events.
    recorder: Any | None = None
    if options.record:
        from extensions.recording.headless_source import open_headless_recorder

        recorder = open_headless_recorder(
            options.record,
            width=options.record_width,
            height=options.record_height,
            command=f"clawcodex --record {options.record} -p ...",
        )
        capture = recorder.__enter__()
        if capture is not None:
            options.capture = recorder

    try:
        return _run_headless_core(options)
    finally:
        if options.capture is not None:
            recorder.__exit__(None, None, None)


def _run_headless_core(options: HeadlessOptions) -> int:
    """Original ``run_headless`` body, split so recording can wrap it."""

    # start the opt-in freeze-detection watchdog when the
    # env var is set. Headless runs are the primary failure-mode
    # targets (provider streams that stop emitting chunks, tools that
    # hang forever). Placed at the top of the entry point so direct API
    # callers (e.g. ``extensions.api.query.QueryRunner`` via
    # ``run_headless_session``) also benefit without depending on the
    # CLI bootstrap path that calls ``init()``.
    try:
        from clawcodex_ext.diagnostics import FreezeDetector

        FreezeDetector.maybe_start_from_env()
    except Exception:  # nosec B110
        pass  # Freeze diagnostics are optional and must not block headless startup.

    if options.output_format not in OUTPUT_FORMATS:
        cli_error(f"error: --output-format must be one of {', '.join(OUTPUT_FORMATS)}", 2)
    if options.input_format not in INPUT_FORMATS:
        cli_error(f"error: --input-format must be one of {', '.join(INPUT_FORMATS)}", 2)
    if options.input_format == "stream-json" and options.output_format != "stream-json":
        cli_error(
            "error: --input-format stream-json requires --output-format stream-json",
            2,
        )

    stdout = options.stdout or sys.stdout
    stderr = options.stderr or sys.stderr
    stdin = options.stdin or sys.stdin

    # ch02 round-3 GAP B: warm the user/system context memos now so the
    # CLAWCODEX.md walk and git probes overlap with provider + registry
    # construction below instead of running inside the first turn.
    # Mirrors TS main.tsx:1973-1990 (non-interactive early kicks; trust
    # is implicit in -p mode and was granted by run_pre_action).
    # MUST use the resolved workspace_root, not the process cwd — the
    # memos are key-less and first-writer pins the content the query
    # path (which passes workspace_root) will read.
    workspace_root = options.workspace_root or Path.cwd()
    from src.deferred_init import start_deferred_prefetches

    start_deferred_prefetches(cwd=str(workspace_root))

    provider_free_exit = _try_run_provider_free_goal_summary(
        options,
        workspace_root=workspace_root,
        stdout=stdout,
        stderr=stderr,
    )
    if provider_free_exit is not None:
        return provider_free_exit

    provider_name = options.provider_name or get_default_provider()
    provider = options.provider_instance
    if provider is None:
        # Keep every entrypoint aligned on provider validation and its
        # user-facing errors. Direct tests/integrations may inject a ready
        # provider instance and intentionally bypass config validation.
        from src.entrypoints.provider_validation import validate_provider_at_startup

        try:
            provider_cfg = get_provider_config(provider_name)
        except Exception:
            validate_provider_at_startup(provider_name, interactive=False, exit_code=2)
            raise  # pragma: no cover - validator exits on failure
        api_key = resolve_api_key(provider_name, provider_cfg)
        if not api_key and provider_requires_api_key(provider_name):
            validate_provider_at_startup(provider_name, interactive=False, exit_code=2)
        provider_cls = get_provider_class(provider_name)
        model = options.model or provider_cfg.get("default_model")
        provider = provider_cls(api_key=api_key, base_url=provider_cfg.get("base_url"), model=model)
    else:
        model = options.model or getattr(provider, "model", None)

    # session assembly. Three input sources, in priority
    # order:
    #   1. ``options.external_session`` — the canonical headless path
    #      via ``RuntimeContext.build()`` already produced a session
    #      (resume / fork / create handled upstream). Reuse it so the
    #      session_id stays consistent with telemetry and downstream
    #      ``--resume`` calls.
    #   2. ``options.resume_session_id`` — direct ``Session.resume()``
    #      for legacy callers (e.g. ``run_print_mode`` in
    #      ``clawcodex_ext/cli/runners.py``) that bypass RuntimeContext.
    #      The session_id is reused so subsequent ``--resume`` sees
    #      the newly appended messages.
    #   3. Fresh ``Session.create()`` — unchanged behaviour.
    if options.external_session is not None:
        session = options.external_session
    elif options.resume_session_id:
        try:
            session = Session.resume(options.resume_session_id)
        except Exception as exc:
            cli_error(
                f"error: failed to resume session '{options.resume_session_id}': {exc}",
                2,
            )
        if session is None:
            cli_error(
                f"error: no session found with ID '{options.resume_session_id}'",
                2,
            )
    else:
        session = Session.create(provider_name, getattr(provider, "model", model or ""))

    # ``resume_session_at`` truncation. Run after session
    # assembly so it applies uniformly to both ``external_session`` and
    # direct ``Session.resume()`` paths. ``fork_session_id`` always
    # mints a fresh ID (RuntimeContext handles that — this branch only
    # fires when external_session is None and resume_session_id is None
    # but fork_session_id is set, which is the rare standalone fork
    # path).
    if options.fork_session_id and options.external_session is None and options.resume_session_id is None:
        old = Session.resume(options.fork_session_id)
        if old is None:
            cli_error(
                f"error: no session found with ID '{options.fork_session_id}'",
                2,
            )
        new_session = Session.create(provider_name, getattr(provider, "model", model or ""))
        if old.conversation and old.conversation.messages:
            new_session.conversation.messages = list(old.conversation.messages)
        session = new_session

    if options.resume_session_at is not None and session is not None:
        idx = options.resume_session_at
        if session.conversation and session.conversation.messages:
            total = len(session.conversation.messages)
            if 0 <= idx < total:
                session.conversation.messages = session.conversation.messages[: idx + 1]
            elif idx < 0:
                cli_error(
                    f"error: --resume-session-at index {idx} is negative",
                    2,
                )
            else:
                cli_error(
                    f"error: --resume-session-at index {idx} out of range (session has {total} messages)",
                    2,
                )

    # best-effort session_start. The session id is the same one
    # the conversation persists under so the per-day aggregator can
    # cross-link events to a known session. Failures are swallowed.
    try:
        from telemetry import record_session_start

        record_session_start(
            session_id=session.session_id,
            entrypoint="headless",
            client_type=os.environ.get("CLAUDE_CODE_ENTRYPOINT", "cli"),
            is_non_interactive=True,
        )
    except Exception:  # nosec B110
        pass  # Telemetry is optional and must not affect command execution.

    # resume-time checks (C8 / C11 / R8). All best-effort
    # and non-fatal — a missing metadata file or unreadable transcript
    # is silently skipped. Warnings go to stderr so structured stdout
    # output stays clean. Only fires when a session was actually
    # resumed (external_session / resume_session_id / fork_session_id);
    # fresh sessions have no prior state to compare against.
    _run_resume_checks(options, session, provider_name, provider, stderr)

    tool_registry = build_default_registry(provider=provider)
    replace_cron_tools(tool_registry)
    # Canonicalize BOTH sets up front (before either filter runs) so an alias
    # form (e.g. --disallowed-tools KillShell) resolves while its tool is still
    # registered.
    allow = tool_registry.canonicalize_names(options.allowed_tools) if options.allowed_tools else None
    deny = tool_registry.canonicalize_names(options.disallowed_tools) if options.disallowed_tools else None
    if allow is not None:
        _filter_registry(tool_registry, keep=lambda n: n.lower() in allow)
    if deny is not None:
        _filter_registry(tool_registry, keep=lambda n: n.lower() not in deny)

    # warn when ``--allowed-tools`` / ``--disallowed-tools``
    # silently strips a tool that the resumed conversation's history
    # already references. Without this warning the LLM sees a
    # ``tool_use`` block in context but the registry has no matching
    # tool — leading to hallucinated tool calls, retry loops, or opaque
    # "unknown tool" errors. Only fires when a session was actually
    # resumed (external_session / resume_session_id / fork_session_id).
    _warn_history_tool_conflicts(options, session, tool_registry, stderr)

    # ★ MVP multi-agent: apply coordinator-mode tool filter in headless flow.
    # The REPL already does this in core.py:4712, but the headless / orchestrator
    # path historically did not, so CLAUDE_CODE_COORDINATOR_MODE was inert when
    # the agent was launched by orchestrator. This block fixes that by reusing
    # the same filter_coordinator_tools() helper the REPL uses, restricting the
    # tool registry to {Agent, SendMessage, TaskStop, Read, WebSearch, WebFetch}
    # so the LLM has no choice but to delegate via Agent/SendMessage tools.
    #
    # Note: ToolRegistry has no ``unregister`` method (the _filter_registry
    # helper above silently no-ops on this — that's a pre-existing bug we
    # work around here by mutating the registry's internal storage in place).
    try:
        from clawcodex_ext.coordinator.mode import (
            is_coordinator_mode,
            filter_coordinator_tools,
        )

        if is_coordinator_mode():
            coordinator_tools = filter_coordinator_tools(tool_registry.list_tools())
            allowed_names = {t.name for t in coordinator_tools}
            # Direct in-place mutation of registry storage. The ToolRegistry
            # exposes ``_tools`` (list) and ``_by_name`` (dict including
            # aliases); rebuild both restricted to coordinator-allowed tools.
            kept_tools = [t for t in tool_registry._tools if t.name in allowed_names]
            kept_by_name: dict = {}
            for t in kept_tools:
                kept_by_name[t.name.lower()] = t
                for alias in getattr(t, "aliases", ()):
                    kept_by_name[alias.lower()] = t
            tool_registry._tools = kept_tools
            tool_registry._by_name = kept_by_name
            _LOGGER.info(
                "Coordinator mode: tool registry filtered to %d tools: %s",
                len(kept_tools),
                sorted(allowed_names),
            )
    except Exception as _exc:
        # Never block the headless run on coordinator-mode setup failure.
        _LOGGER.warning(
            "Coordinator-mode tool filter setup failed (continuing without): %s",
            _exc,
        )

    # (workspace_root already resolved above, before the prefetch kick.)

    # Compute the effective permission context. ``skip_permissions=True`` is
    # the legacy alias and means "user passed --dangerously-skip-permissions";
    # ``permission_mode`` / ``is_bypass_permissions_mode_available`` are the
    # round-5 fields. When skip_permissions wins, force bypass mode + bypass
    # availability so the registry's ``has_permissions_to_use_tool`` check
    # short-circuits to ``allow``.
    if options.skip_permissions:
        effective_mode: str = "bypassPermissions"
        bypass_available = True
    else:
        effective_mode = options.permission_mode or "default"
        bypass_available = bool(options.is_bypass_permissions_mode_available)

    # Per-session abort controller. SIGINT trips this so the running
    # tool (Bash supervisor, Agent subagent) unwinds immediately rather
    # than waiting for the next safe interpreter bytecode boundary.
    # Without this wiring, Ctrl-C only fires ``KeyboardInterrupt`` at
    # the next safe boundary — which can be several minutes for a
    # subprocess.wait() or an in-flight subagent.
    # An externally-supplied controller (options.abort_controller) takes
    # precedence so callers like QueryRunner can abort a runaway session
    # from OUTSIDE the executor thread (timeout / stop command).
    abort_controller = options.abort_controller or AbortController()
    # Shared mutable flag used both by the cron scheduler (via is_loading)
    # and the SIGINT handler to distinguish idle from in-flight states.
    in_agent_loop = _InAgentLoopFlag()
    # C1: load persisted permission rules (settings files) at startup so
    # "always allow" rules saved in interactive sessions auto-allow here
    # too. Setup warnings intentionally unsurfaced until phase C6.
    from src.permissions.settings_paths import default_setup_paths
    from src.permissions.setup import setup_permissions

    _perm_setup = setup_permissions(
        cwd=str(workspace_root),
        mode=effective_mode,  # type: ignore[arg-type]
        is_bypass_available=bypass_available,
        **default_setup_paths(str(workspace_root)),
    )
    tool_context = ToolContext(
        workspace_root=workspace_root,
        permission_context=_perm_setup.context,
        abort_controller=abort_controller,
        env=options.env,
        startup_agent=options.startup_agent,
        agent_type=getattr(options.startup_agent, "agent_type", None),
        bundle_context=getattr(options, "bundle_context", None),
    )
    # wire agent_id + runtime_tasks so the
    # pending_messages drain fires for the orchestrator's top-level
    # agent (enables real-time inject at ToolResult boundaries).
    if getattr(options, "agent_id", None):
        tool_context.agent_id = options.agent_id
    if getattr(options, "runtime_tasks", None) is not None:
        tool_context.runtime_tasks = options.runtime_tasks
    tool_context.mcp_clients = getattr(options, "mcp_clients", {}) or {}
    tool_context.mcp_manager_loop = getattr(options, "mcp_manager_loop", None)
    from clawcodex_ext.runtime.tool_context_binding import bind_tool_context_runtime

    bind_tool_context_runtime(
        tool_context,
        tool_registry=tool_registry,
        session=session,
        provider=provider,
    )
    tool_context.goal_thread_id = session.session_id
    tool_context.options.is_non_interactive_session = True
    if options.resume_session_id:
        from clawcodex_ext.goal.runtime import (
            restore_goal_runtime_after_session_resume,
        )

        restore_goal_runtime_after_session_resume(tool_context)

    # ★ MVP peer-to-peer SendMessage: auto-wire team context for multi-agent
    # demos. When the workspace has a pre-existing `.clawcodex/team.json` and
    # the environment specifies `CLAUDE_CODE_AGENT_NAME`, populate the
    # tool_context.team dict so the SendMessage tool can route mailbox writes
    # using the right sender name. This lets a peer-to-peer multi-agent demo
    # work without requiring the agent to call TeamCreate first.
    try:
        _team_file = workspace_root / ".clawcodex" / "team.json"
        _agent_name = os.environ.get("CLAUDE_CODE_AGENT_NAME")
        if _team_file.exists() and _agent_name:
            _team_data = json.loads(_team_file.read_text(encoding="utf-8"))
            _team_data["sender_name"] = _agent_name
            tool_context.team = _team_data
            _LOGGER.info(
                "MVP peer multi-agent: tool_context.team set (team=%s, sender_name=%s)",
                _team_data.get("team_name"),
                _agent_name,
            )
    except Exception as _exc:
        _LOGGER.warning(
            "Failed to auto-wire team context (non-fatal): %s",
            _exc,
        )

    # Initialize upstream plugin/command surfaces and the settings-driven
    # output style for non-interactive sessions too.
    try:
        from src.plugins.init_builtin import init_builtin_plugins

        init_builtin_plugins()
    except Exception:  # noqa: BLE001  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.

    from src.outputStyles import output_style_from_settings

    _settings_style = output_style_from_settings(cwd=str(workspace_root))
    if _settings_style:
        tool_context.output_style_name = _settings_style

    # Settings hooks need the same snapshot, registry and trust state as the
    # interactive entrypoint; otherwise PermissionRequest/Stop hooks silently
    # disappear in print mode.
    from src.hooks.config_manager import bootstrap_hook_config_manager

    tool_context.hook_config_manager = bootstrap_hook_config_manager(
        cwd=str(workspace_root),
    )
    try:
        from src.services.startup_gates import check_trust_accepted

        tool_context.workspace_trusted = check_trust_accepted(workspace_root)
    except Exception:  # noqa: BLE001
        _LOGGER.debug("headless trust check failed", exc_info=True)

    try:
        from src.services.compact.autocompact import AutoCompactTracking

        _run_compact_tracking = AutoCompactTracking()
    except Exception:  # noqa: BLE001
        _run_compact_tracking = None

    def _build_turn_pipeline_config():
        if _run_compact_tracking is None:
            return None
        try:
            from src.services.compact.pipeline import build_production_pipeline_config

            return build_production_pipeline_config(
                provider,
                tool_context,
                _run_compact_tracking,
            )
        except Exception:  # noqa: BLE001
            return None

    if options.skip_permissions or effective_mode == "bypassPermissions":
        tool_context.allow_docs = True
        tool_context.permission_handler = None
    else:
        # Never block a pipe on stdin. Auto-deny any permission request.
        tool_context.permission_handler = _auto_deny_permission_handler(stderr)
    # AskUserQuestion has no terminal to read from in headless mode.
    tool_context.ask_user = _noop_ask_user

    # wire persistent cron scheduler to the headless tool context.
    # is_loading polls the in-agent-loop flag so cron fires are deferred
    # while a query is in flight (busy gate), matching REPL/TUI behavior.
    attach_cron_runtime(
        tool_context,
        autostart=True,
        is_loading=lambda: in_agent_loop.value,
    )

    # seed ``read_file_fingerprints`` from the resumed
    # conversation's historical Read tool_use blocks. Without this,
    # Edit/Write/NotebookEdit staleness checks (``was_file_read_and_unchanged``)
    # reject edits with "file must be read first" even though the model
    # already saw the file in the prior run, and the Read dedup path
    # cannot collapse re-reads to ``file_unchanged``. Mirrors CCB
    # ``print.ts:1173-1176`` ``extractReadFilesFromMessages``. Only
    # meaningful when a session was resumed; on a fresh session the
    # message list is empty and the call is a no-op.
    if (
        options.resume_session_id or options.fork_session_id or options.external_session is not None
    ) and session is not None:
        try:
            from clawcodex_ext.agent.read_file_seed import (
                seed_read_file_state_from_history,
            )

            history_msgs = session.conversation.messages if session.conversation is not None else []
            seed_read_file_state_from_history(
                history_msgs,
                tool_context,
                workspace_root=workspace_root,
            )
        except Exception:
            _LOGGER.debug(
                "read-file-state seeding failed (non-fatal)",
                exc_info=True,
            )

    # Build the input iterator. Slash commands, including /goal, are
    # dispatched by the command engine inside the input loop below.
    if options.input_format == "stream-json":
        inputs: Iterable[UserInputMessage] = StreamJsonReader(stdin)
    else:
        prompt_text = options.prompt
        if prompt_text is None or prompt_text == "-":
            prompt_text = stdin.read()
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            cli_error("error: no prompt provided (pass an argument or pipe stdin)", 2)
        inputs = [UserInputMessage(text=prompt_text, raw={"prompt": prompt_text})]

    writer: StreamJsonWriter | None = None
    if options.output_format == "stream-json":
        writer = StreamJsonWriter(stdout)
        tools = [tool.name for tool in tool_registry.list_tools()]
        writer.write(
            SystemEvent(
                subtype="init",
                session_id=session.session_id,
                goal_operation_id=session.session_id,
                model=getattr(provider, "model", None),
                provider=provider_name,
                cwd=str(workspace_root),
                tools=tools,
                permission_mode=effective_mode,
            )
        )

    aggregate_text: list[str] = []
    aggregate_tool_events: list[dict] = []
    num_turns_total = 0
    usage_total: dict[str, int] = {}
    exit_code = 0
    start = time.monotonic()

    # Two-mode SIGINT handler:
    # * Idle (waiting on stdin for the next stream-json input) → raise
    #   ``KeyboardInterrupt`` immediately so the blocking read returns.
    # * In-flight ``run_agent_loop`` → first strike trips the controller
    #   (cooperative unwind), second strike force-quits via
    #   ``KeyboardInterrupt``. Both map to exit 130.
    # See ``_install_sigint_handler`` for the full handler logic; the
    # for-loop's outer ``except (AbortError, KeyboardInterrupt)`` is the
    # single chokepoint that catches whatever the handler raises.
    # ``restore_sigint`` runs in the ``finally`` so we don't leak global
    # signal state to embedders.
    restore_sigint = _install_sigint_handler(abort_controller, in_agent_loop, stderr)
    try:
        # track active cron tasks for run claim/finalize lifecycle.
        active_tasks: dict[str, str] = {}

        def _run_cron_prompt(prompt: str, task_id: str, run_id: str) -> bool:
            """Execute a single drained cron prompt and finalize its run."""
            _claim_cron_task(workspace_root, active_tasks, task_id)
            # F-REC-H: record automated cron prompts as input too.
            if options.capture is not None:
                options.capture.emit_input(prompt)
            session.conversation.add_user_message(prompt)
            try:
                result = _run_one_agent_loop(
                    session=session,
                    provider=provider,
                    tool_registry=tool_registry,
                    tool_context=tool_context,
                    abort_controller=abort_controller,
                    options=options,
                    writer=writer,
                    aggregate_tool_events=aggregate_tool_events,
                    in_agent_loop=in_agent_loop,
                    pipeline_config_factory=_build_turn_pipeline_config,
                )
            except AbortError:
                _finalize_cron_task(workspace_root, active_tasks, task_id, "cancelled")
                raise
            except Exception as exc:
                _finalize_cron_task(
                    workspace_root,
                    active_tasks,
                    task_id,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return False
            if writer is not None:
                writer.write(AssistantEvent(text=result.response_text))
            _finalize_cron_task(workspace_root, active_tasks, task_id, "completed")
            return True

        try:
            for user_msg in inputs:
                # drain any cron prompts that fired while waiting for
                # the next input and run them before the user prompt.
                _process_cron_outbox(tool_context, active_tasks, _run_cron_prompt)

                # expand @agent-name mentions before sending to LLM.
                text = user_msg.text
                command_tokens = text.strip().split(maxsplit=1)
                is_goal_command_input = bool(command_tokens and command_tokens[0].lower() == "/goal")
                try:
                    from clawcodex_ext.agent.load_agents_dir import (
                        get_agent_definitions_with_overrides,
                    )
                    from clawcodex_ext.agent.agent_definitions import get_built_in_agents
                    from src.command_system.input_processing import (
                        expand_agent_mentions,
                        find_unknown_agent_mentions,
                        format_at_mention_attachments,
                        format_unknown_agent_mention_error,
                    )

                    cwd = str(workspace_root)
                    agents = list(get_agent_definitions_with_overrides(cwd)) or list(get_built_in_agents())

                    # Mirror REPL/TUI: surface a friendly error and skip the
                    # turn when the user mentions an agent that doesn't
                    # exist. Without this guard the model would burn a
                    # whole turn trying to delegate to a fake subagent.
                    unknown_types = find_unknown_agent_mentions(text, agents)
                    if unknown_types:
                        err_msg = format_unknown_agent_mention_error(unknown_types, agents)
                        if writer is not None:
                            writer.write(
                                ResultEvent(
                                    subtype="error",
                                    session_id=session.session_id,
                                    num_turns=0,
                                    result="",
                                    duration_ms=0,
                                    is_error=True,
                                    error=err_msg,
                                )
                            )
                        else:
                            print(f"error: {err_msg}", file=stderr)
                        exit_code = 78  # EX_CONFIG: config-level misuse
                        break

                    agent_attachments = expand_agent_mentions(text, agents)
                    if agent_attachments:
                        extra = format_at_mention_attachments(agent_attachments)
                        if extra:
                            text = f"{extra}\n{text}"
                except Exception:  # nosec B110
                    pass  # best-effort: agent expansion failure is non-fatal

                # Detect and handle slash commands in headless mode.
                # Only commands with ``supports_non_interactive=True`` are
                # allowed; others are rejected with a clear error.
                _skip_agent_loop = False
                try:
                    parsed = parse_user_input(text)
                    if parsed.input_type == "command":
                        cmd = _find_command_for_workspace(
                            parsed.command_name,
                            workspace_root=workspace_root,
                        )

                        if cmd is not None and isinstance(cmd, LocalCommand) and not cmd.supports_non_interactive:
                            err = f"Command /{parsed.command_name} is not available in non-interactive mode"
                            if writer is not None:
                                writer.write(
                                    ResultEvent(
                                        subtype="error",
                                        session_id=session.session_id,
                                        num_turns=0,
                                        result="",
                                        duration_ms=0,
                                        is_error=True,
                                        error=err,
                                    )
                                )
                            else:
                                print(f"error: {err}", file=stderr)
                            exit_code = 1
                            break

                        if isinstance(cmd, PromptCommand):
                            if cmd.disable_non_interactive:
                                err = f"Command /{parsed.command_name} is not available in non-interactive mode"
                                if writer is not None:
                                    writer.write(
                                        ResultEvent(
                                            subtype="error",
                                            session_id=session.session_id,
                                            num_turns=0,
                                            result="",
                                            duration_ms=0,
                                            is_error=True,
                                            error=err,
                                        )
                                    )
                                else:
                                    print(f"error: {err}", file=stderr)
                                exit_code = 1
                                break

                            cmd_ctx = create_command_context(
                                workspace_root=workspace_root,
                                conversation=session.conversation,
                                tool_registry=tool_registry,
                                tool_context=tool_context,
                            )
                            prompt_registry = CommandRegistry()
                            prompt_registry.register(cmd)
                            prompt_result = asyncio.run(
                                CommandEngine(
                                    registry=prompt_registry,
                                    workspace_root=workspace_root,
                                    context=cmd_ctx,
                                ).execute(text)
                            )
                            if not prompt_result.success:
                                err = prompt_result.error or (f"Command /{parsed.command_name} failed")
                                if writer is not None:
                                    writer.write(
                                        ResultEvent(
                                            subtype="error",
                                            session_id=session.session_id,
                                            num_turns=0,
                                            result="",
                                            duration_ms=0,
                                            is_error=True,
                                            error=err,
                                        )
                                    )
                                else:
                                    print(f"error: {err}", file=stderr)
                                exit_code = 1
                                break

                            prompt_text = next(
                                (
                                    item.get("text", "")
                                    for item in prompt_result.prompt_content
                                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
                                ),
                                "",
                            )
                            if prompt_result.should_query and prompt_text:
                                # Match REPL/TUI: the expanded skill prompt, not
                                # the literal slash command, becomes the user turn.
                                text = prompt_text
                            else:
                                if prompt_result.text:
                                    if writer is not None:
                                        writer.write(AssistantEvent(text=prompt_result.text))
                                    else:
                                        aggregate_text.append(prompt_result.text)
                                _skip_agent_loop = True

                        # Claude Code supports ``-p '/goal <condition>'``. Goal
                        # is an InteractiveCommand for the REPL/TUI rendering
                        # contract, but its body needs no interactive prompt,
                        # so execute it through the async command engine here.
                        elif (
                            parsed.command_name.lower() == "goal"
                            and cmd is not None
                            and getattr(cmd, "command_type", None)
                            and cmd.command_type.value == "interactive"
                        ):
                            cmd_ctx = create_command_context(
                                workspace_root=workspace_root,
                                conversation=session.conversation,
                                tool_registry=tool_registry,
                                tool_context=tool_context,
                                provider=provider,
                            )
                            goal_registry = CommandRegistry()
                            goal_registry.register(cmd)
                            goal_result = asyncio.run(
                                CommandEngine(
                                    registry=goal_registry,
                                    workspace_root=workspace_root,
                                    context=cmd_ctx,
                                ).execute(text)
                            )
                            if not goal_result.success:
                                err = goal_result.error or "Command /goal failed"
                                if writer is not None:
                                    writer.write(
                                        ResultEvent(
                                            subtype="error",
                                            session_id=session.session_id,
                                            num_turns=0,
                                            result="",
                                            duration_ms=0,
                                            is_error=True,
                                            error=err,
                                        )
                                    )
                                else:
                                    print(f"error: {err}", file=stderr)
                                exit_code = 1
                                break

                            if goal_result.text:
                                if writer is not None:
                                    writer.write(AssistantEvent(text=goal_result.text))
                                aggregate_text.append(goal_result.text)
                            if goal_result.should_query:
                                # The condition itself is the first directive;
                                # never send the literal slash command to the model.
                                text = (parsed.command_args or "").strip()
                            else:
                                _skip_agent_loop = True

                        # ``/dashboard`` is an InteractiveCommand, but
                        # it has no UI dependencies (pure read-only text
                        # rendering). ``execute_command_sync`` would reject it,
                        # so we special-case it here and emit the rendered
                        # snapshot synchronously. Scrollable mode is dropped in
                        # headless.
                        elif (
                            parsed.command_name.lower() in ("dashboard", "dash")
                            and cmd is not None
                            and getattr(cmd, "command_type", None)
                            and cmd.command_type.value == "interactive"
                        ):
                            dash_text, dash_err = _run_dashboard_headless(
                                args=parsed.command_args or "",
                                workspace_root=workspace_root,
                                tool_context=tool_context,
                                cwd=workspace_root,
                            )
                            if dash_err is not None:
                                if writer is not None:
                                    writer.write(
                                        ResultEvent(
                                            subtype="error",
                                            session_id=session.session_id,
                                            num_turns=0,
                                            result="",
                                            duration_ms=0,
                                            is_error=True,
                                            error=dash_err,
                                        )
                                    )
                                else:
                                    print(f"error: {dash_err}", file=stderr)
                                exit_code = 1
                                break
                            if dash_text:
                                if writer is not None:
                                    writer.write(AssistantEvent(text=dash_text))
                                else:
                                    aggregate_text.append(dash_text)
                            _skip_agent_loop = True

                        # ``/btw`` is an InteractiveCommand, but it
                        # has no UI dependencies (pure read-only single-turn
                        # query). ``execute_command_sync`` would reject it as
                        # "not implemented for sync execution", so we special
                        # case it here and emit the rendered answer
                        # synchronously. Scrollable mode is dropped — there is
                        # no TTY viewer in headless, so the flat text is the
                        # final answer.
                        elif (
                            parsed.command_name.lower() == "btw"
                            and cmd is not None
                            and getattr(cmd, "command_type", None)
                            and cmd.command_type.value == "interactive"
                        ):
                            btw_text, btw_err = _run_btw_headless(
                                args=parsed.command_args or "",
                                workspace_root=workspace_root,
                                conversation=session.conversation,
                                tool_context=tool_context,
                                provider=provider,
                                cwd=workspace_root,
                            )
                            if btw_err is not None:
                                if writer is not None:
                                    writer.write(
                                        ResultEvent(
                                            subtype="error",
                                            session_id=session.session_id,
                                            num_turns=0,
                                            result="",
                                            duration_ms=0,
                                            is_error=True,
                                            error=btw_err,
                                        )
                                    )
                                else:
                                    print(f"error: {btw_err}", file=stderr)
                                exit_code = 1
                                break
                            if btw_text:
                                if writer is not None:
                                    writer.write(AssistantEvent(text=btw_text))
                                else:
                                    aggregate_text.append(btw_text)
                            _skip_agent_loop = True
                        else:
                            cmd_ctx = create_command_context(
                                workspace_root=workspace_root,
                                conversation=session.conversation,
                                tool_registry=tool_registry,
                                tool_context=tool_context,
                            )
                            success, result_text, error = execute_command_sync(
                                parsed.command_name, parsed.command_args, cmd_ctx
                            )
                            if success:
                                if writer is not None:
                                    writer.write(AssistantEvent(text=result_text or ""))
                                else:
                                    aggregate_text.append(result_text or "")
                                _skip_agent_loop = True
                            else:
                                err = error or f"Command /{parsed.command_name} failed"
                                if writer is not None:
                                    writer.write(
                                        ResultEvent(
                                            subtype="error",
                                            session_id=session.session_id,
                                            num_turns=0,
                                            result="",
                                            duration_ms=0,
                                            is_error=True,
                                            error=err,
                                        )
                                    )
                                else:
                                    print(f"error: {err}", file=stderr)
                                exit_code = 1
                                break
                except Exception as exc:
                    # Most slash-command probing remains best-effort, but a
                    # failed /goal command must never fall through as a literal
                    # model prompt. Surface a non-zero command failure instead.
                    if is_goal_command_input:
                        err = str(exc).strip() or "Command /goal failed"
                        if writer is not None:
                            writer.write(
                                ResultEvent(
                                    subtype="error",
                                    session_id=session.session_id,
                                    num_turns=0,
                                    result="",
                                    duration_ms=0,
                                    is_error=True,
                                    error=err,
                                )
                            )
                        else:
                            print(f"error: {err}", file=stderr)
                        exit_code = 1
                        break
                    # best-effort: other slash-command detection failures are
                    # non-fatal and retain the historical fallback behaviour.
                    pass

                if _skip_agent_loop:
                    continue

                # F-REC-H: record the user prompt as an "i" frame.
                if options.capture is not None:
                    options.capture.emit_input(text)

                session.conversation.add_user_message(text)

                try:
                    result = _run_one_agent_loop(
                        session=session,
                        provider=provider,
                        tool_registry=tool_registry,
                        tool_context=tool_context,
                        abort_controller=abort_controller,
                        options=options,
                        writer=writer,
                        aggregate_tool_events=aggregate_tool_events,
                        in_agent_loop=in_agent_loop,
                        pipeline_config_factory=_build_turn_pipeline_config,
                    )
                except AbortError:
                    # Re-raise to the outer ``except`` so the cancelled
                    # ResultEvent is emitted in exactly one place.
                    raise
                except Exception as exc:
                    exit_code = 1
                    failed_usage = getattr(exc, "aggregate_usage", None)
                    failed_turns = int(getattr(exc, "num_turns", 0) or 0)
                    num_turns_total += failed_turns
                    if isinstance(failed_usage, dict):
                        for key, value in failed_usage.items():
                            usage_total[key] = usage_total.get(key, 0) + int(value or 0)
                    # best-effort error event with stable
                    # fingerprint. The session_id lets the aggregator
                    # correlate the crash with the same conversation
                    # that emitted the assistant / tool events.
                    try:
                        from telemetry import record_error

                        record_error(session_id=session.session_id, exc=exc)
                    except Exception:  # nosec B110
                        pass  # Telemetry is optional and must not affect command execution.
                    if writer is not None:
                        writer.write(
                            ResultEvent(
                                subtype="error",
                                session_id=session.session_id,
                                goal_operation_id=session.session_id,
                                num_turns=num_turns_total,
                                result=str(exc),
                                duration_ms=int((time.monotonic() - start) * 1000),
                                usage=usage_total or None,
                                is_error=True,
                                error=str(exc),
                            )
                        )
                    else:
                        print(f"error: {exc}", file=stderr)
                    break

                num_turns_total += result.num_turns
                if result.usage:
                    for key, value in result.usage.items():
                        usage_total[key] = usage_total.get(key, 0) + int(value)

                if writer is not None:
                    writer.write(AssistantEvent(text=result.response_text))
                aggregate_text.append(result.response_text)

                # drain cron prompts that fired while the agent was
                # busy with the user turn and run them before the next input.
                _process_cron_outbox(tool_context, active_tasks, _run_cron_prompt)
            # one last drain after the input stream ends so cron
            # prompts that fired during the final turn are not dropped.
            _process_cron_outbox(tool_context, active_tasks, _run_cron_prompt)
        except (AbortError, KeyboardInterrupt) as exc:
            # Cancellation from ANY point in the loop body lands here:
            # * ``AbortError`` from a cooperative unwind inside
            #   ``run_agent_loop`` (first SIGINT, in-flight mode).
            # * ``KeyboardInterrupt`` from the SIGINT handler's idle
            #   branch (raised mid-``inputs.__iter__()`` while blocked on
            #   stdin), or from the in-flight second-strike force-quit.
            # All map to exit 130 for shell parity. ``error`` is left
            # unset — ``subtype: "cancelled"`` already carries the
            # signal, and pairing ``is_error=False`` with a populated
            # ``error`` field would confuse consumers.
            exit_code = 130
            # best-effort cancellation record. Distinguishes
            # user cancellation (KeyboardInterrupt / AbortError) from
            # a real provider/tool crash by passing the exception
            # instance — the recorder's fingerprint path picks up
            # KeyboardInterrupt fingerprints, which is useful for
            # surfacing repeat-abort issues without counting them as
            # crashes. Failures are swallowed.
            try:
                from telemetry import record_error

                record_error(session_id=session.session_id, exc=exc)
            except Exception:  # nosec B110
                pass  # Telemetry is optional and must not affect command execution.
            if writer is not None:
                writer.write(
                    ResultEvent(
                        subtype="cancelled",
                        session_id=session.session_id,
                        goal_operation_id=session.session_id,
                        num_turns=num_turns_total,
                        result="",
                        duration_ms=int((time.monotonic() - start) * 1000),
                        is_error=False,
                    )
                )
    finally:
        restore_sigint()
        # Async Agent workers are detached from the per-turn event loop.
        # Headless owns the session lifetime, so make their cancellation and
        # subprocess teardown explicit before returning to the caller.
        try:
            tool_context.task_manager.shutdown(timeout=2.0)
        except Exception:  # nosec B110
            pass  # Cleanup is best-effort and must not replace the primary operation result.
        # stop the cron scheduler background thread. The scheduler
        # registers its own atexit hooks, but explicit stop prevents the
        # thread from outliving the headless run in long-lived embedders.
        scheduler = getattr(tool_context, "cron_scheduler", None)
        if scheduler is not None:
            try:
                scheduler.stop()
            except Exception:  # nosec B110
                pass  # Cleanup is best-effort and must not replace the primary operation result.
        # persist accumulated transcript at end-of-run so
        # the next ``--resume <sid>`` sees the messages we just generated.
        # Without this, headless resume is "load history, run once,
        # discard" — the resume-history trap described in
        # ``Session.save`` writes both the transcript
        # (via ``save_to_session_storage``) AND the trailing
        # ``session_snapshot`` line (cost block for next resume).
        # Best-effort: a save failure must not mask the user's exit code.
        if options.persist_on_exit and session is not None:
            try:
                session.save()
            except Exception:
                _LOGGER.debug(
                    "failed to persist headless session transcript",
                    exc_info=True,
                )

    duration_ms = int((time.monotonic() - start) * 1000)
    final_text = "\n\n".join(t for t in aggregate_text if t).strip()
    multimodel_payload = None
    if getattr(provider, "last_result", None):
        from clawcodex_ext.multimodel.display import ModelDisplayState, SummaryBuilder

        states = [ModelDisplayState.from_result(item) for item in provider.last_result]
        multimodel_payload = json.loads(
            SummaryBuilder.build_json(states, strategy=getattr(provider.strategy, "name", "parallel"))
        )
        final_text = SummaryBuilder.build_text(states)

    if options.output_format == "text":
        if final_text:
            stdout.write(final_text + "\n")
            stdout.flush()
        # S-R1: print resume hint to TTY after text output. JSON / stream-json
        # already carry session_id in their structured payload, so skip them.
        from clawcodex_ext.utils.resume_hint import print_resume_hint

        print_resume_hint(getattr(session, "session_id", None), stream=stdout)
    elif options.output_format == "json":
        if exit_code == 0:
            json_subtype = "success"
        elif exit_code == 130:
            json_subtype = "cancelled"
        else:
            json_subtype = "error"
        payload = {
            "type": "result",
            "subtype": json_subtype,
            "session_id": session.session_id,
            "goal_operation_id": session.session_id,
            "provider": provider_name,
            "model": getattr(provider, "model", None),
            "num_turns": num_turns_total,
            "result": final_text,
            "duration_ms": duration_ms,
            "usage": usage_total or None,
            "tool_events": aggregate_tool_events,
            "is_error": exit_code not in (0, 130),
        }
        if multimodel_payload is not None:
            payload["multimodel"] = multimodel_payload
        stdout.write(ndjson_safe_dumps(payload) + "\n")
        stdout.flush()
    elif options.output_format == "stream-json" and writer is not None and exit_code == 0:
        writer.write(
            ResultEvent(
                subtype="success",
                session_id=session.session_id,
                goal_operation_id=session.session_id,
                num_turns=num_turns_total,
                result=final_text,
                duration_ms=duration_ms,
                usage=usage_total or None,
            )
        )

    # best-effort session_end + command_run. The session id is
    # the same one the conversation persists under so the per-day
    # aggregator can cross-link events to a known session. Failures
    # are swallowed — telemetry must never block the user's exit.
    try:
        from telemetry import record_command_run, record_session_end

        duration_s = time.monotonic() - start
        record_session_end(
            session_id=session.session_id,
            duration_s=duration_s,
            exit_status=exit_code,
        )
        record_command_run(
            session_id=session.session_id,
            command_name="headless",
            mode="non_interactive",
            success=(exit_code == 0),
            duration_s=duration_s,
            exit_status=exit_code,
        )
    except Exception:  # nosec B110
        pass  # Telemetry is optional and must not affect command execution.

    return exit_code


# ---------------------------------------------------------------------------
# Helpers


def _try_run_provider_free_goal_summary(
    options: HeadlessOptions,
    *,
    workspace_root: Path,
    stdout: IO[str],
    stderr: IO[str],
) -> int | None:
    """Handle local-only ``-p /goal`` status and clear before provider validation.

    Status and clear only touch local session state. They should not fail
    merely because the default chat provider is not configured. Goal creation
    and continuation still take the normal provider-backed path.
    """
    if options.input_format != "text":
        return None
    if options.prompt is None or options.prompt == "-":
        return None

    prompt_text = options.prompt.strip()
    if not prompt_text:
        return None

    parsed = parse_user_input(prompt_text)
    if parsed.input_type != "command":
        return None
    if parsed.command_name.lower() != "goal":
        return None
    command_args = parsed.command_args.strip().lower()
    from clawcodex_ext.goal.command import GOAL_CLEAR_ALIASES

    if command_args and command_args not in GOAL_CLEAR_ALIASES:
        return None

    command = _find_builtin_or_registered_command(parsed.command_name)
    if command is None:
        return None

    if options.external_session is not None:
        session = options.external_session
    elif options.resume_session_id:
        try:
            session = Session.resume(options.resume_session_id)
        except Exception as exc:
            print(
                f"error: failed to resume session '{options.resume_session_id}': {exc}",
                file=stderr,
            )
            return 1
        if session is None:
            print(
                f"error: no session found with ID '{options.resume_session_id}'",
                file=stderr,
            )
            return 1
    else:
        session = Session.create("local", options.model or "")

    started_at = time.monotonic()
    try:
        abort_controller = AbortController()
        tool_context = ToolContext(
            workspace_root=workspace_root,
            abort_controller=abort_controller,
            env=options.env,
            startup_agent=options.startup_agent,
            agent_type=getattr(options.startup_agent, "agent_type", None),
            bundle_context=getattr(options, "bundle_context", None),
        )
        tool_context.mcp_clients = getattr(options, "mcp_clients", {}) or {}
        tool_context.mcp_manager_loop = getattr(options, "mcp_manager_loop", None)
        tool_context.session_id = session.session_id
        tool_context.goal_thread_id = session.session_id
        tool_context.options.is_non_interactive_session = True
        if options.resume_session_id:
            from clawcodex_ext.goal.runtime import (
                restore_goal_runtime_after_session_resume,
            )

            restore_goal_runtime_after_session_resume(tool_context)

        command_context = create_command_context(
            workspace_root=workspace_root,
            conversation=session.conversation,
            tool_context=tool_context,
        )
        command_registry = CommandRegistry()
        command_registry.register(command)
        engine = CommandEngine(
            registry=command_registry,
            workspace_root=workspace_root,
            context=command_context,
        )
        result = asyncio.run(engine.execute(prompt_text))
    except Exception as exc:
        print(f"error: {exc}", file=stderr)
        return 1

    duration_ms = int((time.monotonic() - started_at) * 1000)
    if not result.success:
        error = result.error or result.text or "Command /goal failed"
        if options.output_format == "json":
            stdout.write(
                ndjson_safe_dumps(
                    {
                        "type": "result",
                        "subtype": "error",
                        "session_id": session.session_id,
                        "goal_operation_id": session.session_id,
                        "provider": "local",
                        "model": options.model,
                        "num_turns": 0,
                        "result": "",
                        "duration_ms": duration_ms,
                        "usage": None,
                        "tool_events": [],
                        "is_error": True,
                        "error": error,
                    }
                )
                + "\n"
            )
            stdout.flush()
        elif options.output_format == "stream-json":
            writer = StreamJsonWriter(stdout)
            writer.write(
                SystemEvent(
                    subtype="init",
                    session_id=session.session_id,
                    goal_operation_id=session.session_id,
                    model=options.model,
                    provider="local",
                    cwd=str(workspace_root),
                    tools=[],
                    permission_mode=options.permission_mode,
                )
            )
            writer.write(
                ResultEvent(
                    subtype="error",
                    session_id=session.session_id,
                    goal_operation_id=session.session_id,
                    num_turns=0,
                    result="",
                    duration_ms=duration_ms,
                    is_error=True,
                    error=error,
                )
            )
        else:
            print(f"error: {error}", file=stderr)
        return 1
    if result.should_query:
        print("error: Command /goal unexpectedly requested a model query", file=stderr)
        return 1

    final_text = (result.text or "").strip()
    if options.output_format == "json":
        stdout.write(
            ndjson_safe_dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "session_id": session.session_id,
                    "goal_operation_id": session.session_id,
                    "provider": "local",
                    "model": options.model,
                    "num_turns": 0,
                    "result": final_text,
                    "duration_ms": duration_ms,
                    "usage": None,
                    "tool_events": [],
                    "is_error": False,
                }
            )
            + "\n"
        )
        stdout.flush()
    elif options.output_format == "stream-json":
        writer = StreamJsonWriter(stdout)
        writer.write(
            SystemEvent(
                subtype="init",
                session_id=session.session_id,
                goal_operation_id=session.session_id,
                model=options.model,
                provider="local",
                cwd=str(workspace_root),
                tools=[],
                permission_mode=options.permission_mode,
            )
        )
        if final_text:
            writer.write(AssistantEvent(text=final_text))
        writer.write(
            ResultEvent(
                subtype="success",
                session_id=session.session_id,
                goal_operation_id=session.session_id,
                num_turns=0,
                result=final_text,
                duration_ms=duration_ms,
            )
        )
    else:
        if final_text:
            stdout.write(final_text + "\n")
            stdout.flush()
        from clawcodex_ext.utils.resume_hint import print_resume_hint

        print_resume_hint(session.session_id, stream=stdout)
    if options.persist_on_exit:
        try:
            session.save()
        except Exception:
            _LOGGER.debug(
                "failed to persist provider-free headless command session",
                exc_info=True,
            )
    return 0


def _find_builtin_or_registered_command(command_name: str) -> Any | None:
    command = get_command_registry().get(command_name)
    if command is not None:
        return command
    lowered = command_name.lower()
    for builtin_command in get_builtin_commands():
        if builtin_command.name.lower() == lowered:
            return builtin_command
        if lowered in [alias.lower() for alias in builtin_command.aliases]:
            return builtin_command
    return None


def _find_command_for_workspace(
    command_name: str,
    *,
    workspace_root: Path,
) -> Any | None:
    """Resolve builtins, registered commands, and workspace skill commands."""

    command = _find_builtin_or_registered_command(command_name)
    if command is not None:
        return command

    from clawcodex_ext.command_system.aggregator import get_commands

    lowered = command_name.lower()
    for candidate in get_commands(workspace_root):
        if candidate.name.lower() == lowered:
            return candidate
        if lowered in [alias.lower() for alias in candidate.aliases]:
            return candidate
    return None


class _InAgentLoopFlag:
    """Mutable shared flag indicating whether ``run_agent_loop`` is in flight.

    Read by the SIGINT handler to decide between cooperative abort
    (in-flight: trip the controller, let the loop unwind at the next
    safe boundary) and immediate raise (idle, e.g. blocked on
    ``StreamJsonReader``'s stdin read: the only way to make the read
    return is to actually raise ``KeyboardInterrupt`` on the same
    thread — Python 3 auto-retries EINTR'd reads when the handler
    didn't raise, per PEP 475).
    """

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = False


class _CompatPromptBlocks(list[dict[str, Any]]):
    """Block list retaining the legacy string-inspection surface."""

    def _joined_text(self) -> str:
        return "\n\n".join(
            str(block.get("text", "")) for block in self if isinstance(block, dict) and block.get("type") == "text"
        )

    def startswith(self, prefix: str) -> bool:
        return self._joined_text().startswith(prefix)

    def endswith(self, suffix: str) -> bool:
        return self._joined_text().endswith(suffix)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._joined_text()
        return super().__contains__(item)


def _install_sigint_handler(
    controller: AbortController,
    in_agent_loop: _InAgentLoopFlag,
    stderr: IO[str],
) -> Callable[[], None]:
    """Install a context-aware SIGINT handler.

    - **Idle** (``in_agent_loop.value`` is False, e.g. blocked on
      stdin reading the next stream-json input): raise
      ``KeyboardInterrupt`` immediately. Python 3 PEP 475 retries
      EINTR'd ``read()`` calls when the signal handler did NOT raise,
      so a cooperative abort here would *hang the stdin read* until
      the user hit Ctrl-C a second time — a UX regression vs. the
      pre-fix behaviour where the first Ctrl-C raised at the next
      bytecode boundary and exited the program. Raising on the first
      strike restores parity with that pre-fix path.

    - **Cooperative** (in-flight ``run_agent_loop``, first strike):
      trip ``controller``. Every abort-aware site — the agent loop's
      ``_check_cancel`` boundaries, the Bash supervisor's poll loop,
      the subagent query loop, the streaming executor's per-tool
      controller, hook gates — sees the signal and unwinds gracefully
      with a partial result that's appended to the conversation. A
      message is printed to stderr so the user knows the request was
      received but unwind may take a moment.

    - **Cooperative** (in-flight, second strike): re-install the
      platform default handler (defense-in-depth against a possible
      third strike landing during unwind) and raise
      ``KeyboardInterrupt`` directly. This is the force-quit escape
      hatch for the rare case where a tool doesn't honour the abort.

    Returns a callable that restores whatever handler was installed
    before us, so embedders that drive ``run_headless`` from inside a
    larger program don't have their global signal state mutated.

    ``signal.signal`` is only callable from the main thread; if we are
    not the main thread (e.g. an SDK harness that runs headless in a
    worker thread), the install is skipped and the returned restore is
    a no-op. Cancellation in that case falls back to the agent loop's
    natural turn-boundary checks via ``KeyboardInterrupt`` propagation
    from whatever signal facility the embedder is using.
    """

    previous = _signal.getsignal(_signal.SIGINT)

    def _handler(signum, frame):
        if not in_agent_loop.value:
            # Idle on input — raise so the blocking stdin read returns.
            # No need to swap to ``default_int_handler``: there's no
            # cooperative-unwind escalation state to escalate from, and
            # ``restore_sigint()`` in the ``finally`` block will revert
            # the user's pre-existing handler shortly after the raise
            # unwinds out of ``run_headless``. A second SIGINT before
            # that finally runs would just re-enter this handler and
            # raise again — fine.
            raise KeyboardInterrupt
        if controller.signal.aborted:
            # Second strike during cooperative unwind: re-arm the
            # platform default handler (so any third strike terminates
            # the process the usual way) and raise the force-quit.
            _signal.signal(_signal.SIGINT, _signal.default_int_handler)
            raise KeyboardInterrupt
        controller.abort("user_interrupt")
        try:
            # Plain ASCII for portability — some legacy Windows code
            # pages can't encode U+2026 and would silently drop the
            # message via the outer except.
            stderr.write("\nCancelling... (Ctrl-C again to force quit)\n")
            stderr.flush()
        except Exception:  # nosec B110
            # A broken stderr (closed pipe etc.) must not stop the
            # cancellation from propagating — the controller is already
            # tripped, the agent loop will unwind regardless.
            pass

    try:
        _signal.signal(_signal.SIGINT, _handler)
    except (ValueError, OSError):
        # Not in main thread (ValueError) or SIGINT not supported on
        # this platform (OSError on some Windows configurations).
        # Fall back to the agent loop's natural turn-boundary cancel
        # checks via ``KeyboardInterrupt`` — the pre-fix behaviour.
        return lambda: None

    def _restore() -> None:
        try:
            _signal.signal(_signal.SIGINT, previous)
        except (ValueError, OSError):
            pass  # Signal restoration is best-effort during teardown.

    return _restore


def _filter_registry(registry, *, keep) -> None:
    """In-place best-effort filter of a ToolRegistry.

    Drops every tool for which ``keep(name)`` is False so that
    ``--allowed-tools`` / ``--disallowed-tools`` remove the tool from the pool
    the model sees (schemas are emitted from ``registry.list_tools()``), not
    just block it at execution time.

    Prefer the canonical ``remove_tool`` API, retain ``unregister`` support
    for older/custom registries, and finally fall back to rebuilding the
    internal indexes for legacy registry doubles.
    """
    try:
        entries = list(registry.list_tools())
    except Exception:
        return

    remover = getattr(registry, "remove_tool", None)
    if not callable(remover):
        remover = getattr(registry, "unregister", None)
    if callable(remover):
        for tool in entries:
            name = getattr(tool, "name", "")
            if not keep(name):
                try:
                    remover(name)
                except Exception as exc:  # nosec B110
                    _LOGGER.warning(
                        "Failed to remove filtered tool %s (%s)",
                        name,
                        type(exc).__name__,
                    )
        return

    # Fallback for legacy/opaque test registries with no removal API. Touch
    # only ``_tools`` and ``_by_name`` and preserve alias lookup entries.
    kept_tools = [t for t in entries if keep(getattr(t, "name", ""))]
    kept_by_name: dict = {}
    for t in kept_tools:
        kept_by_name[t.name.lower()] = t
        for alias in getattr(t, "aliases", ()):
            kept_by_name[alias.lower()] = t
    try:
        registry._tools = kept_tools
        registry._by_name = kept_by_name
    except Exception:  # nosec B110
        # Read-only / opaque registry — give up silently to preserve
        # the original "best-effort" contract.
        pass


def _auto_deny_permission_handler(stderr: IO[str]):
    from src.permissions.types import PermissionAskReply, PermissionAskRequest

    def handler(
        request: PermissionAskRequest | str,
        message: str | None = None,
        _suggestion: str | None = None,
    ) -> PermissionAskReply | tuple[bool, bool]:
        """Deny both current request objects and the legacy callback shape.

        ``ToolContext.permission_handler`` remains a public compatibility
        seam used directly by SDK callers.  Registry dispatch adapts either
        shape, but callers predating ``PermissionAskRequest`` still invoke
        ``(tool_name, message, suggestion)`` and expect a boolean tuple.
        """
        legacy_call = not isinstance(request, PermissionAskRequest)
        tool_name = str(request) if legacy_call else request.tool_name
        request_message = (message or "") if legacy_call else request.message
        stderr.write(
            f"[headless] denying permission for {tool_name}: "
            f"{request_message}"
            " (pass --dangerously-skip-permissions to bypass)\n"
        )
        try:
            stderr.flush()
        except Exception:  # nosec B110
            pass  # Cleanup is best-effort and must not replace the primary operation result.
        if legacy_call:
            return False, False
        return PermissionAskReply(behavior="deny")

    return handler


_NON_INTERACTIVE_ANSWER = (
    "No interactive user is available (running headless/non-interactive). "
    "Proceed autonomously with your best judgment and reasonable default "
    "assumptions; do not ask again."
)


def _noop_ask_user(questions):  # type: ignore[override]
    # Non-interactive mode: there is no user to answer. Returning bare
    # empty strings left the model with no signal about WHY the answer was
    # empty — observed live (terminal-bench raman-fitting) to make it flail,
    # re-asking / retrying instead of committing to an approach. Return an
    # explicit "proceed autonomously" answer so the model moves on
    # decisively. (The interactive TUI still shows the real dialog; only the
    # headless surface — which cannot collect input — substitutes this.)
    answers: dict = {}
    for q in questions or []:
        if isinstance(q, dict) and isinstance(q.get("question"), str):
            answers[q["question"]] = _NON_INTERACTIVE_ANSWER
    return answers


# ---------------------------------------------------------------------------
# /dashboard in headless / --print mode
# ---------------------------------------------------------------------------
# /dashboard is an ``InteractiveCommand`` so the REPL can render long snapshots
# in a keyboard-scrolled viewer. It does not need a UI surface, so in headless
# mode we run it synchronously and emit the flat text. Scrollable mode is
# dropped — there is no TTY viewer in headless.


def _run_dashboard_headless(
    *,
    args: str,
    workspace_root: Path,
    tool_context: Any,
    cwd: Path,
) -> "tuple[str | None, str | None]":
    """Run ``/dashboard`` synchronously in headless mode and return ``(text, error)``."""
    from clawcodex_ext.command_system.dashboard_command import DASHBOARD_COMMAND

    ctx = create_command_context(
        workspace_root=workspace_root,
        tool_context=tool_context,
        tool_registry=getattr(tool_context, "tool_registry", None),
        cwd=cwd,
    )
    try:
        # Use the pre-built ``DASHBOARD_COMMAND`` singleton (which carries
        # ``name`` and ``description``) rather than constructing a fresh
        # ``DashboardCommand()`` — the latter raises
        # ``InteractiveCommand.__init__() missing 2 required positional
        # arguments: 'name' and 'description'`` because ``CommandBase``
        # is a frozen dataclass with no defaults on those fields.
        outcome = asyncio.run(DASHBOARD_COMMAND.run(args, ctx))
    except Exception as exc:  # noqa: BLE001 — surface any failure cleanly
        return (None, f"Dashboard failed: {exc}")

    if outcome is None or getattr(outcome, "display", None) == "skip":
        return ("", None)
    return (outcome.message or "", None)


# ---------------------------------------------------------------------------
# /btw side-question in headless / --print mode
# ---------------------------------------------------------------------------
# /btw is an ``InteractiveCommand`` (it drives a UI surface for scrollable
# viewing in the REPL). ``execute_command_sync`` rejects anything that
# isn't a ``LocalCommand`` (see ``builtins.py:1495``), so the headless
# dispatcher can't route it through the standard path. We special-case
# ``btw`` here: it is the only InteractiveCommand that makes sense in a
# non-interactive flow (it's a pure read-only single-turn query that does
# not need a UI surface), and the planning doc calls out exactly
# this degradation — "non-interactive mode /btw falls back to synchronous
# stdout print".
#
# Behaviour:
#   * Empty args → Usage hint (no API call).
#   * Calls ``btw_command_run`` synchronously via ``asyncio.run``.
#   * Returns the rendered text directly. ``scrollable`` is dropped — there
#     is no TTY viewer in headless, so the flat text is the final answer.
#   * Errors are returned as ``(None, error_str)`` so the caller can emit
#     a normal CLI error stream.
#
# The function is intentionally side-effect-free on the *session* — no
# messages are appended to ``conversation`` (matches the isolation
# invariant: a side question must never leak into the main transcript).


def _run_btw_headless(
    *,
    args: str,
    workspace_root: Path,
    conversation: Any,
    tool_context: Any,
    provider: Any,
    cwd: Path,
) -> "tuple[str | None, str | None]":
    """Run /btw synchronously in headless mode and return ``(text, error)``.

    Returns:
        ``(text, None)`` on success (``text`` is the rendered answer, may
        be empty if the command body short-circuited to a Usage hint), or
        ``(None, error_str)`` on failure. The two are mutually exclusive.
    """
    from clawcodex_ext.command_system.btw_command import btw_command_run

    if not args.strip():
        return (
            "Usage: /btw <your question> —— 在不中断工作会话的前提下快速询问",
            None,
        )

    # ``create_command_context`` already wires tool_context and conversation;
    # we additionally pass ``provider`` so ``_build_cache_safe_params``'s
    # fallback path can attach it to ``tool_context._active_provider`` (the
    # fork child requires this attribute to be set before ``run_forked_agent``
    # builds its ``QueryParams``).
    ctx = create_command_context(
        workspace_root=workspace_root,
        conversation=conversation,
        tool_context=tool_context,
        tool_registry=getattr(tool_context, "tool_registry", None),
        provider=provider,
        cwd=cwd,
    )

    try:
        outcome = asyncio.run(btw_command_run(args, ctx))
    except Exception as exc:  # noqa: BLE001 — surface any failure cleanly
        return (None, f"Side question failed: {exc}")

    if outcome is None or getattr(outcome, "display", None) == "skip":
        return ("", None)
    return (outcome.message or "", None)


def _build_event_bridge(writer: StreamJsonWriter | None, sink: list[dict]):
    def on_event(event: ToolEvent) -> None:
        if event.kind == "tool_use":
            record = {
                "type": "tool_use",
                "tool_use_id": event.tool_use_id,
                "name": event.tool_name,
                "input": event.tool_input or {},
            }
            sink.append(record)
            if writer is not None:
                writer.write(
                    ToolUseEvent(
                        tool_use_id=event.tool_use_id,
                        name=event.tool_name,
                        input=dict(event.tool_input or {}),
                    )
                )
        elif event.kind in ("tool_result", "tool_error"):
            record = {
                "type": "tool_result",
                "tool_use_id": event.tool_use_id,
                "name": event.tool_name,
                "output": _jsonable(event.tool_output),
                "is_error": bool(event.is_error),
            }
            if event.error:
                record["error"] = event.error
            sink.append(record)
            if writer is not None:
                writer.write(
                    ToolResultEvent(
                        tool_use_id=event.tool_use_id,
                        name=event.tool_name,
                        output=_jsonable(event.tool_output),
                        is_error=bool(event.is_error),
                    )
                )

    return on_event


def _get_active_evaluator_goal(tool_context: ToolContext) -> Any | None:
    """Return the active Claude-style goal, without leaking lookup failures."""

    try:
        from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
        from clawcodex_ext.goal.runtime import goal_runtime_for_context

        runtime = goal_runtime_for_context(tool_context)
        if runtime is None:
            return None
        goal = runtime.service.get_goal(runtime.thread_id)
        if (
            goal is not None
            and goal.status is ThreadGoalStatus.ACTIVE
            and goal.completion_mode is GoalCompletionMode.EVALUATOR
        ):
            return goal
    except Exception:
        return None
    return None


def _run_one_agent_loop(
    session: Session,
    provider: Any,
    tool_registry: Any,
    tool_context: ToolContext,
    abort_controller: AbortController,
    options: HeadlessOptions,
    writer: StreamJsonWriter | None,
    aggregate_tool_events: list[dict],
    in_agent_loop: _InAgentLoopFlag,
    *,
    on_text_chunk: Callable[[str], None] | None = None,
    pipeline_config_factory: Callable[[], Any] | None = None,
) -> AgentLoopResult:
    """Run the current conversation through the canonical query() loop once.

    The caller must have already appended the user/cron message to
    ``session.conversation``. This helper wraps the asyncio adapter,
    handles event bridging, and re-wraps the result into the legacy
    ``AgentLoopResult`` shape.
    """
    on_event = _build_event_bridge(writer, aggregate_tool_events)
    _ext_cb = options.on_event
    if _ext_cb is not None:
        _internal_on_event = on_event

        def on_event(event: ToolEvent) -> None:  # type: ignore[misc]
            _internal_on_event(event)
            try:
                _ext_cb(event)
            except Exception:  # nosec B110
                pass  # The optional callback is isolated so it cannot interrupt the owning event loop.

    # F-REC-H: bridge tool events and text chunks to the asciicast recorder.
    _recorder = options.capture
    if _recorder is not None:
        _orig_on_event = on_event

        def on_event(event: ToolEvent) -> None:  # type: ignore[misc]
            _orig_on_event(event)
            try:
                _recorder.emit_tool_event(event)
            except Exception:  # nosec B110
                pass  # The optional callback is isolated so it cannot interrupt the owning event loop.

        _orig_text_chunk = on_text_chunk

        def _rec_text_chunk(chunk: str) -> None:
            if _orig_text_chunk is not None:
                _orig_text_chunk(chunk)
            try:
                _recorder.emit_text(chunk)
            except Exception:  # nosec B110
                pass  # This side-channel notification is best-effort and must not alter the primary result.

        on_text_chunk = _rec_text_chunk

    if on_text_chunk is None and writer is not None and options.include_partial_messages:

        def _emit_partial(chunk: str) -> None:
            writer.write(PartialTextEvent(text=chunk))

        on_text_chunk = _emit_partial

    from src.outputStyles import resolve_output_style

    _style_prompt = resolve_output_style(
        getattr(tool_context, "output_style_name", None),
        getattr(tool_context, "output_style_dir", None),
    ).prompt
    try:
        effective_system_prompt = build_effective_system_prompt(
            _style_prompt,
            tool_context,
            provider=provider,
        )
    except TypeError:
        effective_system_prompt = build_effective_system_prompt(
            _style_prompt,
            tool_context,
        )
    if isinstance(effective_system_prompt, list):
        effective_system_prompt = _CompatPromptBlocks(effective_system_prompt)
    if options.append_system_prompt:
        if isinstance(effective_system_prompt, list):
            effective_system_prompt.append({"type": "text", "text": options.append_system_prompt})
        else:
            effective_system_prompt = f"{effective_system_prompt}\n\n{options.append_system_prompt}"

    def _persist(msg: Any) -> None:
        try:
            conversation = session.conversation
            add_existing = getattr(conversation, "add_existing_message", None)
            if callable(add_existing):
                # Preserve the complete message object (content blocks, usage,
                # model/request metadata) when the canonical Conversation API
                # is available.
                add_existing(msg)
            else:
                # Compatibility for lightweight/custom conversation objects.
                # Newer implementations accept ``usage``; older test doubles
                # may only expose the historic two-argument signature.
                try:
                    conversation.add_message(
                        msg.role,
                        msg.content,
                        usage=getattr(msg, "usage", None),
                    )
                except TypeError as exc:
                    if "usage" not in str(exc):
                        raise
                    conversation.add_message(msg.role, msg.content)

            # Stream-json exposes signed/redacted thinking as a separate
            # assistant content event. The visible final text is still emitted
            # once by the caller after the agent loop completes.
            if writer is not None and msg.role == "assistant":
                from src.types.content_blocks import (
                    RedactedThinkingBlock,
                    ThinkingBlock,
                    content_block_to_dict,
                )

                blocks = msg.content if isinstance(msg.content, list) else []
                thinking = [
                    content_block_to_dict(block)
                    for block in blocks
                    if isinstance(block, (ThinkingBlock, RedactedThinkingBlock))
                ]
                if thinking:
                    writer.write(
                        AssistantEvent(
                            message={
                                "role": "assistant",
                                "content": thinking,
                            }
                        )
                    )
        except Exception:
            _LOGGER.exception(
                "Failed to persist message into conversation: role=%s",
                getattr(msg, "role", "?"),
            )
            raise

    active_evaluator_goal = _get_active_evaluator_goal(tool_context)
    effective_max_turns = options.max_turns
    if active_evaluator_goal is not None and not options.max_turns_explicit:
        effective_max_turns = 0

    # The query adapter appends model/goal output to the main transcript in
    # real time. Flush the already-ordered conversation first so command-side
    # facts (for example goal_set) and the user directive are guaranteed to
    # precede those outputs on disk. The adapter continues the parentUuid
    # chain from the last message passed below; Session.save() then dedupes the
    # same UUIDs at exit instead of appending the inputs after the outputs.
    save_transcript = getattr(session, "save_transcript", None)
    if callable(save_transcript):
        try:
            save_transcript()
        except Exception:
            _LOGGER.exception("Failed to persist headless input before agent loop")

    in_agent_loop.value = True
    try:
        from clawcodex_ext.coordinator.mode import coordinator_main_loop_registry

        try:
            main_loop_registry = coordinator_main_loop_registry(tool_registry)
        except (AttributeError, TypeError):
            main_loop_registry = tool_registry

        compat_result = asyncio.run(
            run_query_as_agent_loop(
                initial_messages=list(session.conversation.messages),
                provider=provider,
                tool_registry=main_loop_registry,
                tool_context=tool_context,
                system_prompt=effective_system_prompt,
                max_turns=effective_max_turns,
                fallback_model=options.fallback_model,
                thinking_effort=options.effort,
                pipeline_config=(pipeline_config_factory() if pipeline_config_factory is not None else None),
                query_source="sdk",
                on_event=on_event,
                on_text_chunk=on_text_chunk,
                on_message=_persist,
                on_attachment=lambda m: session.conversation.add_message(m.role, m.content),
                abort_controller=abort_controller,
            )
        )
    finally:
        in_agent_loop.value = False

    if (
        active_evaluator_goal is not None
        and options.max_turns_explicit
        and compat_result.terminal is not None
        and compat_result.terminal.reason == "max_turns"
    ):
        error = RuntimeError(f"Goal not achieved before --max-turns={options.max_turns}")
        error.aggregate_usage = dict(compat_result.usage or {})
        error.num_turns = compat_result.num_turns
        raise error

    return AgentLoopResult(
        response_text=compat_result.response_text,
        usage=(compat_result.usage if compat_result.num_turns > 0 else None),
        num_turns=compat_result.num_turns,
    )


def _wrap_cron_prompt(prompt: str, task_id: str, run_id: str) -> str:
    """Wrap a cron prompt with context so the LLM knows it's automated.

    This is the headless-side prompt wrapper passed to
    :class:`CronDispatchBridge`. It mirrors the previous module-level
    ``_wrap_cron_prompt(prompt, *, task_id=...)`` signature by
    accepting positional args in the new (prompt, task_id, run_id)
    order expected by the bridge.
    """
    from datetime import datetime

    _ = run_id  # unused — wrapper does not currently show run id
    now = datetime.now()
    # ``%-I`` is POSIX-only; Windows' strftime rejects it.
    time_str = now.strftime("%b %d %I:%M%p").lower().replace(" 0", " ")
    header = f"✻ Running scheduled task ({time_str})"
    if task_id:
        header += f" · {task_id}"
    return f"{header}\n\nThis prompt was generated automatically from a scheduled task.\n\n{prompt}"


def _drain_cron_outbox(
    tool_context: ToolContext,
    active_tasks: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Drain cron_prompt events from ``tool_context.outbox``.

    This delegates the typed-or-dict parsing and prompt wrapping to
    :class:`CronDispatchBridge`. The accumulation guard
    (duplicate task_id in ``active_tasks`` → cancel run) stays here
    because it depends on the per-run-loop closure dictionary, not
    on the bridge.

    Returns runnable prompts as ``(wrapped_prompt, task_id, run_id)``.
    """
    outbox = getattr(tool_context, "outbox", None)
    if not outbox:
        return []
    # Lazy import to avoid circular import with ``runtime.py`` at
    # module load time. ``dispatch.py`` itself imports from ``runs.py``
    # which is already loaded above.
    from clawcodex_ext.cron_system.dispatch import CronDispatchBridge

    bridge = CronDispatchBridge(
        tool_context.workspace_root,
        wrap_prompt=_wrap_cron_prompt,
    )
    events = bridge.drain(outbox)
    drained: list[tuple[str, str, str]] = []
    for event in events:
        if event.task_id and event.task_id in active_tasks:
            try:
                finalize_cron_run(tool_context.workspace_root, event.run_id, "cancelled")
            except Exception:  # nosec B110
                pass  # This side-channel notification is best-effort and must not alter the primary result.
            continue
        if event.task_id and event.run_id:
            active_tasks[event.task_id] = event.run_id
        drained.append((event.wrapped_prompt, event.task_id, event.run_id))
    return drained


def _extract_cron_task_id(user_input: str) -> str | None:
    first_line = user_input.split("\n", 1)[0]
    if not first_line.startswith("✻ Running scheduled task"):
        return None
    sep = " · "
    idx = first_line.find(sep)
    if idx == -1:
        return None
    task_id = first_line[idx + len(sep) :].strip()
    return task_id if task_id else None


def _claim_cron_task(
    workspace_root: Path,
    active_tasks: dict[str, str],
    task_id: str,
) -> str | None:
    run_id = active_tasks.get(task_id)
    if not run_id:
        return None
    try:
        claimed = claim_cron_run(workspace_root, run_id)
    except Exception:
        return run_id
    return claimed.id if claimed is not None else run_id


def _finalize_cron_task(
    workspace_root: Path,
    active_tasks: dict[str, str],
    task_id: str,
    status: str = "completed",
    *,
    error: str | None = None,
) -> None:
    run_id = active_tasks.pop(task_id, None)
    if not run_id:
        return
    try:
        finalize_cron_run(workspace_root, run_id, status, error=error)  # type: ignore[arg-type]
    except Exception:  # nosec B110
        pass  # This side-channel notification is best-effort and must not alter the primary result.


def _process_cron_outbox(
    tool_context: ToolContext,
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
            run_prompt(prompt, task_id, run_id)


def _jsonable(value):
    """Coerce arbitrary tool output into a JSON-safe shape."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        return str(value)
    except Exception:
        return repr(value)


def _run_resume_checks(
    options: HeadlessOptions,
    session: Any,
    provider_name: str,
    provider: Any,
    stderr: IO[str],
) -> None:
    """Run resume-time checks (C8 / C11 / R8).

    All checks are best-effort and non-fatal — failures are swallowed
    so a corrupt transcript or missing metadata file never blocks the
    run. Only fires when a session was actually resumed
    (``external_session`` / ``resume_session_id`` / ``fork_session_id``);
    fresh sessions have no prior state to compare against.
    """
    resumed = options.resume_session_id or options.fork_session_id or options.external_session is not None
    if not resumed or session is None:
        return
    try:
        from clawcodex_ext.agent.resume_checks import (
            restore_metadata_from_session,
            warn_provider_model_mismatch,
            warn_system_prompt_drift,
        )

        # C8 / R10: system-prompt drift.
        history_msgs = session.conversation.messages if getattr(session, "conversation", None) is not None else []
        warn_system_prompt_drift(
            history_msgs,
            getattr(options, "append_system_prompt", "") or "",
            stderr,
        )

        # C11 / R11: provider/model mismatch. The "original" provider /
        # model come from the resumed Session's recorded fields (set
        # by ``Session.load`` from the transcript's ``session_init``
        # line or legacy ``session.json``).
        orig_provider = str(getattr(session, "provider", "") or "")
        orig_model = str(getattr(session, "model", "") or "")
        cur_model = str(options.model or getattr(provider, "model", "") or "")
        warn_provider_model_mismatch(
            orig_provider,
            orig_model,
            provider_name,
            cur_model,
            stderr,
        )

        # R8: inherit title/tags from the source session. For
        # ``--resume`` the source is the same id; for ``--fork-session``
        # the source is the fork origin. ``restore_metadata_from_session``
        # is a no-op when the source has no metadata.
        source_sid = options.resume_session_id or options.fork_session_id or getattr(session, "session_id", None)
        if source_sid:
            agent_name = str(getattr(options.startup_agent, "name", "") if options.startup_agent is not None else "")
            restore_metadata_from_session(
                target_session_id=getattr(session, "session_id", ""),
                source_session_id=str(source_sid),
                agent_name=agent_name,
            )
    except Exception:
        _LOGGER.debug(
            "resume checks failed (non-fatal)",
            exc_info=True,
        )


def _warn_history_tool_conflicts(
    options: HeadlessOptions,
    session: Any,
    tool_registry: Any,
    stderr: IO[str],
) -> None:
    """Warn when ``--allowed-tools`` strips a tool the resumed
    history already references.

    Only fires when a session was actually resumed (``external_session``
    / ``resume_session_id`` / ``fork_session_id``). Walks the resumed
    conversation's ``tool_use`` blocks, collects tool names, and reports
    any name that is NOT present in the (post-filter) registry. The
    warning is best-effort: a malformed history or a missing
    ``conversation`` attribute is silently skipped — headless must never
    block on telemetry-shaped diagnostics.
    """
    if not (options.allowed_tools or options.disallowed_tools):
        return
    if session is None:
        return
    conversation = getattr(session, "conversation", None)
    if conversation is None:
        return
    messages = getattr(conversation, "messages", None)
    if not messages:
        return

    history_tools: set[str] = set()
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) or not isinstance(content, (list, tuple)):
            continue
        for block in content:
            name = _block_tool_use_name(block)
            if name:
                history_tools.add(name)

    if not history_tools:
        return

    try:
        registry_names = {str(t.name) for t in tool_registry.list_tools()}
    except Exception:
        return

    missing = sorted(n for n in history_tools if n not in registry_names)
    if not missing:
        return

    try:
        print(
            "warning: --allowed-tools/--disallowed-tools removed tool(s) "
            f"referenced in resumed history: {', '.join(missing)}. "
            "The model may see tool_use blocks it cannot re-invoke; "
            "consider re-running without the filter or re-reading the "
            "affected files.",
            file=stderr,
        )
    except Exception:  # nosec B110
        pass  # Warning output is best-effort and must not block startup.


def _block_tool_use_name(block: Any) -> str | None:
    """Return the ``tool_use`` block's tool name, or ``None`` if not a
    tool_use block / shape is unrecognised.
    """
    if isinstance(block, dict):
        if str(block.get("type", "")).lower() != "tool_use":
            return None
        name = block.get("name")
        return str(name) if name else None
    type_attr = getattr(block, "type", None)
    if type_attr is None or str(type_attr).lower() != "tool_use":
        return None
    name = getattr(block, "name", None)
    return str(name) if name else None
