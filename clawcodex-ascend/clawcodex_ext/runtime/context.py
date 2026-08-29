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

# pylint: disable=attribute-defined-outside-init,no-name-in-module,not-callable
"""Downstream RuntimeContext — unified provider/tool/session factory."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clawcodex_ext.cli.model_cmd.resolver import resolve
from clawcodex_ext.cron_system.runtime import attach_cron_runtime, replace_cron_tools
from clawcodex_ext.runtime.observer import (
    notify_observers,
)


@dataclass
class RuntimeOptions:
    """Options for building a RuntimeContext, merged from TUIOptions and HeadlessOptions."""

    provider_name: str | None = None
    model: str | None = None
    fallback_model: str | None = None
    effort: str | None = None
    prompt: str | None = None
    output_format: str = "text"
    input_format: str = "text"
    include_partial_messages: bool = False
    max_turns: int = 20
    max_turns_explicit: bool = False
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    workspace_root: Path | None = None
    stream: bool = True
    permission_mode: str = "default"
    is_bypass_permissions_mode_available: bool = False
    skip_permissions: bool = False  # backward-compat alias for headless
    resume_session_id: str | None = None
    resume_browse: bool = False
    fork_session_id: str | None = None
    resume_session_at: int | None = None  # S-R4-AT: message index to resume at
    verbose: bool = False
    append_system_prompt: str = ""
    agent_dir_override: Path | None = None
    gateway: bool = False
    gateway_origin: str | None = None
    gateway_sock: str | None = None
    startup_agent: Any | None = None
    bundle_path: Path | None = None
    record: str | None = None
    record_width: int | None = None
    record_height: int | None = None
    multimodel_group: str = ""
    worktree_session: Any | None = None


@dataclass
class RuntimeContext:
    """Unified runtime context carrying provider, tool registry, tool context, and session.

    Produced by :meth:`RuntimeContext.build`. Consumed by all frontends
    (REPL, TUI, headless) to avoid duplicating provider/tool/session setup.
    """

    provider: Any
    provider_name: str
    tool_registry: Any
    tool_context: Any
    session: Any | None
    workspace_root: Path
    options: RuntimeOptions
    # selection resolved by the CLI or a runtime slash command.
    multimodel_group: str = ""
    # ``resume_session_with_tail`` returns a TailFollower that
    # headless never iterates. Without an explicit release the follower
    # holds a reference to the session transcript path and keeps the
    # ``_offset`` / asyncio event state alive for the lifetime of the
    # RuntimeContext. Frontends that DO iterate (AgentBridge) set this
    # to None after consuming; headless calls :meth:`close_tail_follower`
    # in its finally block.
    tail_follower: Any | None = None
    # MCP runtime handle. Kept so frontends can close connections on exit.
    _mcp_manager: Any | None = field(default=None, repr=False)

    @classmethod
    def build(cls, options: RuntimeOptions) -> RuntimeContext:
        """Build a RuntimeContext from options.

        Unifies the provider/registry/context/session construction that was
        previously duplicated across headless.py, tui.py, and repl/core.py.
        """
        from src.permissions.types import ToolPermissionContext
        from src.providers.runtime import build_provider_from_config
        from src.tool_system.context import ToolContext
        from src.tool_system.defaults import build_default_registry

        workspace_root = options.workspace_root or Path.cwd()

        # Resolve effective permission mode (handle skip_permissions alias)
        if options.skip_permissions:
            effective_mode = "bypassPermissions"
            bypass_available = True
        else:
            effective_mode = options.permission_mode
            bypass_available = options.is_bypass_permissions_mode_available

        # Build provider
        resolution = resolve(
            cli_provider=options.provider_name,
            cli_model=options.model,
            project_root=workspace_root,
        )
        provider_name = resolution.provider
        options.provider_name = provider_name
        options.model = resolution.model

        # replace the ordinary provider only after resolving the base
        # runtime settings, keeping the core query loop unaware of ensembles.
        if options.multimodel_group:
            from clawcodex_ext.multimodel.config import default_config_path, load_config
            from clawcodex_ext.multimodel.factory import build_router

            group = load_config().groups[options.multimodel_group]
            provider = build_router(
                group,
                build_provider_from_config,
                audit_path=default_config_path().parent / "multimodel-audit.jsonl",
            )
            provider_name = "multimodel"
        else:
            provider = build_provider_from_config(provider_name, resolution.model)

        # Build tool registry
        tool_registry = build_default_registry(
            provider=provider,
            load_agent_tools=options.bundle_path is None,
        )

        replace_cron_tools(tool_registry)

        # Apply tool filtering
        _filter_registry(
            tool_registry,
            allowed=options.allowed_tools,
            denied=options.disallowed_tools,
        )

        # Build tool context
        tool_context = ToolContext(
            workspace_root=workspace_root,
            permission_context=ToolPermissionContext(
                mode=effective_mode,
                is_bypass_permissions_mode_available=bypass_available,
            ),
            tool_registry=tool_registry,
        )
        if effective_mode == "bypassPermissions":
            tool_context.allow_docs = True
        tool_context.options.is_non_interactive_session = False

        # Wire persistent cron scheduler to the tool context.
        # Runs a background daemon thread that checks for due tasks
        # every second and pushes cron_prompt events to the outbox.
        attach_cron_runtime(tool_context, autostart=True)

        # Wire the dreaming system (background memory consolidation).
        try:
            from clawcodex_ext.dreaming.runner import wire_real_dream_runner
            from clawcodex_ext.dreaming.service import init_auto_dream

            wire_real_dream_runner()
            init_auto_dream(registry=tool_context.runtime_tasks)
        except Exception:
            logging.getLogger(__name__).debug(
                "dreaming system wiring failed; dream feature may be unavailable",
                exc_info=True,
            )

        # Bootstrap configured MCP servers so tools like MCP / ListMcpResourcesTool
        # can actually call them in headless/print mode (and any other frontend
        # that builds a RuntimeContext). Failures are best-effort: if a server
        # is down or unconfigured, the rest of the runtime keeps working.
        mcp_manager = _bootstrap_mcp_sync(tool_context)
        if mcp_manager is not None:
            # Register per-server wrapped tools (e.g. mcp__server__tool_name)
            # into the registry so the model can see and call them directly.
            for mcp_tool in mcp_manager.all_tools():
                try:
                    tool_registry.register(mcp_tool)
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Failed to register MCP tool %s; skipping",
                        getattr(mcp_tool, "name", "<unknown>"),
                        exc_info=True,
                    )

        # Resume session if requested
        session = None
        tail_follower = None
        if options.resume_session_id:
            from clawcodex_ext.agent.session_ext import resume_session_with_tail

            session, tail_follower = resume_session_with_tail(
                options.resume_session_id,
            )

        # Fork session: load existing history into a new session (S-R4-F)
        if options.fork_session_id and not options.resume_session_id:
            from src.agent import Session as AgentSession

            old_session = AgentSession.resume(options.fork_session_id)
            if old_session is not None:
                # Create a brand new session
                new_session = AgentSession.create(
                    provider_name,
                    options.model or getattr(provider, "model", ""),
                )
                # Copy conversation messages from old session
                if old_session.conversation and old_session.conversation.messages:
                    new_session.conversation.messages = list(old_session.conversation.messages)
                session = new_session

        # S-R4-AT: truncate conversation to a specific message index
        if session is not None and options.resume_session_at is not None:
            idx = options.resume_session_at
            if session.conversation and session.conversation.messages:
                total = len(session.conversation.messages)
                if 0 <= idx < total:
                    session.conversation.messages = session.conversation.messages[: idx + 1]

        runtime = cls(
            provider=provider,
            provider_name=provider_name,
            tool_registry=tool_registry,
            tool_context=tool_context,
            session=session,
            workspace_root=workspace_root,
            options=options,
            tail_follower=tail_follower,
            _mcp_manager=mcp_manager,
        )
        runtime._single_provider_name = resolution.provider
        runtime._single_model = resolution.model
        attach_cron_runtime(runtime)
        return runtime

    def close_tail_follower(self) -> None:
        """Release the TailFollower obtained during resume.

        ``resume_session_with_tail`` returns a follower that headless
        never iterates — without an explicit release the follower keeps
        a reference to the transcript path and asyncio event state for
        the lifetime of the RuntimeContext. Best-effort: ``stop`` is
        async so we run it on a fresh event loop; any failure is
        swallowed.
        """
        follower = self.tail_follower
        if follower is None:
            return
        self.tail_follower = None
        try:
            stop = getattr(follower, "stop", None)
            if stop is None:
                return

            # Use a fresh event loop for the stop() coroutine. Avoids
            # the DeprecationWarning from ``get_event_loop()`` when no
            # loop is bound to the current thread.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(stop())
            finally:
                loop.close()
        except Exception:
            logging.getLogger(__name__).debug(
                "tail follower release failed (non-fatal)",
                exc_info=True,
            )

    def close(self) -> None:
        """Release all runtime resources held by this context.

        Headless/print mode calls this in a ``finally`` block so MCP server
        subprocesses and tail followers are cleaned up. Failures are best-effort.
        """
        self.close_tail_follower()
        manager = getattr(self, "_mcp_manager", None)
        if manager is None:
            return
        self._mcp_manager = None
        close_all = getattr(manager, "close_all", None)
        if close_all is None:
            return
        loop = getattr(manager, "_clawcodex_event_loop", None)
        try:
            if loop is not None and not loop.is_closed():
                loop.run_until_complete(close_all())
                loop.close()
            else:
                asyncio.run(close_all())
        except Exception:
            logging.getLogger(__name__).debug(
                "MCP manager cleanup failed (non-fatal)",
                exc_info=True,
            )

    def swap_provider(self, provider_name: str, model: str | None = None) -> None:
        from clawcodex_ext.cli.model_cmd.registry import ModelRegistry
        from clawcodex_ext.cli.model_cmd.errors import (
            ProviderMismatchError,
            UnknownModelError,
        )
        from clawcodex_ext.cli.provider_cmd.errors import UnknownProviderError
        from src.providers.runtime import build_provider_from_config
        from src.tool_system.defaults import build_default_registry

        # Runtime switching is validated by constructing the target provider.
        # Registry discovery hooks may perform network I/O and must not make a
        # local `/model <name>` switch wait for a catalog refresh first.
        registry = ModelRegistry(discovery_hooks={})
        try:
            provider_name = registry.validate_provider(provider_name)
        except UnknownProviderError:
            import sys

            print(
                f"Warning: provider '{provider_name}' is not in the built-in list — proceeding anyway",
                file=sys.stderr,
            )
        if model is not None:
            try:
                registry.validate_model(model, provider_name)
            except (UnknownModelError, ProviderMismatchError):
                pass  # Unknown model on unknown provider is fine

        provider = build_provider_from_config(provider_name, model)
        tool_registry = build_default_registry(
            provider=provider,
            load_agent_tools=getattr(self.options, "bundle_path", None) is None,
        )
        replace_cron_tools(tool_registry)
        _filter_registry(
            tool_registry,
            allowed=self.options.allowed_tools,
            denied=self.options.disallowed_tools,
        )

        self.provider = provider
        self.provider_name = provider_name
        self.tool_registry = tool_registry
        self.options.provider_name = provider_name
        self.options.model = getattr(provider, "model", model)
        self.multimodel_group = ""
        self.options.multimodel_group = ""
        self._single_provider_name = provider_name
        self._single_model = self.options.model

        for attr, value in (
            ("provider", provider),
            ("provider_name", provider_name),
            ("tool_registry", tool_registry),
        ):
            if hasattr(self.tool_context, attr):
                setattr(self.tool_context, attr, value)

        # Fan-out to downstream observers (REPL, TUI, AgentBridge).
        # See clawcodex_ext/runtime/observer.py for the Protocol contract.
        notify_observers(self)

    def swap_multimodel(self, group_name: str) -> None:
        """Activate a configured ensemble for the current interactive session."""
        from clawcodex_ext.multimodel.config import default_config_path, load_config
        from clawcodex_ext.multimodel.factory import build_router
        from src.providers.runtime import build_provider_from_config
        from src.tool_system.defaults import build_default_registry

        config = load_config()
        if group_name not in config.groups:
            raise ValueError(f"unknown model group '{group_name}'")
        provider = build_router(
            config.groups[group_name],
            build_provider_from_config,
            audit_path=default_config_path().parent / "multimodel-audit.jsonl",
        )
        tool_registry = build_default_registry(
            provider=provider,
            load_agent_tools=self.options.bundle_path is None,
        )
        replace_cron_tools(tool_registry)
        _filter_registry(tool_registry, allowed=self.options.allowed_tools, denied=self.options.disallowed_tools)
        self.provider = provider
        self.provider_name = "multimodel"
        self.tool_registry = tool_registry
        self.multimodel_group = group_name
        self.options.multimodel_group = group_name
        for attr, value in (
            ("provider", provider),
            ("provider_name", "multimodel"),
            ("tool_registry", tool_registry),
        ):
            if hasattr(self.tool_context, attr):
                setattr(self.tool_context, attr, value)
        notify_observers(self)

    def disable_multimodel(self) -> None:
        """Return to the single-provider selection that existed before the group."""
        provider_name = getattr(self, "_single_provider_name", None) or self.options.provider_name
        model = getattr(self, "_single_model", None) or self.options.model
        if not provider_name or provider_name == "multimodel":
            raise RuntimeError("no single-provider selection is available")
        self.multimodel_group = ""
        self.options.multimodel_group = ""
        self.swap_provider(provider_name, model)


def _bootstrap_mcp_sync(tool_context: Any) -> Any | None:
    """Best-effort bootstrap of configured MCP servers.

    Headless/print mode (and any frontend using RuntimeContext) needs active
    MCP clients in ``tool_context.mcp_clients`` so the ``MCP`` / ``MCPBatch`` /
    ``ListMcpResourcesTool`` / ``ReadMcpResourceTool`` tools can actually call
    configured servers. Returns the connection manager so callers can close it
    on exit, or ``None`` when no servers are configured or bootstrap fails.

    The manager's clients own stdio transports that are bound to the event
    loop they were created on. ``asyncio.run`` would close that loop before
    the tools get a chance to use them, so we create a dedicated loop here and
    keep it open for the lifetime of the runtime.
    """
    try:
        from clawcodex_ext.services.mcp import bootstrap_mcp_runtime
    except Exception as exc:
        logging.getLogger(__name__).debug("MCP runtime unavailable: %s", exc)
        return None

    loop = asyncio.new_event_loop()
    try:
        manager = loop.run_until_complete(bootstrap_mcp_runtime(prefetch_claudeai=False))
    except Exception as exc:
        logging.getLogger(__name__).debug("MCP bootstrap failed: %s", exc)
        try:
            loop.close()
        except Exception:  # nosec B110
            pass  # Cleanup is best-effort and must not replace the primary operation result.
        return None

    # Attach the loop so tool calls can run on the same loop that owns the
    # stdio transports, and so shutdown can close it cleanly.
    manager._clawcodex_event_loop = loop  # type: ignore[attr-defined]

    clients = getattr(manager, "_clients", None)
    if clients:
        tool_context.mcp_clients = clients
        tool_context.mcp_manager_loop = loop
    return manager


def _filter_registry(
    registry,
    *,
    allowed: tuple[str, ...] = (),
    denied: tuple[str, ...] = (),
) -> None:
    """Filter tool registry by allowed/denied name sets.

    Moved from src/entrypoints/tui.py so RuntimeContext.build() can use it
    without importing from the TUI entrypoint.
    """
    names = [t.name for t in registry.list_tools()]
    for name in names:
        should_remove = False
        if allowed and name.lower() not in {n.lower() for n in allowed}:
            should_remove = True
        if denied and name.lower() in {n.lower() for n in denied}:
            should_remove = True
        if should_remove:
            try:
                registry.unregister(name)
            except Exception:
                try:
                    del registry._tools[name]  # type: ignore[attr-defined]
                except Exception as exc:  # nosec B110
                    logging.getLogger(__name__).warning(
                        "Failed to remove filtered tool %s (%s)",
                        name,
                        type(exc).__name__,
                    )
