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

"""
Command execution engine for Claw Codex.

Handles execution of commands and integration with the REPL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .registry import CommandRegistry
from .types import (
    CommandContext,
    CommandType,
    InteractiveCommand,
    InteractiveOutcome,
    InteractiveUnavailableError,
    LocalCommand,
    NullUIHost,
    PromptCommand,
    SkillPromptCommand,
    attach_downstream_context,
)


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool
    command_name: str
    result_type: str = "text"  # "text" | "prompt" | "skip"
    text: str = ""
    prompt_content: list[dict[str, Any]] = field(default_factory=list)
    prompt_is_meta: bool = False
    should_query: bool = False
    display: str = "system"  # "skip" | "system" | "user" | "assistant"
    meta_messages: list[str] = field(default_factory=list)
    error: str | None = None
    # propagated from InteractiveOutcome.scrollable — surfaces that
    # recognise scrollable rendering (currently: REPL) branch off their
    # normal text path; others ignore the flag and print ``text`` as usual.
    scrollable: bool = False
    transient: bool = False

    @classmethod
    def success_text(cls, command_name: str, text: str) -> "CommandResult":
        """Create a successful text result."""
        return cls(
            success=True,
            command_name=command_name,
            result_type="text",
            text=text,
            display="system",
        )

    @classmethod
    def success_assistant(cls, command_name: str, text: str) -> "CommandResult":
        """Create a completed assistant result that must not be queried again."""
        return cls(
            success=True,
            command_name=command_name,
            result_type="text",
            text=text,
            prompt_is_meta=True,
            should_query=False,
            display="assistant",
        )

    @classmethod
    def success_prompt(
        cls,
        command_name: str,
        prompt_content: list[dict[str, Any]],
        should_query: bool = True,
        prompt_is_meta: bool = False,
    ) -> "CommandResult":
        """Create a successful prompt result."""
        return cls(
            success=True,
            command_name=command_name,
            result_type="prompt",
            prompt_content=prompt_content,
            prompt_is_meta=prompt_is_meta,
            should_query=should_query,
            display="user",
        )

    @classmethod
    def failure(cls, command_name: str, error: str) -> "CommandResult":
        """Create an error result."""
        return cls(
            success=False,
            command_name=command_name,
            result_type="text",
            text=f"Error: {error}",
            error=error,
            display="system",
        )

    @classmethod
    def skip(cls, command_name: str) -> "CommandResult":
        """Create a skip result."""
        return cls(
            success=True,
            command_name=command_name,
            result_type="skip",
            display="skip",
        )


@dataclass
class CommandEngine:
    """Command execution engine."""

    registry: CommandRegistry
    workspace_root: Path
    context: CommandContext | None = None
    _command_hooks: list[Callable[[str, CommandResult], None]] = field(default_factory=list)

    async def execute(
        self,
        command_input: str,
    ) -> CommandResult:
        """
        Execute a command.

        Args:
            command_input: Command input (e.g., "/help args")

        Returns:
            CommandResult with the execution result
        """
        # Parse command and args
        if not command_input.startswith("/"):
            return CommandResult.failure(
                "",
                "Commands must start with '/'",
            )

        parts = command_input[1:].split(maxsplit=1)
        command_name = parts[0].strip()
        args = parts[1].strip() if len(parts) > 1 else ""

        # Get command
        command = self.registry.get(command_name)
        if command is None:
            return CommandResult.failure(
                command_name,
                f"Unknown command: {command_name}",
            )

        # Check if command is enabled
        if not command.is_enabled():
            return CommandResult.failure(
                command_name,
                f"Command {command_name} is disabled",
            )

        # Execute based on type
        result: CommandResult
        if command.command_type == CommandType.LOCAL:
            result = await self._execute_local(command, args)
        elif command.command_type == CommandType.PROMPT:
            result = await self._execute_prompt(command, args)
        elif command.command_type == CommandType.INTERACTIVE:
            result = await self._execute_interactive(command, args)
        else:
            result = CommandResult.failure(
                command_name,
                f"Unknown command type: {command.command_type}",
            )

        # Run hooks
        for hook in self._command_hooks:
            try:
                hook(command_name, result)
            except Exception:  # nosec B110
                # Don't let hook failures break command execution
                pass

        return result

    async def _execute_local(
        self,
        command: LocalCommand,
        args: str,
    ) -> CommandResult:
        """Execute a local command."""
        try:
            local_result = await command.call(args, self.context)

            if local_result.type == "skip":
                return CommandResult.skip(command.name)
            if local_result.type == "prompt":
                return CommandResult.success_prompt(
                    command.name,
                    [{"type": "text", "text": local_result.value}],
                    should_query=True,
                )

            display_text = local_result.display_text or local_result.value
            if local_result.type == "compact":
                # C3b: preserve the compact result type so UI surfaces can
                # render a boundary row (TS CompactBoundaryMessage) instead
                # of a plain system line. text still carries the
                # user_display_message for surfaces that don't special-case.
                return CommandResult(
                    success=True,
                    command_name=command.name,
                    result_type="compact",
                    text=display_text,
                    display="system",
                )
            return CommandResult.success_text(
                command.name,
                display_text,
            )
        except Exception as e:
            return CommandResult.failure(
                command.name,
                str(e),
            )

    async def _execute_prompt(
        self,
        command: PromptCommand,
        args: str,
    ) -> CommandResult:
        """Execute a prompt command."""
        try:
            prompt_content = await command.get_prompt_for_command(args, self.context)
            if isinstance(command, SkillPromptCommand) and command.context == "fork":
                result_text = next(
                    (
                        item.get("text", "")
                        for item in prompt_content
                        if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
                    ),
                    "",
                )
                if not result_text:
                    return CommandResult.failure(
                        command.name,
                        f"Forked skill produced empty result: {command.name}",
                    )
                conversation = getattr(self.context, "conversation", None)
                add_assistant = getattr(conversation, "add_assistant_message", None)
                if callable(add_assistant):
                    add_assistant(result_text)  # pylint: disable=not-callable
                return CommandResult.success_assistant(command.name, result_text)
            return CommandResult.success_prompt(
                command.name,
                prompt_content,
                should_query=True,
                prompt_is_meta=isinstance(command, SkillPromptCommand),
            )
        except Exception as e:
            return CommandResult.failure(
                command.name,
                str(e),
            )

    async def _execute_interactive(
        self,
        command: InteractiveCommand,
        args: str,
    ) -> CommandResult:
        """Execute an interactive command (port of TS ``local-jsx``).

        Runs the command body against ``ctx.ui`` and maps its
        :class:`InteractiveOutcome` onto a ``CommandResult``, propagating the
        ``display`` / ``should_query`` / ``meta_messages`` fields that the
        LOCAL arm hardcodes away (``_execute_local`` forces ``system`` /
        ``False`` / drops meta). A surface that wired no ``ui`` gets a
        ``NullUIHost`` so the body can always assume ``ctx.ui`` exists; that
        host raises :class:`InteractiveUnavailableError` for mutating prompts,
        which we surface as a clean error result.
        """
        # Substitute the null surface when none was wired (SDK /
        # non-interactive). Done here, once, so command bodies never see a
        # ``None`` ui. Idempotent: a real surface sets ``ui`` at startup.
        ctx = self.context
        if ctx is None:
            # Lazy default context for callers (e.g. unit tests) that
            # did not supply one. Constructed on demand so the engine
            # still works in headless smoke checks.
            from clawcodex_ext.command_system.types import CommandContext  # pylint: disable=redefined-outer-name

            ctx = CommandContext(
                workspace_root=self.workspace_root,
                cwd=self.workspace_root,
            )
            self.context = ctx
        if ctx.ui is None:
            ctx.ui = NullUIHost()

        try:
            outcome = await command.run(args, ctx)
        except InteractiveUnavailableError as e:
            # Expected on the null surface — a clean, typed message rather
            # than a stack trace.
            return CommandResult.failure(command.name, str(e))
        except Exception as e:
            return CommandResult.failure(command.name, str(e))

        if not isinstance(outcome, InteractiveOutcome):
            return CommandResult.failure(
                command.name,
                f"interactive command returned {type(outcome).__name__}, expected InteractiveOutcome",
            )

        # ``display == 'skip'`` (e.g. the cancelled path) → no output.
        if outcome.display == "skip":
            return CommandResult.skip(command.name)

        return CommandResult(
            success=True,
            command_name=command.name,
            result_type="text",
            text=outcome.message or "",
            should_query=outcome.should_query,
            display=outcome.display,
            meta_messages=list(outcome.meta_messages),
            # thread the scrollable hint onto the result so
            # surface adapters (REPL _handle_command_result) can branch
            # into a keyboard-scrolled view instead of a flat print.
            scrollable=outcome.scrollable,
            transient=outcome.transient,
        )

    def add_command_hook(
        self,
        hook: Callable[[str, CommandResult], None],
    ) -> None:
        """Add a command execution hook."""
        self._command_hooks.append(hook)

    def remove_command_hook(
        self,
        hook: Callable[[str, CommandResult], None],
    ) -> None:
        """Remove a command execution hook."""
        if hook in self._command_hooks:
            self._command_hooks.remove(hook)


def create_command_context(
    workspace_root: str | Path,
    conversation: Any = None,
    cost_tracker: Any = None,
    history: Any = None,
    cwd: str | Path | None = None,
    config: dict[str, Any] | None = None,
    app_state_store: Any = None,
    provider: Any = None,
    ui: Any = None,
    tool_registry: Any = None,
    tool_context: Any = None,
    runtime_context: Any = None,
) -> CommandContext:
    """
    Create a command context.

    Args:
        workspace_root: Root directory of the workspace
        conversation: Conversation object
        cost_tracker: Cost tracker object
        history: History log object
        cwd: Current working directory (defaults to workspace_root)
        config: Optional configuration dict
        app_state_store: Optional reactive AppState store. Commands that
            mutate global session state (e.g. /advisor) need this.
        provider: Optional active LLM provider. Commands that gate on
            provider type (e.g. /advisor) need this.
        ui: Optional ``UIHost`` interaction port. Interactive commands drive
            it; when None the engine substitutes a ``NullUIHost``.
        tool_registry: Optional active tool registry for downstream commands.
        tool_context: Optional active tool execution context for downstream commands.
        runtime_context: Optional active runtime context for downstream commands.

    Returns:
        CommandContext instance
    """
    root = Path(workspace_root).expanduser().resolve()
    current = Path(cwd).expanduser().resolve() if cwd is not None else root

    context = CommandContext(
        workspace_root=root,
        cwd=current,
        conversation=conversation,
        cost_tracker=cost_tracker,
        history=history,
        config=config or {},
        app_state_store=app_state_store,
        provider=provider,
        ui=ui,
        tool_context=tool_context,
    )
    attach_downstream_context(
        context,
        tool_registry=tool_registry,
        # tool_context was promoted to a real CommandContext field above; we
        # still call attach_downstream_context so other downstream code paths
        # that read it via getattr keep working.
        tool_context=tool_context,
        runtime_context=runtime_context,
    )
    return context
