# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Slash-command adapter for the Textual TUI.

The legacy :class:`src.repl.core.ClawcodexREPL` has its own slash-command
dispatcher that mixes local built-ins (``/exit``, ``/tools``, …) with the
newer :mod:`src.command_system` registry (``/help``, ``/clear``, …) and
``PromptCommand`` slash commands like ``/init`` that expand into model
prompts. The TUI reuses the same :mod:`src.command_system` registry so
the two UIs stay feature-aligned; this module is the thin adapter that
glues the registry to Textual's message pump without pulling
``prompt_toolkit`` into the TUI import graph.

Built-ins that are not yet covered by the command registry (``/exit``,
``/quit``, ``/repl``, ``/clear``, ``/tools``, ``/help``, ``/tui``) stay
in-app so the TUI continues to work when the command registry fails to
initialise.
"""

from __future__ import annotations
# pylint: disable=E0611

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from src.agent import Session


# Local slash commands (resolved inside the Textual app, never go through
# the command registry). Keep aligned with
# :attr:`src.repl.core.ClawcodexREPL._original_built_ins` so the two REPLs
# advertise the same surface.
LOCAL_BUILTINS: tuple[str, ...] = (
    "/help",
    "/exit",
    "/quit",
    "/q",
    "/repl",
    "/clear",
    "/tools",
    "/stream",
    "/render-last",
    "/skills",
    # Phase 2 dialogs:
    "/model",
    "/effort",
    "/history",
    "/cost",
    "/idle",
    "/theme",
    # Phase 3 dialogs:
    "/diff",
    "/mcp",
    "/tasks",
    "/rewind",
    "/resume",
    # Phase 4 dialogs:
    "/permission",
)


@dataclass
class CommandDispatchResult:
    """What the TUI should do after a slash command was handled.

    * ``handled`` — command was fully resolved locally, no agent call.
    * ``prompt_text`` — the command resolved into a prompt (e.g. ``/init``)
      that should be forwarded to the agent as a user turn.
    * ``assistant_text`` — a forked skill's completed result, rendered without
      forwarding it through the main agent again.
    * ``system_text`` — text to append to the transcript as a system
      message (from commands that emit a textual response).
    * ``should_query`` — start the command-owned continuation after rendering
      the local result (used by ``/goal <condition>``).
    * ``open_dialog`` — name of a Phase 2 dialog screen to push
      (``model``, ``effort``, ``history``, ``cost``, ``idle``, ``exit``,
      ``theme``). The app resolves the name to a concrete
      :class:`DialogScreen` subclass.
    * ``error`` — surfaced to the user as a red system line.
    """

    handled: bool
    prompt_text: str | None = None
    assistant_text: str | None = None
    assistant_name: str | None = None
    system_text: str | None = None
    system_render: str = "plain"
    should_query: bool = False
    transient: bool = False
    open_dialog: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class CommandSuggestion:
    """A rich slash-command entry shown in the completion popup.

    Mirrors the TS ``SuggestionItem`` shape just enough to render the
    two-column ``/name  description`` layout. ``tag`` is an optional
    short label (e.g. ``"workflow"``) and ``aliases`` is the list of
    alternative names — both surface in the display row.
    ``takes_args`` is true when the command accepts argument context
    (e.g. ``/model gpt-4`` or ``/theme dark``), in which case selecting
    the command appends a trailing space instead of submitting.
    """

    name: str  # without the leading slash
    description: str = ""
    aliases: tuple[str, ...] = ()
    tag: str | None = None
    source: str = "builtin"
    takes_args: bool = False

    @property
    def slash(self) -> str:
        return f"/{self.name}"


_LOCAL_BUILTIN_DESCRIPTIONS: dict[str, str] = {
    "/help": "Show available slash commands",
    "/exit": "Exit the CLI",
    "/quit": "Exit the CLI",
    "/q": "Exit the CLI",
    "/repl": "Return to CLI / terminal (keep session in background)",
    "/clear": "Clear the conversation transcript",
    "/tools": "List available tools",
    "/stream": "Toggle streaming output",
    "/render-last": "Re-render the last assistant message",
    "/skills": "List discovered skills",
    "/model": "Show current model and available list, or switch to a named model",
    "/effort": "Choose reasoning effort",
    "/history": "Open prompt history",
    "/cost": "Show session token + cost usage",
    "/idle": "Configure idle-mode preferences",
    "/theme": "Change the color theme",
    "/diff": "Show pending diff",
    "/mcp": "Manage MCP servers",
    "/tasks": "Browse background tasks",
    "/rewind": "Rewind conversation to an earlier turn",
    "/resume": "Resume a saved session",
    "/permission": "Change permission mode",
}


def build_command_suggestions(
    workspace_root: Path,
    tool_context: Any | None = None,
) -> list[CommandSuggestion]:
    """Return rich slash-command entries for the completion popup.

    Mirrors :func:`build_command_words` but carries descriptions,
    aliases, and tags so the popup can render the same two-column
    layout the TypeScript reference uses.
    """

    out: list[CommandSuggestion] = []
    seen: set[str] = set()

    def push(sugg: CommandSuggestion) -> None:
        key = sugg.name.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(sugg)

    for word in LOCAL_BUILTINS:
        push(
            CommandSuggestion(
                name=word[1:],
                description=_LOCAL_BUILTIN_DESCRIPTIONS.get(word, ""),
                source="builtin",
            )
        )

    try:
        from src.command_system.builtins import get_builtin_commands
        from src.command_system.registry import CommandRegistry

        from clawcodex_ext.away_summary.registration import register_away_summary_commands
        from clawcodex_ext.intent_forecast.registration import register_intent_forecast_commands
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.multimodel.runtime_command import register_multimodel_runtime_command

        # Use a fresh private registry so we don't clobber the global
        # registry's LocalCommand for /model and /provider (F-43).
        # register_builtin_commands(None) would overwrite them with
        # InteractiveCommand / PromptCommand variants that can't run
        # through execute_command_sync.
        private_reg = CommandRegistry()
        for cmd in get_builtin_commands():
            private_reg.register(cmd)
        register_away_summary_commands(private_reg)
        register_intent_forecast_commands(private_reg)
        register_runtime_commands(private_reg)
        # Keep the REPL/TUI completion palette aligned with the runtime
        # registry.  ``/multimodel`` is a local command, so it must be
        # registered on this private completion registry as well.
        register_multimodel_runtime_command(private_reg)

        # F-53: auto-expose non-core tools as /<tool-name> slash commands
        # in the TUI command registry. Mirrors the REPL wiring in
        # ``clawcodex_ext/repl/app.py`` / ``clawcodex_ext/repl/core.py``.
        tool_command_names: set[str] = set()
        try:
            from clawcodex_ext.cli.tool_cmd import register_tool_commands

            names_before_tools = {command.name.lower() for command in private_reg.list_commands(include_disabled=True)}
            register_tool_commands(private_reg)
            tool_command_names = {
                command.name.lower()
                for command in private_reg.list_commands(include_disabled=True)
                if command.name.lower() not in names_before_tools
            }
        except Exception:  # nosec B110
            pass
        for cmd in private_reg.list_commands(include_disabled=True):
            if getattr(cmd, "is_hidden", False):
                continue
            if cmd.name not in {"recap", "forecast"} and not cmd.is_enabled():
                continue
            aliases = tuple(getattr(cmd, "aliases", []) or [])
            tag = "workflow" if getattr(cmd, "kind", None) == "workflow" else None
            push(
                CommandSuggestion(
                    name=cmd.name,
                    description=getattr(cmd, "description", "") or "",
                    aliases=aliases,
                    tag=tag,
                    source=(
                        "tools"
                        if cmd.name.lower() in tool_command_names
                        else getattr(cmd, "loaded_from", "builtin") or "builtin"
                    ),
                )
            )
    except Exception:  # nosec B110
        pass

    try:
        from src.skills.loader import get_all_skills

        cwd = getattr(tool_context, "cwd", None) or workspace_root
        for skill in get_all_skills(project_root=cwd):
            push(
                CommandSuggestion(
                    name=skill.name,
                    description=getattr(skill, "description", "") or "",
                    source="skills",
                )
            )
    except Exception:  # nosec B110
        pass

    return out


def build_command_words(
    workspace_root: Path,
    tool_context: Any | None = None,
) -> list[str]:
    """Return the flat list of slash words shown in the completion popup.

    Kept as a backwards-compatible wrapper around
    :func:`build_command_suggestions`. New callers should prefer the
    rich variant so descriptions surface in the popup.
    """

    words: list[str] = []
    seen: set[str] = set()
    for sugg in build_command_suggestions(workspace_root, tool_context):
        for w in (sugg.slash, *(f"/{a}" for a in sugg.aliases)):
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(w)
    return words


def dispatch_local_command(
    text: str,
    *,
    session: "Session",
    workspace_root: Path,
    tool_registry: Any,
) -> CommandDispatchResult:
    """Resolve a slash command that the TUI handles directly.

    Returns a :class:`CommandDispatchResult` describing whether the
    command was handled and how the screen should react. Commands that
    need async dispatch (PromptCommand like ``/init``, or any
    registry-backed command) are NOT handled here — callers should fall
    through to :func:`dispatch_registry_command` for those.
    """

    raw = text.strip()
    if not raw.startswith("/"):
        return CommandDispatchResult(handled=False)

    name = raw.split(" ", 1)[0].lower()

    if name in ("/exit", "/quit", "/q"):
        return CommandDispatchResult(handled=True, system_text="__exit__")
    if name == "/repl":
        return CommandDispatchResult(handled=True, system_text="__repl__")
    if name == "/clear":
        return CommandDispatchResult(handled=True, system_text="__clear__")
    if name == "/help":
        lines = [
            "Slash commands:",
            "  " + "  ".join(LOCAL_BUILTINS),
            "Tip: press `/` in the prompt to open the palette.",
            "Exit with /exit or Ctrl+D. /repl returns to CLI in background.",
        ]
        return CommandDispatchResult(handled=True, system_text="\n".join(lines))
    if name == "/tools":
        try:
            names = sorted(t.name for t in tool_registry.list_tools())
            return CommandDispatchResult(
                handled=True,
                system_text="Available tools: " + ", ".join(names),
            )
        except Exception as exc:
            return CommandDispatchResult(handled=True, error=f"/tools: {exc}")
    if name == "/skills":
        try:
            from src.skills.loader import get_all_skills

            cwd = workspace_root
            skills = list(get_all_skills(project_root=cwd))
            skills.sort(key=lambda s: s.name.lower())
            if not skills:
                return CommandDispatchResult(
                    handled=True,
                    system_text="No skills found.\nCreate skills in ~/.clawcodex/skills/ or .clawcodex/skills/ in your project.",
                )
            from collections import defaultdict

            by_source: dict[str, list] = defaultdict(list)
            for s in skills:
                loaded = getattr(s, "loaded_from", "") or "unknown"
                by_source[loaded].append(s)
            lines = [f"Available Skills ({len(skills)}):"]
            for source in sorted(by_source.keys()):
                source_skills = by_source[source]
                lines.append(f"\n{source.title()} Skills:")
                for s in source_skills:
                    desc = (s.description or "").strip()
                    tag = f" — {desc}" if desc else ""
                    lines.append(f"  /{s.name}{tag}")
            return CommandDispatchResult(
                handled=True,
                system_text="\n".join(lines),
            )
        except Exception as exc:
            return CommandDispatchResult(handled=True, error=f"/skills: {exc}")
    if name == "/stream":
        parts = raw.split(maxsplit=1)
        if len(parts) == 1:
            return CommandDispatchResult(handled=True, system_text="__stream_status__")
        action = parts[1].strip().lower()
        if action in ("on", "true", "1", "enable", "enabled"):
            return CommandDispatchResult(handled=True, system_text="__stream_on__")
        elif action in ("off", "false", "0", "disable", "disabled"):
            return CommandDispatchResult(handled=True, system_text="__stream_off__")
        elif action == "toggle":
            return CommandDispatchResult(handled=True, system_text="__stream_toggle__")
        else:
            return CommandDispatchResult(handled=True, error="Usage: /stream [on|off|toggle]")
    if name == "/tui":
        # Already in TUI — no-op message instead of silently handing off.
        return CommandDispatchResult(
            handled=True,
            system_text="Already running in Textual TUI.",
        )

    # Phase 2 dialogs: the command itself has no state to resolve here,
    # it just asks the app to push the corresponding modal screen.
    if name == "/model" and raw.lower() == "/model":
        return CommandDispatchResult(handled=True, open_dialog="model")
    if name == "/effort":
        return CommandDispatchResult(handled=True, open_dialog="effort")
    if name in ("/history", "/hist"):
        return CommandDispatchResult(handled=True, open_dialog="history")
    if name == "/cost":
        return CommandDispatchResult(handled=True, open_dialog="cost")
    if name == "/idle":
        return CommandDispatchResult(handled=True, open_dialog="idle")
    if name == "/theme":
        return CommandDispatchResult(handled=True, open_dialog="theme")
    if name == "/diff":
        return CommandDispatchResult(handled=True, open_dialog="diff")
    if name == "/mcp":
        return CommandDispatchResult(handled=True, open_dialog="mcp")
    if name == "/tasks":
        return CommandDispatchResult(handled=True, open_dialog="tasks")
    if name == "/rewind":
        return CommandDispatchResult(handled=True, open_dialog="rewind")
    if name == "/resume":
        return CommandDispatchResult(handled=True, open_dialog="resume")
    if name == "/permission":
        return CommandDispatchResult(handled=True, open_dialog="permission")
    if name == "/forecast":
        parts = raw.split(maxsplit=1)
        action = parts[1].strip().lower() if len(parts) > 1 else ""
        if action in ("", "run"):
            return CommandDispatchResult(handled=True, open_dialog="forecast")

    return CommandDispatchResult(handled=False)


async def dispatch_registry_command(
    text: str,
    *,
    command_context: Any,
) -> CommandDispatchResult:
    """Resolve a slash command through the :mod:`src.command_system` async
    path. Returns ``handled=False`` when the command is unknown so the
    TUI can fall back to forwarding the text as a regular prompt.
    """

    raw = text.strip()
    if not raw.startswith("/"):
        return CommandDispatchResult(handled=False)

    parts = raw[1:].split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    try:
        from src.command_system.builtins import execute_command_async

        result = await execute_command_async(name, args, command_context)
    except Exception as exc:
        return CommandDispatchResult(handled=False, error=f"/{name}: {exc}")

    if not result.success:
        if result.error and "Unknown command" in result.error:
            return CommandDispatchResult(handled=False)
        return CommandDispatchResult(handled=True, error=result.error)

    if result.result_type == "text":
        if result.display == "assistant":
            return CommandDispatchResult(
                handled=True,
                assistant_text=result.text or "",
                assistant_name=result.command_name,
            )
        return CommandDispatchResult(
            handled=True,
            system_text=result.text or "",
            system_render="markdown" if name in {"recap", "forecast"} else "plain",
            should_query=bool(result.should_query),
            transient=bool(getattr(result, "transient", False)),
        )

    if result.result_type == "prompt":
        prompt_text = ""
        for item in result.prompt_content or []:
            if isinstance(item, dict) and item.get("type") == "text":
                prompt_text = item.get("text", "") or ""
                break
        return CommandDispatchResult(
            handled=True,
            prompt_text=prompt_text or None,
        )

    if result.result_type == "skip":
        return CommandDispatchResult(handled=True)

    return CommandDispatchResult(handled=True)
