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

"""DynamicToolCommand — F-53 adapter from a single ``Tool`` to a REPL/TUI
``LocalCommand``.

Each ``DynamicToolCommand`` wraps a tool's *snapshot* (name, schema,
description) captured at registration time. At invocation time, the
command resolves the *current* tool from ``context.tool_registry`` — so
if the runtime tool is replaced (resume, swap_provider), the command
still works. If the tool is no longer present, the command returns a
friendly error rather than crashing.

This decoupling also makes registration cheap: we don't need to hold
strong references to tool objects, and the same ``DynamicToolCommand``
instance works across the lifetime of the REPL/TUI session even as the
underlying tool is re-registered.
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from typing import TYPE_CHECKING, Any, Callable, Mapping

from clawcodex_ext.command_system.types import LocalCommand, LocalCommandResult

from . import core_filter, schema_parser

if TYPE_CHECKING:
    from clawcodex_ext.tool_system.build_tool import Tool
    from clawcodex_ext.tool_system.registry import ToolRegistry

log = logging.getLogger(__name__)


def _format_output(output: Any) -> str:
    """Format a tool's output for display as a LocalCommandResult.

    Tools can return strings, dicts, lists, or any JSON-serializable
    value. We render dicts / lists as pretty JSON; everything else goes
    through ``str()``.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (dict, list, tuple, int, float, bool)):
        try:
            return json.dumps(output, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(output)
    return str(output)


def _format_error(tool_name: str, message: str) -> str:
    return f"Tool '{tool_name}' failed: {message}"


class DynamicToolCommand:
    """Wrap a single tool as a REPL/TUI ``LocalCommand``.

    The instance is mostly stateless; the heavy data (schema, name,
    description) is captured in :attr:`local_command` and the
    :class:`argparse.ArgumentParser` for parsing.
    """

    def __init__(
        self,
        tool: "Tool",
        *,
        tool_resolver: Callable[[str, "ToolRegistry | None"], "Tool | None"] | None = None,
    ) -> None:
        self._tool_name = tool.name
        self._tool_aliases: tuple[str, ...] = tuple(getattr(tool, "aliases", ()) or ())
        self._schema: Mapping[str, Any] | None = tool.input_schema if isinstance(tool.input_schema, Mapping) else None
        self._description: str = self._derive_description(tool)
        self._argument_hint: str = self._derive_argument_hint(tool)
        self._parser: argparse.ArgumentParser = schema_parser.build_arg_parser(tool.name, self._schema)
        self._tool_resolver = tool_resolver or _default_tool_resolver
        self.local_command: LocalCommand = self._build_local_command()

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def argument_hint(self) -> str:
        return self._argument_hint

    @staticmethod
    def _derive_description(tool: "Tool") -> str:
        try:
            desc = tool.description({})
        except Exception:  # noqa: BLE001
            desc = ""
        if not isinstance(desc, str) or not desc.strip():
            desc = f"Dynamically exposed tool '{tool.name}' (F-53)."
        return textwrap.shorten(desc, width=200, placeholder="…")

    @staticmethod
    def _derive_argument_hint(tool: "Tool") -> str:
        schema = getattr(tool, "input_schema", None)
        if not isinstance(schema, Mapping):
            return "".strip() or " "  # placeholder
        properties = schema.get("properties") or {}
        if not isinstance(properties, Mapping) or not properties:
            return ""
        required = set(schema.get("required") or ())
        parts: list[str] = []
        for name, prop in properties.items():
            if not isinstance(prop, Mapping):
                prop = {}
            if name in required:
                marker = ""
            elif "default" in prop:
                marker = "?"
            else:
                marker = "?"
            kind = prop.get("type", "string")
            if kind == "boolean":
                parts.append(f"--{name}{marker}")
            else:
                parts.append(f"--{name}{marker} <{kind}>")
        if len(parts) <= 4:
            return " ".join(parts)
        return " ".join(parts[:4]) + " …"

    def _build_local_command(self) -> LocalCommand:
        lc = LocalCommand(
            name=self._tool_name,
            description=self._description,
            argument_hint=self._argument_hint,
            is_hidden=False,
        )
        lc.set_call(self._call)
        return lc

    # --- Invocation ----------------------------------------------------

    def _call(self, args: str, context: Any) -> LocalCommandResult:
        """LocalCommand entry point. Parses *args* and dispatches.

        *context* is a :class:`CommandContext`; we look up the tool via
        ``context.tool_registry`` (set by ``attach_downstream_context``).
        Falls back to the global default registry if not available.
        """
        argv = self._tokenize(args)
        registry = getattr(context, "tool_registry", None)
        tool = self._tool_resolver(self._tool_name, registry)
        if tool is None:
            return _text_result(
                self._tool_name,
                f"Tool '{self._tool_name}' is not registered in the current session.",
            )

        # Re-derive schema from the *live* tool so renames / migrations
        # take effect without a REPL restart. Falls back to the snapshot.
        schema = tool.input_schema if isinstance(tool.input_schema, Mapping) else self._schema
        try:
            parsed = schema_parser.parse_tool_args(self._tool_name, schema, argv)
        except argparse.ArgumentError as exc:
            usage = (exc.message or str(exc)).strip() or "invalid arguments"
            return _text_result(self._tool_name, usage)

        # Drop our internal bookkeeping before dispatching.
        extras = parsed.pop("_extra", None) if isinstance(parsed, dict) else None
        if extras:
            # Forward unknown args as a list under the first non-required
            # array property, or stash in ``_unknown_args`` for callers
            # that want to handle it. We don't error out — extra args
            # usually mean the tool's schema was extended after we
            # built the parser.
            parsed["_unknown_args"] = list(extras)

        tool_context = getattr(context, "tool_context", None)
        tool_use_id = f"f53-{self._tool_name}-{id(parsed) & 0xFFFF:04x}"

        from clawcodex_ext.tool_system.protocol import ToolCall

        call = ToolCall(name=self._tool_name, input=parsed, tool_use_id=tool_use_id)

        try:
            if registry is None:
                return _text_result(
                    self._tool_name,
                    "No active tool registry in this context — cannot dispatch.",
                )
            result = registry.dispatch(call, tool_context)
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %r raised during F-53 dispatch", self._tool_name)
            return _text_result(self._tool_name, _format_error(self._tool_name, str(exc)))

        if getattr(result, "is_error", False):
            err = result.output if isinstance(result.output, dict) else {"error": str(result.output)}
            msg = err.get("error") if isinstance(err, dict) else str(err)
            return _text_result(self._tool_name, _format_error(self._tool_name, msg))

        return _text_result(self._tool_name, _format_output(result.output))

    # --- CLI argv entry ------------------------------------------------

    def invoke_from_argv(
        self,
        argv: list[str],
        *,
        tool_registry: "ToolRegistry | None" = None,
        tool_context: Any = None,
    ) -> int:
        """Invoke from a CLI argv list (``clawcodex-dev tool <name> args``).

        Returns the exit code (0 success, 1 tool error, 2 usage error).
        """
        tool = self._tool_resolver(self._tool_name, tool_registry)
        if tool is None:
            print(f"error: tool '{self._tool_name}' is not registered", flush=True)
            return 1
        if core_filter.is_core_tool(tool):
            print(
                f"error: tool '{self._tool_name}' is a built-in core tool and is not exposed as a CLI subcommand",
                flush=True,
            )
            return 2

        schema = tool.input_schema if isinstance(tool.input_schema, Mapping) else self._schema
        try:
            parsed = schema_parser.parse_tool_args(self._tool_name, schema, argv)
        except argparse.ArgumentError as exc:
            print(self._parser.format_help(), flush=True)
            print(f"\nerror: {exc.message or exc}", flush=True)
            return 2

        extras = parsed.pop("_extra", None) if isinstance(parsed, dict) else None
        if extras:
            parsed["_unknown_args"] = list(extras)

        from clawcodex_ext.tool_system.protocol import ToolCall

        tool_use_id = f"f53-cli-{self._tool_name}-{id(parsed) & 0xFFFF:04x}"
        call = ToolCall(name=self._tool_name, input=parsed, tool_use_id=tool_use_id)

        if tool_registry is None:
            print("error: no tool registry available", flush=True)
            return 1

        try:
            result = tool_registry.dispatch(call, tool_context)
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %r raised during F-53 CLI dispatch", self._tool_name)
            print(_format_error(self._tool_name, str(exc)), flush=True)
            return 1

        if getattr(result, "is_error", False):
            err = result.output if isinstance(result.output, dict) else {"error": str(result.output)}
            msg = err.get("error") if isinstance(err, dict) else str(err)
            print(_format_error(self._tool_name, msg), flush=True)
            return 1

        output = _format_output(result.output)
        if output:
            print(output, flush=True)
        return 0

    # --- Tokenization --------------------------------------------------

    @staticmethod
    def _tokenize(args: str) -> list[str]:
        """Tokenize a slash-command args string into argv.

        Handles simple shell-like splitting with single/double quotes.
        Empty string → empty list.
        """
        if not args:
            return []
        # Use shlex for robust quoting support.
        import shlex

        try:
            return shlex.split(args)
        except ValueError:
            # Unbalanced quotes — fall back to whitespace split.
            return args.split()


def _text_result(tool_name: str, text: str) -> LocalCommandResult:
    """Wrap *text* as a system-displayed command result."""
    return LocalCommandResult(type="text", value=text, display_text=text)


def _default_tool_resolver(name: str, registry: "ToolRegistry | None") -> "Tool | None":
    """Resolve a tool by name from *registry*, falling back to a default.

    The default registry is built lazily; this is the *only* path that
    imports the heavy tool graph, so REPL/TUI startup stays cheap until
    the first ``/tool-name`` invocation.
    """
    if registry is not None:
        try:
            tool = registry.get(name)
            if tool is not None:
                return tool
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    try:
        from clawcodex_ext.tool_system.defaults import build_default_registry

        default = build_default_registry()
        return default.get(name)
    except Exception:  # noqa: BLE001
        return None
