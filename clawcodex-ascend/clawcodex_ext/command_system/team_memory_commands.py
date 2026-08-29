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

"""TeamMem — ``/team memory`` debug command family (P93-F).

Human-facing CLI for inspecting and mutating team shared memory. The
agent uses the :class:`TeamMemoryTool`; this command is for developers
debugging the store live.

Subcommands::

 /team memory status
 /team memory recall "deployment checklist"
 /team memory remember --tag build "Run stability gate before commit."
 /team memory list --tag review
 /team memory compact
 /team memory delete <entry_id> [reason]

The command is a :class:`LocalCommand` (no UI surface needed — all
output is plain text). It is gated by ``is_team_memory_enabled`` via
``is_enabled`` so it disappears from ``/help`` when team memory is off.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from clawcodex_ext.memdir.team_mem_paths import is_team_memory_enabled  # pylint: disable=no-name-in-module

from .types import CommandContext, LocalCommand, LocalCommandResult

__all__ = ["TEAM_MEMORY_COMMAND", "TeamMemoryCommand"]

_VALID_SCOPES = frozenset({"team", "lead_only", "agent_pair"})
_VALID_SOURCES = frozenset({"manual", "send_message", "task_result", "review", "system"})


def _resolve_agent_id(context: CommandContext) -> str:
    """Best-effort agent id from the command context.

    Commands run as the team lead by default (the user driving the
    REPL is the lead). Falls back to ``"team-lead"`` so the policy
    treats the operator as a member when the team file lists them as
    lead, and rejects otherwise (fail-closed).
    """
    tc = getattr(context, "tool_context", None)
    if tc is not None:
        agent_id = getattr(tc, "agent_id", None)
        if isinstance(agent_id, str) and agent_id:
            return agent_id
        team = getattr(tc, "team", None)
        if isinstance(team, dict):
            lead = team.get("lead_agent_id")
            if isinstance(lead, str) and lead:
                return lead
    return "team-lead"


def _workspace_root(context: CommandContext) -> Path:
    tc = getattr(context, "tool_context", None)
    if tc is not None and getattr(tc, "workspace_root", None) is not None:
        return Path(tc.workspace_root)
    return Path(context.workspace_root)


def _build_service(context: CommandContext):
    """Lazily build a :class:`TeamMemoryService`. Returns ``(service, err)``."""
    from extensions.agents.team_memory import (
        TeamMemoryConfig,
        TeamMemoryService,
        TeamNotFoundError,
    )

    if not is_team_memory_enabled():
        return None, "team memory disabled (set CLAUDE_CODE_TEAM_MEMORY=1 and enable auto-memory)"
    try:
        service = TeamMemoryService(
            workspace_root=_workspace_root(context),
            config=TeamMemoryConfig(enabled=True),
        )
    except TeamNotFoundError:
        return None, "no active team — run TeamCreate first"
    return service, None


def _cmd_status(service, _args: list[str]) -> str:
    total = len(service.store.list_entries(include_expired=True))
    live = len(service.store.list_entries(include_expired=False))
    return (
        f"Team memory: team={service.team_id!r}\n"
        f"  total entries (incl. tombstoned/expired): {total}\n"
        f"  live entries: {live}\n"
        f"  enabled: {service.config.enabled}\n"
        f"  max_entry_bytes: {service.config.max_entry_bytes}\n"
        f"  prompt_top_k: {service.config.prompt_top_k}\n"
    )


def _cmd_recall(service, args: list[str]) -> str:
    from extensions.agents.team_memory import TeamMemoryQuery

    if not args:
        return 'Usage: /team memory recall "<query>"'
    query = " ".join(args).strip()
    # Strip surrounding quotes if present.
    if query.startswith('"') and query.endswith('"') and len(query) >= 2:
        query = query[1:-1]
    results = service.recall(
        TeamMemoryQuery(
            team_id=service.team_id,
            query=query,
            requester_agent_id="team-lead",
            top_k=service.config.query_top_k,
        )
    )
    if not results:
        return f"No team memory matches for {query!r}."
    lines = [f"Recalled {len(results)} entries for {query!r}:"]
    for i, r in enumerate(results, 1):
        tag_str = (" [" + ", ".join(r.entry.tags) + "]") if r.entry.tags else ""
        lines.append(
            f"{i}. {r.entry.summary}{tag_str} (score={r.score:.3f}, source={r.entry.source}, scope={r.entry.scope})"
        )
        lines.append(f"   id={r.entry.id} author={r.entry.author_agent_id} at={r.entry.created_at}")
    return "\n".join(lines) + "\n"


def _cmd_remember(service, args: list[str]) -> str:
    from extensions.agents.team_memory import (
        TeamMemoryDisabledError,
        TeamMemoryPermissionError,
        TeamMemoryTooLargeError,
    )

    # Parse: optional --tag X (repeatable) --scope S --source M --summary "..."
    # then the positional content (possibly quoted).
    tags: list[str] = []
    scope = "team"
    source = "manual"
    summary: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tag" and i + 1 < len(args):
            tags.append(args[i + 1])
            i += 2
            continue
        if a == "--scope" and i + 1 < len(args):
            scope = args[i + 1]
            i += 2
            continue
        if a == "--source" and i + 1 < len(args):
            source = args[i + 1]
            i += 2
            continue
        if a == "--summary" and i + 1 < len(args):
            summary = args[i + 1]
            i += 2
            continue
        positional.append(a)
        i += 1
    if not positional:
        return (
            "Usage: /team memory remember [--tag X]... [--scope team|lead_only|agent_pair] "
            '[--source manual|task_result|review|send_message|system] [--summary S] "<content>"'
        )
    if scope not in _VALID_SCOPES:
        return f"Invalid --scope {scope!r}; expected one of {sorted(_VALID_SCOPES)}"
    if source not in _VALID_SOURCES:
        return f"Invalid --source {source!r}; expected one of {sorted(_VALID_SOURCES)}"
    content = " ".join(positional).strip()
    if content.startswith('"') and content.endswith('"') and len(content) >= 2:
        content = content[1:-1]
    if not content:
        return "Content must not be empty."
    try:
        entry = service.remember(
            content,
            author_agent_id="team-lead",
            tags=tags,
            source=source,
            scope=scope,  # type: ignore[arg-type]
            summary=summary,
        )
    except TeamMemoryPermissionError as exc:
        return f"Permission denied: {exc}"
    except TeamMemoryTooLargeError as exc:
        return f"Entry too large: {exc}"
    except TeamMemoryDisabledError as exc:
        return f"Disabled: {exc}"
    return f"Remembered entry {entry.id}: {entry.summary}\n"


def _cmd_list(service, args: list[str]) -> str:
    tags: list[str] = []
    sources: list[str] = []
    limit = 50
    i = 0
    positional: list[str] = []
    while i < len(args):
        a = args[i]
        if a == "--tag" and i + 1 < len(args):
            tags.append(args[i + 1])
            i += 2
            continue
        if a == "--source" and i + 1 < len(args):
            sources.append(args[i + 1])
            i += 2
            continue
        if a == "--limit" and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                return f"Invalid --limit {args[i + 1]!r}"
            i += 2
            continue
        positional.append(a)
        i += 1
    entries = service.list_entries(
        requester_agent_id="team-lead",
        limit=limit,
        tags=tags,
        sources=sources,
    )
    if not entries:
        return "No team memory entries match."
    lines = [f"{len(entries)} entries (newest first):"]
    for e in entries:
        tag_str = (" [" + ", ".join(e.tags) + "]") if e.tags else ""
        lines.append(f"- {e.summary}{tag_str} (source={e.source}, scope={e.scope}, author={e.author_agent_id})")
        lines.append(f"  id={e.id} at={e.created_at}")
    return "\n".join(lines) + "\n"


def _cmd_delete(service, args: list[str]) -> str:
    from extensions.agents.team_memory import TeamMemoryPermissionError

    if not args:
        return "Usage: /team memory delete <entry_id> [reason...]"
    entry_id = args[0]
    reason = " ".join(args[1:]) or "manual delete via /team memory"
    try:
        removed = service.delete(entry_id, actor="team-lead", reason=reason)
    except TeamMemoryPermissionError as exc:
        return f"Permission denied: {exc}"
    if not removed:
        return f"Entry {entry_id!r} not found or already deleted."
    return f"Deleted entry {entry_id}.\n"


def _cmd_compact(service, _args: list[str]) -> str:
    from extensions.agents.team_memory import TeamMemoryPermissionError

    try:
        entry = service.compact(actor="team-lead")
    except TeamMemoryPermissionError as exc:
        return f"Permission denied: {exc}"
    return f"Compacted team memory into summary entry {entry.id}: {entry.summary}\n"


_SUBCOMMANDS = {
    "status": _cmd_status,
    "recall": _cmd_recall,
    "remember": _cmd_remember,
    "list": _cmd_list,
    "delete": _cmd_delete,
    "compact": _cmd_compact,
}


def _team_memory_run(args: str, context: CommandContext) -> LocalCommandResult:
    """Dispatch ``/team memory <sub> [args...]``."""
    raw = (args or "").strip()
    # ``/team memory`` with no subcommand → status.
    if not raw:
        sub, sub_args = "status", []
    else:
        try:
            parts = shlex.split(raw)
        except ValueError:
            # Mismatched quotes — fall back to a plain split.
            parts = raw.split()
        if not parts:
            sub, sub_args = "status", []
        else:
            sub, sub_args = parts[0], parts[1:]
    handler = _SUBCOMMANDS.get(sub)
    if handler is None:
        return LocalCommandResult(
            type="text",
            value=(f"Unknown /team memory subcommand {sub!r}. Valid: {sorted(_SUBCOMMANDS)}"),
        )
    service, err = _build_service(context)
    if service is None:
        return LocalCommandResult(type="text", value=f"Team memory unavailable: {err}")
    try:
        text = handler(service, sub_args)
    except Exception as exc:  # pragma: no cover — defensive in CLI
        text = f"Team memory error: {exc}"
    return LocalCommandResult(type="text", value=text)


@dataclass(frozen=True)
class TeamMemoryCommand(LocalCommand):
    """``/team memory`` — debug CLI for team shared memory."""

    async def call(self, args: str, context: CommandContext) -> LocalCommandResult:  # type: ignore[override]
        return _team_memory_run(args, context)


TEAM_MEMORY_COMMAND = TeamMemoryCommand(
    name="team memory",
    description="Inspect and mutate the team's shared memory.",
    argument_hint="[status|recall|remember|list|delete|compact] ...",
    supports_non_interactive=True,
    is_enabled=is_team_memory_enabled,
)


# Wire the call implementation into the frozen dataclass. ``set_call``
# uses ``object.__setattr__`` to bypass the frozen guard (see
# LocalCommand.set_call).
TEAM_MEMORY_COMMAND.set_call(_team_memory_run)
