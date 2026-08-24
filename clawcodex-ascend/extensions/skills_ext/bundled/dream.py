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
#
"""Dream slash skill.

The ``/dream`` command exposes the auto-dream service to REPL/headless
surfaces. Mirrors the upstream ``/dream`` bundled skill
(``claude-code-best/src/skills/bundled/dream.ts``).

Subcommands:

* ``/dream`` (no args)    — show usage.
* ``/dream run``          — force a dream consolidation pass now
  (bypasses all gates; stamps the lock optimistically).
* ``/dream once``         — alias for ``run``.
* ``/dream status``       — show in-flight dream tasks on the shared
  :class:`RuntimeTaskRegistry`.
* ``/dream help``         — show usage.

The implementation is intentionally a :class:`LocalCommand` (not a
:class:`PromptCommand`): the user is asking the **local daemon** to do
work, not the model. Wraps :func:`clawcodex_ext.dreaming.manual_dream`
for ``run``/``once`` and reads the
:class:`clawcodex_ext.dreaming.service.RuntimeTaskRegistry` for
``status``.

Wired via :func:`register_dream_skill`, called from
:func:`extensions.skills_ext.init_skills_ext` so the command lands
in the global command registry during skills_ext initialization.
"""

from __future__ import annotations

import logging
from typing import Any

from clawcodex_ext.command_system.types import (
    LocalCommand,
    LocalCommandResult,
)

_log = logging.getLogger(__name__)


def _build_dream_command() -> LocalCommand:
    command = LocalCommand(
        name="dream",
        description=(
            "Force a memory consolidation pass (auto-dream) or show "
            "in-flight dream tasks. Subcommands: run, once, status, help."
        ),
        argument_hint="[run|once|status|help]",
        loaded_from="bundled",
    )
    command.set_call(_dream_call)
    return command


def get_dream_command() -> LocalCommand:
    """Return the extension-owned command for common builtin registration."""

    return DREAM_COMMAND


def register_dream_skill(registry: Any | None = None) -> None:
    """Register /dream in the supplied or global command registry."""

    from clawcodex_ext.command_system.registry import get_command_registry

    (registry or get_command_registry()).register(DREAM_COMMAND)


def _dream_call(args: str, context: Any) -> LocalCommandResult:
    """Dispatch ``/dream [subcommand]`` to the right handler."""
    tokens = args.split()
    subcommand = tokens[0].lower() if tokens else "help"

    if subcommand in ("run", "once"):
        return _dream_run()
    if subcommand == "status":
        return _dream_status()
    # default + help + unknown → usage
    return _dream_help(subcommand)


def _dream_run() -> LocalCommandResult:
    """Force a dream consolidation pass via :func:`manual_dream`."""
    from clawcodex_ext.dreaming import manual_dream

    try:
        manual_dream()
    except Exception as e:
        _log.warning("/dream run: manual_dream raised: %s", e)
        return LocalCommandResult(
            type="text",
            value=f"Error: failed to trigger dream run: {e}",
        )
    return LocalCommandResult(
        type="text",
        value=(
            "Dream consolidation triggered. The task runs in the background; use `/dream status` to check progress."
        ),
    )


def _dream_status() -> LocalCommandResult:
    """List in-flight dream tasks on the shared task registry.

    Reads from the registry the dream service was initialized with via
    :func:`clawcodex_ext.dreaming.get_active_registry`, so ``/dream
    status`` reports tasks fired by :func:`manual_dream` (and the
    auto-dream loop) without requiring the caller to thread the
    registry through. Falls back to a fresh registry when the service
    has not been initialized yet.
    """
    try:
        from clawcodex_ext.dreaming import get_active_registry

        registry = get_active_registry()
    except Exception as e:
        return LocalCommandResult(
            type="text",
            value=f"Error: cannot access task registry: {e}",
        )

    try:
        tasks = registry.by_type("dream")
    except Exception as e:
        return LocalCommandResult(
            type="text",
            value=f"Error: failed to query dream tasks: {e}",
        )

    if not tasks:
        return LocalCommandResult(
            type="text",
            value=("No dream tasks in flight. (Auto-dream has not fired recently, or all tasks have completed.)"),
        )

    lines = [f"In-flight dream tasks ({len(tasks)}):", ""]
    for t in tasks:
        try:
            tid = getattr(t, "id", "?")
            status = getattr(t, "status", "?")
            phase = getattr(t, "phase", "?")
            sessions = getattr(t, "sessions_reviewing", 0)
            turns = len(getattr(t, "turns", []) or [])
            lines.append(f"- {tid}  status={status}  phase={phase}  sessions={sessions}  turns={turns}")
        except Exception:
            lines.append(f"- (malformed task: {t!r})")
    return LocalCommandResult(type="text", value="\n".join(lines))


def _dream_help(unknown: str | None = None) -> LocalCommandResult:
    """Show usage. If an unknown subcommand was passed, mention it."""
    body = [
        "Usage: /dream [run|once|status|help]",
        "",
        "Subcommands:",
        "  run      Force a memory consolidation pass now (bypasses gates).",
        "  once     Alias for `run`.",
        "  status   List in-flight dream tasks.",
        "  help     Show this message.",
        "",
        "Without arguments, shows this help.",
    ]
    if unknown and unknown not in ("help", ""):
        body.insert(0, f"Unknown subcommand: {unknown!r}")
        body.insert(1, "")
    return LocalCommandResult(type="text", value="\n".join(body))


DREAM_COMMAND = _build_dream_command()


__all__ = ["DREAM_COMMAND", "get_dream_command", "register_dream_skill"]
