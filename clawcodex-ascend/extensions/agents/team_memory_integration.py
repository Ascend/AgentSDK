#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""TeamMem — lifecycle & prompt integration.

Thin integration layer that wires :class:`TeamMemoryService` into the
Team / Coordinator / Agent runtime without touching ``src/``. The hooks
here are called from the existing TeamCreate / TeamDelete / SendMessage
paths (or their downstream wrappers in ``clawcodex_ext/``).

Three integration points (§1.6 / §1.8 / §4):

  * **TeamCreate** → :func:`initialize_team_memory` ensures the
    ``<auto_mem>/team/`` dir exists and writes an empty ``MEMORY.md``.
  * **TeamDelete** → :func:`archive_team_memory` snapshots the store
    into ``archive/<ts>.jsonl`` *before* the team file is removed
    (default: archive, never delete — §0.2 / acceptance #9).
  * **SendMessage** → :func:`sink_send_message_summary` records a
    compact summary of broadcast / key peer messages as a
    ``source=send_message`` entry.

The prompt-injection path is :func:`build_team_memory_prompt_section`,
called by the context builder to assemble the ``<team_memory>`` block.
It delegates to :meth:`TeamMemoryService.build_prompt_section` and
returns ``""`` when disabled (the builder drops empty sections).

All hooks are **no-ops when team memory is disabled** (env
``CLAUDE_CODE_TEAM_MEMORY`` off or auto-memory off) — they neither
raise nor write files (§1.10 ``TeamMemoryDisabledError`` is for
explicit tool calls; background hooks fail silent).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from clawcodex_ext.memdir.team_mem_paths import (  # pylint: disable=import-error
    PathTraversalError,
    get_team_mem_path,
    is_team_memory_enabled,
)
from clawcodex_ext.services.swarm.team_file import (  # pylint: disable=import-error
    read_team_file,
)

from .team_memory import (
    TeamMemoryConfig,
    TeamMemoryDisabledError,
    TeamMemoryEntry,
    TeamMemoryService,
    TeamNotFoundError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "initialize_team_memory",
    "archive_team_memory",
    "sink_send_message_summary",
    "build_team_memory_prompt_section",
    "get_team_memory_service",
    "is_team_memory_active",
]


def is_team_memory_active(config: TeamMemoryConfig | None = None) -> bool:
    """True iff team memory is enabled both by config and by env/auto-mem."""
    cfg_enabled = config.enabled if config is not None else True
    return cfg_enabled and is_team_memory_enabled()


def get_team_memory_service(
    workspace_root: Path,
    *,
    config: TeamMemoryConfig | None = None,
) -> TeamMemoryService | None:
    """Build a :class:`TeamMemoryService` for ``workspace_root``.

    Returns ``None`` when team memory is disabled or no team file exists
    — callers (prompt builder, hooks) treat ``None`` as "skip silently".
    Explicit tool/command callers that want the disabled error should
    construct the service directly so the error surfaces.
    """
    if not is_team_memory_active(config):
        return None
    # When the caller did not supply a config, build one with
    # ``enabled=True`` — we already verified the env/auto-memory gate
    # above, so the service's internal ``self._config.enabled`` checks
    # (remember / recall / record_message_summary) should pass. The
    # default ``TeamMemoryConfig.enabled=False`` is the *opt-out* state
    # for explicit construction; integration callers want opt-in.
    if config is None:
        config = TeamMemoryConfig(enabled=True)
    try:
        return TeamMemoryService(workspace_root=workspace_root, config=config)
    except TeamNotFoundError:
        return None


# -- TeamCreate hook (P93-E) -----------------------------------------------


def initialize_team_memory(
    workspace_root: Path,
    *,
    config: TeamMemoryConfig | None = None,
) -> bool:
    """Ensure the team-memory dir and ``MEMORY.md`` exist after TeamCreate.

    Returns ``True`` if initialized, ``False`` if disabled or no team
    file. Idempotent — safe to call on every TeamCreate even when the
    dir already exists (acceptance #2).
    """
    if not is_team_memory_active(config):
        return False
    team_file = read_team_file(workspace_root)
    if team_file is None:
        return False
    try:
        team_dir = Path(get_team_mem_path())
    except PathTraversalError as exc:
        logger.warning("team_memory init refused: %s", exc)
        return False
    team_dir.mkdir(parents=True, exist_ok=True)
    entrypoint = team_dir / "MEMORY.md"
    if not entrypoint.exists():
        team_id = team_file.team_name or workspace_root.name
        entrypoint.write_text(
            f"# Team Memory\n\nTeam: `{team_id}` — 0 live entries.\n",
            encoding="utf-8",
        )
    return True


# -- TeamDelete hook (P93-E) -----------------------------------------------


def archive_team_memory(
    workspace_root: Path,
    *,
    config: TeamMemoryConfig | None = None,
    reason: str = "TeamDelete",
) -> Path | None:
    """Snapshot team memory before TeamDelete removes the team file.

    Default behavior is **archive, not delete** (§0.2 / acceptance
    #9). Returns the archive path, or ``None`` when disabled or no
    team file. The caller (TeamDelete tool wrapper) is responsible for
    the actual team-file removal — this hook only snapshots.
    """
    service = get_team_memory_service(workspace_root, config=config)
    if service is None:
        return None
    try:
        return service.archive(reason=reason)
    except (OSError, PathTraversalError) as exc:
        logger.warning("team_memory archive failed: %s", exc)
        return None


# -- SendMessage hook (P93-E) ----------------------------------------------


def sink_send_message_summary(
    workspace_root: Path,
    *,
    sender: str,
    sender_agent_id: str | None,
    recipients: Iterable[str],
    summary: str,
    message: str,
    config: TeamMemoryConfig | None = None,
) -> TeamMemoryEntry | None:
    """Record a SendMessage exchange into team memory.

    Called by the SendMessage dispatch path after a successful delivery
    (broadcast or named recipient). No-op when disabled. Returns the
    created entry or ``None`` (rejected / disabled / empty recipients).
    """
    service = get_team_memory_service(workspace_root, config=config)
    if service is None:
        return None
    try:
        return service.record_message_summary(
            sender=sender,
            sender_agent_id=sender_agent_id,
            recipients=recipients,
            summary=summary,
            message=message,
        )
    except (OSError, PathTraversalError) as exc:
        logger.warning("team_memory message summary sink failed: %s", exc)
        return None


# -- Prompt injection (P93-G) ----------------------------------------------


def build_team_memory_prompt_section(
    workspace_root: Path,
    *,
    requester_agent_id: str,
    task: str,
    config: TeamMemoryConfig | None = None,
) -> str:
    """Assemble the ``<team_memory>`` prompt section for a teammate.

    Empty string when disabled, no team file, or no recall hits. The
    context builder is expected to skip empty sections (§1.8 /
    acceptance #1).
    """
    service = get_team_memory_service(workspace_root, config=config)
    if service is None:
        return ""
    try:
        return service.build_prompt_section(
            requester_agent_id=requester_agent_id,
            task=task,
        )
    except (TeamMemoryDisabledError, TeamNotFoundError, PathTraversalError) as exc:
        logger.debug("team_memory prompt section skipped: %s", exc)
        return ""
    except Exception as exc:  # pragma: no cover — defensive in prompt path
        logger.warning("team_memory prompt section failed: %s", exc)
        return ""
