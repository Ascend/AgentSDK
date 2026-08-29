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

"""Shared command-allowlist configuration for the IM gateway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_REPL_COMMAND_ALLOWLIST: tuple[str, ...] = (
    "/stop",
    "/clear",
    "/reset",
    "/new",
    "/goal",
    "/help",
    "/?",
    "/cost",
    "/history",
    "/context",
    "/recap",
    "/btw",
    "/cron-list",
    "/cron-status",
    "/cron-runs",
    "/tools",
    "/skills",
    "/diff",
    "/mcp",
    "/tasks",
    "/idle",
    "/doctor",
    "/release-notes",
)

DEFAULT_ORCHESTRATOR_COMMAND_ALLOWLIST: tuple[str, ...] = (
    "/server status",
    "/issue list",
    "/issue show",
    "/issue tail",
    "/issue stop",
    "/issue pause",
    "/issue resume",
    "/issue clarify",
    "/issue inject",
    "/issue feedback",
    "/issue review",
    "/issue retry",
    "/issue workspace",
    "/issue rebase",
)


def _normalize_command_allowlist(
    commands: Any,
    *,
    path: str,
    max_parts: int,
) -> tuple[str, ...]:
    if not isinstance(commands, (list, tuple)):
        raise ValueError(f"{path}: expected a YAML list of slash commands")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in commands:
        if not isinstance(item, str):
            raise ValueError(f"{path}: every command must be a string")
        command = " ".join(item.strip().lower().split())
        parts = command.split()
        if not command.startswith("/") or len(parts) > max_parts:
            raise ValueError(f"{path}: invalid command {item!r}; expected at most {max_parts} slash token(s)")
        if command not in seen:
            seen.add(command)
            normalized.append(command)
    return tuple(normalized)


@dataclass(frozen=True)
class CommandAllowlistConfig:
    """Runtime slash-command allowlists persisted in ``channels.yaml``."""

    repl: tuple[str, ...] = DEFAULT_REPL_COMMAND_ALLOWLIST
    orchestrator: tuple[str, ...] = DEFAULT_ORCHESTRATOR_COMMAND_ALLOWLIST

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repl",
            _normalize_command_allowlist(
                self.repl,
                path="command_allowlists.repl",
                max_parts=1,
            ),
        )
        object.__setattr__(
            self,
            "orchestrator",
            _normalize_command_allowlist(
                self.orchestrator,
                path="command_allowlists.orchestrator",
                max_parts=2,
            ),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "repl": list(self.repl),
            "orchestrator": list(self.orchestrator),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CommandAllowlistConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("command_allowlists: expected a YAML mapping")
        return cls(
            repl=data["repl"] if "repl" in data else DEFAULT_REPL_COMMAND_ALLOWLIST,
            orchestrator=(data["orchestrator"] if "orchestrator" in data else DEFAULT_ORCHESTRATOR_COMMAND_ALLOWLIST),
        )


__all__ = [
    "CommandAllowlistConfig",
    "DEFAULT_ORCHESTRATOR_COMMAND_ALLOWLIST",
    "DEFAULT_REPL_COMMAND_ALLOWLIST",
]
