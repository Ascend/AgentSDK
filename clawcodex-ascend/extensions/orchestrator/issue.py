#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Normalized issue representation used by the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Issue:
    """Normalized issue used by the orchestrator."""

    id: str | None = None
    identifier: str | None = None
    title: str | None = None
    description: str | None = None
    priority: int | None = None
    state: str | None = None
    branch_name: str | None = None
    url: str | None = None
    assignee_id: str | None = None
    author_login: str | None = None
    blocked_by: list[dict[str, Any]] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    assigned_to_worker: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Per-issue python interpreter override (cascade level 1,
    # highest priority). When non-empty, this value overrides every
    # other source — workspace ``python_executable``,
    # ``_detect_python_in_workspace``, and ``agent.python_executable``.
    # Empty string (the default) means "fall through to the next
    # level of the cascade." Populated by ``LocalTrackerAdapter`` from
    # the issue markdown frontmatter (``python_executable: ...``);
    # remote trackers (GitHub/Gitee/GitCode/Linear) leave it empty
    # and rely on workspace-level or agent-level configuration.
    python_executable: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert the issue to a plain dict for template rendering."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def label_names(self) -> list[str]:
        """Return a copy of the issue's label names."""
        return list(self.labels)
