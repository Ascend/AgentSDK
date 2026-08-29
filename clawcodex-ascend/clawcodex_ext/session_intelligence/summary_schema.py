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

"""Schema helpers for session summary sidecars."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionSummary:
    session_id: str
    cwd: str = ""
    updated_at: float = field(default_factory=time.time)
    transcript_mtime: float = 0.0
    title: str = ""
    goals: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    commands_seen: list[str] = field(default_factory=list)
    user_preferences: list[str] = field(default_factory=list)
    next_action_candidates: list[str] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "updated_at": self.updated_at,
            "transcript_mtime": self.transcript_mtime,
            "title": self.title,
            "goals": self.goals,
            "completed": self.completed,
            "open_threads": self.open_threads,
            "files_touched": self.files_touched,
            "commands_seen": self.commands_seen,
            "user_preferences": self.user_preferences,
            "next_action_candidates": self.next_action_candidates,
        }
