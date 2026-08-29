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

"""Bash tool activity — parity with ``ShellProgress`` / ``BashProgress``.

Inflight: shows the command being executed (truncated) so the user can
tell what's running.  Completion: renders stdout (or stderr on failure)
in a bordered preview panel with the usual truncation limits.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text

from .base import ToolActivity, truncated_panel


class BashActivity(ToolActivity):
    def inflight_text(self) -> Text:
        cmd = (self.tool_input.get("command") or "").strip()
        if len(cmd) > 120:
            cmd = cmd[:117] + "…"
        return Text(f"$ {cmd}" if cmd else "$ …", style="dim")

    def result_body(self, output: Any, *, is_error: bool) -> Any | None:
        if not isinstance(output, dict):
            return None
        stdout = output.get("stdout") or ""
        stderr = output.get("stderr") or ""
        body = stdout or stderr
        if not body or not body.strip():
            return None
        style = "red" if is_error else "green"
        return truncated_panel(body, style=style)
