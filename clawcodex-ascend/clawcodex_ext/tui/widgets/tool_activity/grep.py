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

"""Grep tool activity."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from .base import ToolActivity, truncated_panel


class GrepActivity(ToolActivity):
    def inflight_text(self) -> Text:
        pattern = self.tool_input.get("pattern") or ""
        path = self.tool_input.get("path") or ""
        tail = f" in {path}" if path else ""
        return Text(f"grep /{pattern}/{tail}" if pattern else "grep …", style="dim")

    def result_body(self, output: Any, *, is_error: bool) -> Any | None:
        if not isinstance(output, dict):
            return None
        content = output.get("content")
        if isinstance(content, str) and content.strip():
            return truncated_panel(content, style="red" if is_error else "green")
        n = output.get("numFiles")
        mode = output.get("mode")
        if n is not None:
            return Text(
                f"mode={mode} · files={n}",
                style="red" if is_error else "green",
            )
        return None
