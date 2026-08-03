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

"""Glob tool activity — shows up to 10 matched paths."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from .base import ToolActivity, truncated_panel


class GlobActivity(ToolActivity):
    def inflight_text(self) -> Text:
        pattern = self.tool_input.get("pattern") or ""
        return Text(f"glob {pattern}" if pattern else "glob …", style="dim")

    def result_body(self, output: Any, *, is_error: bool) -> Any | None:
        if not isinstance(output, dict):
            return None
        files = output.get("files") or output.get("matches")
        if isinstance(files, list) and files:
            preview = "\n".join(str(p) for p in files[:10])
            if len(files) > 10:
                preview += f"\n… +{len(files) - 10} more"
            return truncated_panel(preview, style="red" if is_error else "green")
        n = output.get("numFiles")
        if n is not None:
            return Text(
                f"matches={n}",
                style="red" if is_error else "green",
            )
        return None
