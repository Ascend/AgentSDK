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

"""Write tool activity — shows target file and operation kind."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from .base import ToolActivity


class WriteActivity(ToolActivity):
    def inflight_text(self) -> Text:
        path = self.tool_input.get("file_path") or self.tool_input.get("filePath") or ""
        return Text(f"write {path}" if path else "write …", style="dim")

    def result_body(self, output: Any, *, is_error: bool) -> Any | None:
        if not isinstance(output, dict):
            return None
        path = output.get("filePath") or output.get("file_path") or ""
        op = output.get("type") or ""
        summary = f"{op} · {path}".strip(" ·")
        if not summary:
            return None
        return Text(summary, style="red" if is_error else "green")
