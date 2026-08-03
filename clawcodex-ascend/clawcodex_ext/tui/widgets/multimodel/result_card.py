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

from __future__ import annotations

from textual.widgets import Static


class ModelResultCard(Static):
    """A result card which can be rendered collapsed or expanded."""

    def set_result(self, state: object, *, selected: bool = False) -> None:
        duration = getattr(state, "duration_ms", None)
        seconds = "?" if duration is None else f"{duration / 1000:.1f}s"
        tokens = getattr(state, "tokens", {}).get("output", 0)
        slot, content = getattr(state, "slot"), getattr(state, "content")
        expanded = getattr(state, "expanded", False)
        prefix = "❯ " if selected else "  "
        text = f"{prefix}{slot} ({seconds}, {tokens} tok)\n"
        text += content if expanded else (content.splitlines()[0] if content else "等待输出…")
        self.update(text)
