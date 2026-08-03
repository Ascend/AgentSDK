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
# pylint: disable=E0611

from textual.widgets import Static

from clawcodex_ext.multimodel.display import MultiModelBridge


class MultiModelLivePanel(Static):
    """Mounted transcript panel for live tabs and the completed summary."""

    def __init__(self, slots: list[str]) -> None:
        super().__init__()
        self.bridge = MultiModelBridge(slots)
        self._render()

    def progress(self, slot: str, chunk: str) -> None:
        self.bridge.on_progress(slot, chunk)
        self._render()

    def complete(self, result) -> None:
        self.bridge.on_complete(result)
        self._render()

    def _render(self) -> None:
        states = self.bridge.display.results
        if self.bridge.display.phase.value == "streaming":
            selected = self.bridge.display.selected
            body = selected.content if selected else ""
            tabs = " │ ".join(("◄ " if item is selected else "") + item.slot for item in states)
            progress = "\n".join(
                f"{'●' if item is selected else '○'} {item.slot} {item.progress_percent}%" for item in states
            )
            self.update(f"── 多模型并行输出 ──\n{tabs}\n\n{body}\n\n{progress}")
        else:
            self.update("✅ 全部完成\n\n" + self.bridge.render_text())
