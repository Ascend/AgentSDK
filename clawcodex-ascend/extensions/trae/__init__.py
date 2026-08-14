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
#
"""Trae IDE integration (Layer 2).

Two complementary sub-features:
  * ``mcp_bridge`` — MCP reverse bridge: Trae IDE calls clawcodex via MCP
  * ``acp_cli_adapter`` — wraps ByteDance's open-source trae-cli as a pseudo ACP server

The two mirror each other: the trae-cli process started by the adapter
can mount the MCP server exposed by the bridge, forming a two-way loop.
See ``docs/feature_plan/06-ccb-benchmark/f-66-acp-protocol.md``.

Fully decoupled in Layer 2 — deleting this directory rolls back cleanly
without affecting ``src/`` or ``clawcodex_ext/``.
"""

from __future__ import annotations

__all__ = ["mcp_bridge", "acp_cli_adapter"]
