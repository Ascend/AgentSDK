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
"""context_providers — Layer 2 reference context providers.

Three reference implementations demonstrating the ``register_section`` API
for injecting dynamic context into the system prompt:

* ``from_issue`` — Issue-tracker context (order=55, tags: workflow/issue-tracker)
* ``from_ci`` — CI pipeline status (order=56, tags: ci)
* ``from_config`` — Declarative YAML-snippet injection (order=57, tags: config)

Importing any of these modules triggers `register_section` at module-load
time — there is no separate "install" step.
"""

from __future__ import annotations

__all__ = [
    "from_ci",
    "from_config",
    "from_issue",
]
