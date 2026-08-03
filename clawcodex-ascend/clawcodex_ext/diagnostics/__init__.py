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

"""Diagnostics primitives (F-108 §十八).

This package houses the freeze-detection watchdog (Layer 1) and its
dump / resolution helpers. Nothing in here modifies behaviour of the
canonical query / agent loop — every consumer is opt-in via
``CLAWCODEX_FREEZE_DIAG=1`` (real-time enable) or an explicit ``start()``
call.

Layer 0 quick fixes (P108-A Permission/AskUser auto-deny, P108-B
headless future budget, P108-C tool timeout) live with their owning
modules (``clawcodex_ext/tui/agent_bridge.py``,
``extensions/api/query.py``); Layer 2 hard timeouts and Layer 3
auto-recovery live in ``extensions/api/query.py`` and
``clawcodex_ext/query/agent_loop_compat.py``. Layer 4 diagnostics CLI
(:mod:`clawcodex_ext.cli.diag_cmd`) wraps this package.
"""

from .freeze_config import (
    DEFAULT_FREEZE_SETTINGS,
    FreezeSettings,
    dump_path,
    env_var_for,
    resolve_freeze_settings,
)
from .freeze_detector import (
    DEFAULT_FREEZE_CHECK_INTERVAL_S,
    DEFAULT_FREEZE_DIAG_ENV,
    FreezeDetector,
    FreezeDump,
    ThreadStackFrame,
)
from .recovery import (
    RecoveryAction,
    RecoverySpec,
    describe,
    recovery_actions,
)

__all__ = [
    "DEFAULT_FREEZE_SETTINGS",
    "DEFAULT_FREEZE_CHECK_INTERVAL_S",
    "DEFAULT_FREEZE_DIAG_ENV",
    "FreezeDetector",
    "FreezeDump",
    "FreezeSettings",
    "RecoveryAction",
    "RecoverySpec",
    "ThreadStackFrame",
    "describe",
    "dump_path",
    "env_var_for",
    "recovery_actions",
    "resolve_freeze_settings",
]
