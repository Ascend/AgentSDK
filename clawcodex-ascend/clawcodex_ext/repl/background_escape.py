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

"""Signal object for Ctrl+B background escape in REPL mode.

When the user presses Ctrl+B during an active agent run in the REPL,
the LiveStatus keybinding handler invokes the ``on_background`` callback,
which sets a flag that causes ``chat()`` to raise this exception.
``chat()`` then catches it and triggers the background runner fork.

Using an exception (rather than a callback that directly calls ``os.fork``)
keeps the LiveStatus keybinding handler free of process-management logic —
it only signals intent, ``chat()`` decides what to do about it.
"""


class BackgroundEscape(Exception):
    """Raised when the user presses Ctrl+B during an active agent run.

    The REPL's ``chat()`` method catches this exception to trigger
    the background runner fork.  Using an exception (rather than a
    callback that directly calls ``os.fork``) keeps the LiveStatus
    keybinding handler free of process-management logic — it only
    signals intent, ``chat()`` decides what to do about it.
    """

    def __init__(self, message: str = "Background escape requested") -> None:
        super().__init__(message)
