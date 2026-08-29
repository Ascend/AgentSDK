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
