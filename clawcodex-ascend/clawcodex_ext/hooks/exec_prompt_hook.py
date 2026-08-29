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

from __future__ import annotations

import logging
import time
from typing import Any

from .hook_types import HookConfig, HookResult

logger = logging.getLogger(__name__)


async def execute_prompt_hook(
    hook: HookConfig,
    stdin_data: dict[str, Any],
) -> HookResult:
    prompt_text = hook.prompt_text
    if not prompt_text:
        return HookResult(exit_code=0)

    start_time = time.monotonic()

    try:
        result = HookResult(
            exit_code=0,
            stdout=prompt_text,
            duration_ms=int((time.monotonic() - start_time) * 1000),
            additional_contexts=[prompt_text],
        )
        return result

    except Exception as e:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return HookResult(
            blocking_error=f"Prompt hook error: {e}",
            exit_code=-1,
            duration_ms=duration_ms,
        )
