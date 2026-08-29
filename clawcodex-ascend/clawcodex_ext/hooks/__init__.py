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

"""Hook system — PreToolUse, PostToolUse, Stop, Notification, PostSampling hook execution runtime.

Mirrors TypeScript utils/hooks.ts and hooks/ directory.
"""

from __future__ import annotations

from .config_manager import HookConfigManager
from .hook_types import (
    AGENT_HOOK_TIMEOUT_MS,
    ALL_HOOK_EVENTS,
    HTTP_HOOK_TIMEOUT_MS,
    TOOL_HOOK_EXECUTION_TIMEOUT_MS,
    HookConfig,
    HookEvent,
    HookProgress,
    HookResult,
    HookSource,
    HookType,
    NotificationHookInput,
    PostSamplingHookInput,
    PostToolUseHookInput,
    PreToolUseHookInput,
    ShellType,
    StopHookInput,
    UserPromptSubmitHookInput,
)
from .registry import (
    AsyncHookRegistry,
    RegisteredHook,
    get_global_hook_registry,
    reset_global_hook_registry,
)
from .shell_invocation import (
    DEFAULT_HOOK_SHELL,
    SHELL_TYPES,
    build_powershell_args,
    find_powershell_path,
)

__all__ = [
    "AGENT_HOOK_TIMEOUT_MS",
    "ALL_HOOK_EVENTS",
    "DEFAULT_HOOK_SHELL",
    "HTTP_HOOK_TIMEOUT_MS",
    "SHELL_TYPES",
    "TOOL_HOOK_EXECUTION_TIMEOUT_MS",
    "AsyncHookRegistry",
    "HookConfig",
    "HookConfigManager",
    "HookEvent",
    "HookProgress",
    "HookResult",
    "HookSource",
    "HookType",
    "NotificationHookInput",
    "PostSamplingHookInput",
    "PostToolUseHookInput",
    "PreToolUseHookInput",
    "RegisteredHook",
    "ShellType",
    "StopHookInput",
    "UserPromptSubmitHookInput",
    "build_powershell_args",
    "find_powershell_path",
    "get_global_hook_registry",
    "reset_global_hook_registry",
]
