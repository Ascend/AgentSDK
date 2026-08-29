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

"""Call handlers for agent-created tools."""

from .bash import (
    BashCallError,
    execute_bash,
    parse_sop_wrapper_stdout,
    resolve_bundle_venv_environment,
)
from .http import HttpCallError, execute_http
from .python import PythonCallError, execute_python
from .sdk_wrapper import (
    SdkWrapperCallError,
    execute_sdk_wrapper_in_process,
    parse_sdk_wrapper_cli_options,
    parse_sdk_wrapper_call_impl,
    should_use_in_process_sdk_wrapper,
    wrapper_uses_instance_cache,
)

__all__ = [
    "BashCallError",
    "execute_bash",
    "parse_sop_wrapper_stdout",
    "resolve_bundle_venv_environment",
    "HttpCallError",
    "execute_http",
    "PythonCallError",
    "execute_python",
    "SdkWrapperCallError",
    "execute_sdk_wrapper_in_process",
    "parse_sdk_wrapper_cli_options",
    "parse_sdk_wrapper_call_impl",
    "should_use_in_process_sdk_wrapper",
    "wrapper_uses_instance_cache",
]
