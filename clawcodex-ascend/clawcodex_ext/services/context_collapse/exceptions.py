#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Context collapse domain exceptions."""


class ContextCollapseError(RuntimeError):
    """Base error for context collapse operations."""


class TokenCountUnavailableError(ContextCollapseError):
    """Raised when no token counter implementation can be loaded.

    This is distinct from a zero-count: it means the runtime could not
    find ``tiktoken`` and no fallback was registered. The caller can
    treat this as a transient error (e.g. log a warning and use a
    heuristic counter instead of failing the request).
    """


class SummaryGeneratorError(ContextCollapseError):
    """Raised when a registered summary generator fails to produce output."""


class CollapseStateCorruptError(ContextCollapseError):
    """Raised when a collapse state file cannot be parsed on load."""


class CollapseStateNotFoundError(ContextCollapseError):
    """Raised when loading a collapse state file that does not exist."""


class ContextLengthExceededError(ContextCollapseError):
    """Raised when the input token count exceeds the configured ceiling.

    The 413 emergency-recovery path catches this and triggers a
    single-shot collapse, then re-raises if the context is still over
    budget after the recovery attempt.
    """
