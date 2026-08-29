#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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
#

"""HeadlessOptions Protocol — interface for headless session configuration.

Lets ``src/api/query.py`` configure headless sessions without importing
from upstream entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Protocol

__all__ = ["HeadlessOptionsProtocol"]


class HeadlessOptionsProtocol(Protocol):
    """Protocol for headless run configuration.

    Concrete: src/entrypoints/headless.HeadlessOptions.
    """

    @property
    def prompt(self) -> str | None: ...

    @property
    def output_format(self) -> str: ...

    @property
    def input_format(self) -> str: ...

    @property
    def provider_name(self) -> str | None: ...

    @property
    def model(self) -> str | None: ...

    @property
    def max_turns(self) -> int: ...

    @property
    def permission_mode(self) -> str: ...

    @property
    def is_bypass_permissions_mode_available(self) -> bool: ...

    @property
    def allowed_tools(self) -> tuple[str, ...]: ...

    @property
    def disallowed_tools(self) -> tuple[str, ...]: ...

    @property
    def include_partial_messages(self) -> bool: ...

    @property
    def verbose(self) -> bool: ...

    @property
    def stdin(self) -> IO[str] | None: ...

    @property
    def stdout(self) -> IO[str] | None: ...

    @property
    def stderr(self) -> IO[str] | None: ...

    @property
    def workspace_root(self) -> Path | None: ...


class HeadlessRunnerProtocol(Protocol):
    """Protocol for the headless run function.

    Concrete: src/entrypoints/headless.run_headless.
    """

    def __call__(self, options: HeadlessOptionsProtocol) -> int: ...
