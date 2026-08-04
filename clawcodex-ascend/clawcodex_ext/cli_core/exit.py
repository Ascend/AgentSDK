#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""CLI exit helpers.

Port of ``typescript/src/cli/exit.ts``. Centralizes the ``print + exit`` block
copy-pasted across subcommand handlers and gives callers a ``NoReturn`` type
so control-flow analysis narrows correctly after the call.
"""

from __future__ import annotations

import sys
from typing import NoReturn


def cli_error(msg: str | None = None, code: int = 1) -> NoReturn:
    """Write ``msg`` to stderr (if provided) and exit with ``code`` (default 1)."""

    if msg:
        print(msg, file=sys.stderr)
    sys.exit(code)


def cli_ok(msg: str | None = None) -> NoReturn:
    """Write ``msg`` to stdout (if provided) and exit with code 0."""

    if msg:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    sys.exit(0)
