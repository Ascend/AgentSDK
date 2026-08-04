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

# pylint: disable=no-name-in-module
"""Auth CLI subcommand — login/logout/status."""

from __future__ import annotations

import sys

from clawcodex_ext.auth.codex_store import (
    AUTH_FILE,
    logout as _logout,
    zeroize_auth,
)
from clawcodex_ext.cli.subcommand_registry import register


@register("auth")
def run_auth_command(args: list[str]) -> int:
    """Handle ``clawcodex auth <subcommand>``.

    Subcommands:
        logout    — delete encrypted auth file + zeroize in-memory secrets
        status    — show whether auth file exists
        zeroize   — scrub in-memory credential cache only (keep file)
    """
    if not args:
        print("Usage: clawcodex auth [logout|status|zeroize]", file=sys.stderr)
        return 1

    cmd = args[0]
    if cmd == "logout":
        if not AUTH_FILE.exists():
            print("Already logged out — no auth file found.")
        else:
            _logout()
            print(f"Logged out. Auth file deleted and memory scrubbed: {AUTH_FILE}")
        return 0

    if cmd == "status":
        if AUTH_FILE.exists():
            size = AUTH_FILE.stat().st_size
            print(f"Auth file exists: {AUTH_FILE} ({size} bytes, encrypted)")
            return 0
        print("No auth file found — not logged in.")
        return 1

    if cmd == "zeroize":
        zeroize_auth()
        print("In-memory credential cache zeroized.")
        return 0

    print(f"Unknown auth subcommand: {cmd}", file=sys.stderr)
    print("Usage: clawcodex auth [logout|status|zeroize]", file=sys.stderr)
    return 1
