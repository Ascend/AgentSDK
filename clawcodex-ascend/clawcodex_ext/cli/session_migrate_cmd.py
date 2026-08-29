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

"""Fast-path ``clawcodex-dev session migrate`` CLI command.

P5-H: register the ``session`` subcommand so operators can run::

    clawcodex-dev session migrate --from-3-file [--all] [--remove-legacy] [SESSION_ID]

Subcommand dispatch is handled by :mod:`clawcodex_ext.cli.subcommand_registry`
via the ``@register`` decorator. The actual migration logic lives in
``src/services/session_migrate.py`` so this module is purely a thin
CLI surface.
"""

from __future__ import annotations

import sys

from clawcodex_ext.cli.subcommand_registry import register

from src.services.session_migrate import handle_session_migrate_cli


@register("session")
def run_session_command(args: list[str]) -> int:
    """Dispatch ``session`` sub-subcommands (currently only ``migrate``)."""
    if not args:
        print(
            "usage: clawcodex-dev session migrate --from-3-file [--all] [--remove-legacy] [SESSION_ID]",
            file=sys.stderr,
        )
        return 2

    subcommand = args[0]
    rest = args[1:]

    if subcommand == "migrate":
        return handle_session_migrate_cli(rest)

    print(f"Unknown session command: {subcommand}", file=sys.stderr)
    print(
        "usage: clawcodex-dev session migrate --from-3-file [--all] [--remove-legacy] [SESSION_ID]",
        file=sys.stderr,
    )
    return 2


__all__ = ["run_session_command"]
