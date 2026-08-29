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

"""output-style — deprecated ``/output-style`` command (port of TS local-jsx).

The TypeScript command (``typescript/src/commands/output-style/``) is a
``local-jsx`` command that renders nothing interactive: its ``call`` immediately
invokes ``onDone(<deprecation notice>, { display: 'system' })``. It exists only
to tell users the feature moved to ``/config``.

Ported as an :class:`InteractiveCommand` because ``local-jsx`` maps onto
``CommandType.INTERACTIVE`` (same remote-safety blocking by type). Unlike the
``/permissions`` exemplar it never touches ``context.ui`` — :meth:`run` returns
the deprecation :class:`InteractiveOutcome` directly, so it behaves identically
on every surface (REPL, Textual, and ``NullUIHost`` headless).
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import (
    CommandContext,
    InteractiveCommand,
    InteractiveOutcome,
)

# Verbatim from typescript/src/commands/output-style/output-style.tsx.
_DEPRECATION_NOTICE = (
    "/output-style has been deprecated. Use /config to change your output "
    "style, or set it in your settings file. Changes take effect on the next "
    "session."
)


@dataclass(frozen=True)
class OutputStyleCommand(InteractiveCommand):
    """Emit the deprecation notice and nothing else. Frozen + no new fields
    (the ``PermissionsCommand`` pattern); behavior lives entirely in
    :meth:`run`.
    """

    async def run(self, args: str, context: CommandContext) -> InteractiveOutcome:
        return InteractiveOutcome(
            message=_DEPRECATION_NOTICE,
            display="system",
        )


OUTPUT_STYLE_COMMAND = OutputStyleCommand(
    name="output-style",
    description="Deprecated: use /config to change output style",
    is_hidden=True,
)


__all__ = ["OUTPUT_STYLE_COMMAND", "OutputStyleCommand"]
