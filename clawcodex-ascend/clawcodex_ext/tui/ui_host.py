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


"""Textual adapter for the interactive-command :class:`UIHost` port.

Implements the surface-agnostic ``UIHost`` (``src.command_system.types``) on
top of the Textual app's modal screen stack. Each ``select`` is an
``await app.push_screen_wait(GenericSelectScreen(...))`` issued from the
existing non-exclusive ``slash-cmd`` worker (``tui/app.py``), so no change to
the worker model — we add an *awaited* modal where today there is a
fire-and-forget ``push_screen(..., callback=)``.

A per-host ``asyncio.Lock`` serializes overlapping interactive commands: the
worker is non-exclusive, so two ``/permissions`` could overlap; the lock makes
the second one queue rather than open a nested modal (plan §8.1).
"""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence

from src.command_system.types import UIOption

from .screens.generic_input import GenericInputScreen
from .screens.generic_select import GenericSelectScreen


class TextualUIHost:
    """``UIHost`` backed by Textual modal screens."""

    def __init__(self, app: object) -> None:
        self._app = app
        # Serializes modal awaits so a second interactive command queues
        # instead of stacking a nested modal on the non-exclusive worker.
        self._lock = asyncio.Lock()

    async def select(
        self,
        title: str,
        options: Sequence[UIOption],
        *,
        current: Optional[str] = None,
    ) -> Optional[str]:
        async with self._lock:
            return await self._app.push_screen_wait(  # type: ignore[attr-defined]
                GenericSelectScreen(
                    title=title,
                    options=list(options),
                    current=current,
                )
            )

    async def prompt_text(
        self,
        title: str,
        *,
        default: str = "",
        placeholder: Optional[str] = None,
    ) -> Optional[str]:
        async with self._lock:
            return await self._app.push_screen_wait(  # type: ignore[attr-defined]
                GenericInputScreen(
                    title=title,
                    default=default,
                    placeholder=placeholder,
                )
            )

    async def display(self, title: str, body: str) -> None:
        # Surface read-only info as a Textual toast if the app supports it;
        # otherwise no-op. /permissions doesn't use this — kept minimal to
        # satisfy the UIHost contract.
        notify = getattr(self._app, "notify", None)
        if callable(notify):
            try:
                notify(body, title=title)
            except Exception:  # nosec B110
                pass  # This presentation update is best-effort and must not interrupt the user flow.


__all__ = ["TextualUIHost"]
