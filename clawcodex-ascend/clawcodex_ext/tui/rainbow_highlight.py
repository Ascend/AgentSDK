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

"""Rich Text highlighting for ultraplan trigger keywords."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from clawcodex_ext.services.ultraplan.keyword_detector import TriggerHit


RAINBOW_STYLES: tuple[str, ...] = ("red", "yellow", "green", "cyan", "blue", "magenta")


def highlight_triggers(
    text: str,
    hits: list[TriggerHit],
    *,
    palette: tuple[str, ...] = RAINBOW_STYLES,
    fallback: str | None = None,
) -> Text:
    rendered = Text(text)
    if not hits:
        return rendered
    for hit in hits:
        if fallback:
            rendered.stylize(fallback, hit.start, hit.end)
            continue
        for offset, index in enumerate(range(hit.start, hit.end)):
            rendered.stylize(palette[offset % len(palette)], index, index + 1)
    return rendered


def should_render_rainbow(stream: Any) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())
