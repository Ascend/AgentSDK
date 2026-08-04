#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Summary and headless serializers for multi-model display results."""

from __future__ import annotations

import json
from typing import Iterable

from .protocol import ModelDisplayState


class SummaryBuilder:
    @staticmethod
    def build_text(results: Iterable[ModelDisplayState]) -> str:
        blocks: list[str] = []
        for result in results:
            seconds = "?" if result.duration_ms is None else f"{result.duration_ms / 1000:.1f}s"
            tokens = result.tokens.get("output", 0)
            blocks.append(f"───── {result.slot} ({seconds}, {tokens} tok) ─────\n{result.content}")
        return "\n\n".join(blocks)

    @staticmethod
    def build_json(results: Iterable[ModelDisplayState], *, strategy: str = "parallel") -> str:
        return json.dumps(
            {"multimodel": True, "strategy": strategy, "results": [result.to_dict() for result in results]},
            ensure_ascii=False,
        )
