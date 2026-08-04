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

"""In-memory multi-model audit and cost accounting bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from clawcodex_ext.capabilities.multimodel_protocol import AggregatedOutput, MultiModelResult  # pylint: disable=no-name-in-module


@dataclass
class SessionBridge:
    """Retain per-slot outcomes for UI inspection and downstream persistence."""

    calls: list[list[MultiModelResult]] = field(default_factory=list)
    aggregated: list[AggregatedOutput | None] = field(default_factory=list)
    audit_path: Path | None = None

    def record(self, results: list[MultiModelResult], output: AggregatedOutput | None = None) -> None:
        self.calls.append(list(results))
        self.aggregated.append(output)
        if self.audit_path is not None:
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "results": [
                        {
                            "slot": item.slot_name,
                            "duration_ms": item.duration_ms,
                            "tokens": item.tokens,
                            "error": item.error,
                            "cancelled": item.cancelled,
                        }
                        for item in results
                    ],
                    "chosen_model": output.chosen.model if output else None,
                    "vote_summary": output.vote_summary if output else None,
                }
                with self.audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except OSError:
                # Auditing is best effort and must not discard an answer.
                pass

    @property
    def total_tokens(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for call in self.calls:
            for result in call:
                for key, value in result.tokens.items():
                    if isinstance(value, int):
                        totals[key] = totals.get(key, 0) + value
        return totals

    @property
    def total_duration_ms(self) -> int:
        return sum(result.duration_ms for call in self.calls for result in call)
