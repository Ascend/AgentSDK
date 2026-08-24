# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
"""Data structures returned by the F-124 issue clarifier."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


AMBIGUITY_TYPES = frozenset({"missing", "vague", "contradictory", "unexecutable"})


@dataclass(frozen=True)
class ClarifyQuestion:
    question: str
    ambiguity_type: str
    evidence: str = ""
    suggested_options: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ambiguity_type": self.ambiguity_type,
            "evidence": self.evidence,
            "suggested_options": list(self.suggested_options),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClarifyQuestion":
        ambiguity_type = str(raw.get("ambiguity_type") or "vague").strip().lower()
        if ambiguity_type not in AMBIGUITY_TYPES:
            ambiguity_type = "vague"
        options = raw.get("suggested_options")
        if not isinstance(options, list):
            options = []
        return cls(
            question=str(raw.get("question") or "").strip()[:1000],
            ambiguity_type=ambiguity_type,
            evidence=str(raw.get("evidence") or "").strip()[:1000],
            suggested_options=tuple(
                str(option).strip()[:300] for option in options[:5] if option is not None and str(option).strip()
            ),
        )


@dataclass(frozen=True)
class ClarifyResult:
    is_clear: bool
    ambiguities: tuple[ClarifyQuestion, ...] = ()
    confidence: float = 0.0
    fingerprint: str = ""
    reason: str = ""
    degraded: bool = False
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def questions(self) -> list[str]:
        return [item.question for item in self.ambiguities if item.question]

    def with_runtime_fields(
        self,
        *,
        fingerprint: str | None = None,
        cached: bool | None = None,
    ) -> "ClarifyResult":
        updates: dict[str, Any] = {}
        if fingerprint is not None:
            updates["fingerprint"] = fingerprint
        if cached is not None:
            updates["cached"] = cached
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_clear": self.is_clear,
            "ambiguities": [item.to_dict() for item in self.ambiguities],
            "confidence": self.confidence,
            "fingerprint": self.fingerprint,
            "reason": self.reason,
            "degraded": self.degraded,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClarifyResult":
        ambiguity_rows = raw.get("ambiguities")
        if not isinstance(ambiguity_rows, list):
            ambiguity_rows = []
        ambiguities = tuple(ClarifyQuestion.from_dict(row) for row in ambiguity_rows if isinstance(row, dict))
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            is_clear=bool(raw.get("is_clear", True)),
            ambiguities=ambiguities,
            confidence=max(0.0, min(1.0, confidence)),
            fingerprint=str(raw.get("fingerprint") or ""),
            reason=str(raw.get("reason") or ""),
            degraded=bool(raw.get("degraded", False)),
            metadata=dict(raw.get("metadata") or {}),
        )
