#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""NDJSON audit logging for ultraplan events."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .executor import StepTransition


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class AuditEntry:
    plan_id: str
    event: str
    timestamp: str
    payload: dict[str, Any]


class AuditLogger:
    def __init__(self, audit_dir: Path | str) -> None:
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def append(self, plan_id: str, event: str, payload: dict[str, Any]) -> AuditEntry:
        entry = AuditEntry(plan_id=plan_id, event=event, timestamp=_now(), payload=payload)
        path = self.path_for(plan_id)
        with self._lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True))
            fh.write("\n")
        return entry

    def record_transition(self, plan_id: str, transition: StepTransition) -> AuditEntry:
        return self.append(
            plan_id,
            "step.transition",
            {
                "step_id": transition.step_id,
                "sub_plan_id": transition.sub_plan_id,
                "old_status": transition.old_status.value,
                "new_status": transition.new_status.value,
                "transition_timestamp": transition.timestamp,
                "note": transition.note,
            },
        )

    def read(self, plan_id: str) -> list[AuditEntry]:
        path = self.path_for(plan_id)
        if not path.exists():
            return []
        entries: list[AuditEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            entries.append(
                AuditEntry(
                    plan_id=str(data["plan_id"]),
                    event=str(data["event"]),
                    timestamp=str(data["timestamp"]),
                    payload=dict(data.get("payload") or {}),
                )
            )
        return entries

    def path_for(self, plan_id: str) -> Path:
        return self.audit_dir / f"{plan_id}.ndjson"
