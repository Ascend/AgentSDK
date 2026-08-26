#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK publishes this standalone src-layout package across independent Parts; the complete
# ClawCodex source and focused tests validate imports and dynamic patterns during migration.
# The target hook also enables legacy default diagnostics beyond its declared high-value set.
# pylint: disable=E0402

"""Validation records persisted by the Plan Graph command pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .error_codes import LkbErrorCode
from .graph_types import RevisionVector
from .refs import NodeRef


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic rule violation."""

    code: LkbErrorCode
    message: str
    rule: str
    severity: Literal["warning", "error"] = "error"
    subject_ref: NodeRef | None = None
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": str(self.code),
            "message": self.message,
            "rule": self.rule,
            "severity": self.severity,
        }
        if self.subject_ref is not None:
            value["subjectRef"] = {
                "graph": self.subject_ref.graph,
                "kind": self.subject_ref.kind,
                "id": self.subject_ref.id,
            }
        if self.blockers:
            value["blockers"] = list(self.blockers)
        return value


@dataclass(frozen=True)
class ValidationRun:
    """Immutable result of validating one Graph command."""

    validation_run_id: str
    proposal_id: str
    subject_ref: NodeRef | None = None
    snapshot_hash: str = ""
    revision_vector: RevisionVector | None = None
    engine: str = "plan-graph"
    result: Literal["pass", "fail", "denied", "stale"] = "pass"
    derived_facts: tuple[str, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    requested_by: str = "system"
    created_at: str = ""

    @property
    def accepted(self) -> bool:
        return self.result == "pass"

    @property
    def status(self) -> Literal["accepted", "denied"]:
        return "accepted" if self.accepted else "denied"

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "validationRunId": self.validation_run_id,
            "proposalId": self.proposal_id,
            "snapshotHash": self.snapshot_hash,
            "revisionVector": (self.revision_vector.to_dict() if self.revision_vector is not None else None),
            "engine": self.engine,
            "result": self.result,
            "status": self.status,
            "derivedFacts": list(self.derived_facts),
            "issues": [issue.to_dict() for issue in self.issues],
            "requestedBy": self.requested_by,
            "createdAt": self.created_at,
        }
        if self.subject_ref is not None:
            value["subjectRef"] = {
                "graph": self.subject_ref.graph,
                "kind": self.subject_ref.kind,
                "id": self.subject_ref.id,
            }
        return value


__all__ = ["ValidationIssue", "ValidationRun"]
