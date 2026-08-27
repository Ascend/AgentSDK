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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=E0402

"""Unified Read Model for the LKB board (spec §8.5, Phase 7).

:class:`LkbBoardView` is the UI-agnostic projection consumed by
``/lkb board``, ``/lkb status``, TaskList top-level summary, the REPL
task snapshot, and the TUI.  Renderers must NEVER read ``board.json`` or
re-derive domain rules - they only consume this view.

Badge priority is fixed (spec §8.3):
``validation_failed > needs_review > needs_recheck > blocked > running
> ready > verified``.  A historically-completed task that is now stale
shows ``NEEDS_RECHECK`` even when it also has active blockers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .graph_types import NodeRef, PlanSnapshot
from .json_store import BoardEnvelope

__all__ = [
    "LkbBoardRow",
    "LkbBoardSummary",
    "LkbBoardIssue",
    "LkbBoardView",
    "LkbBadge",
    "build_board_view",
    "BADGE_PRIORITY",
]


class LkbBadge(str, Enum):
    """Canonical derived-status badges used by LKB read-model consumers."""

    VALIDATION_FAILED = "validation_failed"
    NEEDS_REVIEW = "needs_review"
    NEEDS_RECHECK = "needs_recheck"
    BLOCKED = "blocked"
    RUNNING = "running"
    READY = "ready"
    VERIFIED = "verified"

    def __str__(self) -> str:
        return self.value


BADGE_PRIORITY = tuple(LkbBadge)
_PLAN_TASK_NODE_KIND = "task"


@dataclass(frozen=True)
class LkbBoardRow:
    task_id: str
    title: str
    owner: str
    base_status: str
    badge: LkbBadge
    active_blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class LkbBoardSummary:
    ready: int = 0
    running: int = 0
    blocked: int = 0
    needs_recheck: int = 0
    issues: int = 0


@dataclass(frozen=True)
class LkbBoardIssue:
    task_id: str
    message: str


@dataclass(frozen=True)
class LkbBoardView:
    board_id: str
    display_name: str
    store_revision: int
    plan_id: str
    plan_title: str
    plan_state: str
    plan_revision: int
    summary: LkbBoardSummary
    rows: tuple[LkbBoardRow, ...] = ()
    issues: tuple[LkbBoardIssue, ...] = ()
    suggested_actions: tuple[str, ...] = ()


def _badge_for(
    node: dict[str, Any],
    ref: NodeRef,
    plan: PlanSnapshot,
    validation_run: dict[str, Any] | None,
) -> LkbBadge:
    if validation_run is not None and validation_run.get("result") == "fail":
        return LkbBadge.VALIDATION_FAILED
    payload = node.get("payload") if isinstance(node.get("payload"), dict) else {}
    derived = str(payload.get("derived_status", "") or "")
    if derived == "needs_review":
        return LkbBadge.NEEDS_REVIEW
    if derived == "needs_recheck":
        return LkbBadge.NEEDS_RECHECK
    state = str(node.get("state", "pending"))
    if state == "completed":
        return LkbBadge.VERIFIED
    if ref in plan.blocked_ids:
        return LkbBadge.BLOCKED
    if state == "in_progress":
        return LkbBadge.RUNNING
    return LkbBadge.READY


def build_board_view(
    envelope: BoardEnvelope,
    plan_id: str = "plan",
) -> LkbBoardView:
    """Project an envelope into a UI-agnostic :class:`LkbBoardView`."""
    snapshot = envelope.build_graph_snapshot()
    plan = PlanSnapshot.from_graph(snapshot)

    board_dict = envelope.board if isinstance(envelope.board, dict) else {}
    board_id = str(board_dict.get("board_id", ""))
    display_name = str(board_dict.get("display_name", board_id))
    plan_graph = envelope.graphs.get(plan_id, {})
    plan_revision = int(plan_graph.get("revision", 0))
    plan_metadata = plan_graph.get("plan") if isinstance(plan_graph.get("plan"), dict) else {}

    rows: list[LkbBoardRow] = []
    issues: list[LkbBoardIssue] = []
    counts = {
        LkbBadge.READY: 0,
        LkbBadge.RUNNING: 0,
        LkbBadge.BLOCKED: 0,
        LkbBadge.NEEDS_RECHECK: 0,
    }

    # Dict insertion order is the durable append order used by the command
    # pipeline, so a later run for the same subject supersedes an older one.
    validation_by_ref: dict[str, dict[str, Any]] = {}
    for run in envelope.validation_runs.values():
        if not isinstance(run, dict):
            continue
        subject = run.get("subjectRef") or run.get("subject_ref")
        if isinstance(subject, dict):
            try:
                subject_key = NodeRef(
                    str(subject.get("graph", "")),
                    str(subject.get("kind", "")),
                    str(subject.get("id", "")),
                ).to_str()
            except ValueError:
                continue
            validation_by_ref[subject_key] = run

    # Sort rows by task_id for stable display.
    plan_nodes = sorted(
        (
            (NodeRef.from_str(str(n.get("ref", ""))), n)
            for n in envelope.nodes.values()
            if str(n.get("ref", "")).startswith(f"{plan_id}:{_PLAN_TASK_NODE_KIND}:")
        ),
        key=lambda kv: kv[0].id,
    )

    for ref, node in plan_nodes:
        badge = _badge_for(node, ref, plan, validation_by_ref.get(ref.to_str()))
        blockers = plan.active_blockers.get(ref, ())
        blocker_ids = tuple(b.id for b in blockers)
        rows.append(
            LkbBoardRow(
                task_id=ref.id,
                title=str(node.get("title", "")),
                owner=str(node.get("owner") or "-"),
                base_status=str(node.get("state", "pending")),
                badge=badge,
                active_blockers=blocker_ids,
            )
        )
        if badge in counts:
            counts[badge] += 1
        if badge is LkbBadge.BLOCKED:
            issues.append(
                LkbBoardIssue(
                    task_id=ref.id,
                    message=f"{ref.id} waits for {', '.join(blocker_ids) if blocker_ids else 'unknown'}",
                )
            )
        if badge is LkbBadge.NEEDS_RECHECK:
            cause = (
                (node.get("payload") or {}).get("invalidation_cause") if isinstance(node.get("payload"), dict) else None
            )
            issues.append(
                LkbBoardIssue(
                    task_id=ref.id,
                    message=f"{ref.id} needs recheck" + (f" (cause: {cause})" if cause else ""),
                )
            )
        elif badge in {LkbBadge.VALIDATION_FAILED, LkbBadge.NEEDS_REVIEW}:
            issues.append(
                LkbBoardIssue(
                    task_id=ref.id,
                    message=f"{ref.id} {badge.replace('_', ' ')}",
                )
            )

    suggested = _suggested_actions(rows, plan)
    return LkbBoardView(
        board_id=board_id,
        display_name=display_name,
        store_revision=envelope.store_revision,
        plan_id=plan_id,
        plan_title=str(plan_metadata.get("title") or plan_id),
        plan_state=str(plan_metadata.get("state") or "active"),
        plan_revision=plan_revision,
        summary=LkbBoardSummary(
            ready=counts[LkbBadge.READY],
            running=counts[LkbBadge.RUNNING],
            blocked=counts[LkbBadge.BLOCKED],
            needs_recheck=counts[LkbBadge.NEEDS_RECHECK],
            issues=len(issues),
        ),
        rows=tuple(rows),
        issues=tuple(issues),
        suggested_actions=tuple(suggested),
    )


def _suggested_actions(rows: list[LkbBoardRow], plan: PlanSnapshot) -> list[str]:
    actions: list[str] = []
    ready = [r for r in rows if r.badge is LkbBadge.READY]
    if ready:
        actions.append(f"claim {ready[0].task_id}")
    for row in rows:
        if row.badge is LkbBadge.NEEDS_RECHECK:
            actions.append(f"revalidate {row.task_id}")
    for row in rows:
        if row.badge in {
            LkbBadge.BLOCKED,
            LkbBadge.NEEDS_RECHECK,
            LkbBadge.NEEDS_REVIEW,
            LkbBadge.VALIDATION_FAILED,
        }:
            actions.append(f"explain {row.task_id}")
            break  # one explain suggestion is enough
    return actions
