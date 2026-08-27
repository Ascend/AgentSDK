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
# ruff: noqa: F401

"""Public facade and dispatcher for LKB Plan Graph commands."""

from __future__ import annotations

from .commands import CommandResult, GraphCommand
from .error_codes import LkbErrorCode
from .graph_types import GraphSnapshot
from .json_store import BoardEnvelope
from .validation import ValidationRun
from .plan_graph_core import (
    AddDependencyHandler,
    ClaimTaskHandler,
    CreateTaskHandler,
    PlanCommandHandler,
    RemoveDependencyHandler,
    UpdateTaskFieldsHandler,
    _denied_issue,
    _plan_graph_id,
    _run,
    _task_ref,
    plan_graph_layer1,
)
from .plan_graph_handlers import (
    CompleteTaskHandler,
    DeleteTaskHandler,
    ReleaseTaskHandler,
    ReopenTaskHandler,
    RevalidateHandler,
    StartTaskHandler,
    TransferTaskHandler,
)
from .plan_graph_patch import PatchTaskHandler, _STATUS_KIND

__all__ = [
    "PlanCommandHandler",
    "CreateTaskHandler",
    "UpdateTaskFieldsHandler",
    "AddDependencyHandler",
    "RemoveDependencyHandler",
    "ClaimTaskHandler",
    "ReleaseTaskHandler",
    "TransferTaskHandler",
    "StartTaskHandler",
    "CompleteTaskHandler",
    "ReopenTaskHandler",
    "DeleteTaskHandler",
    "RevalidateHandler",
    "PatchTaskHandler",
    "PlanCommandDispatcher",
    "plan_command_dispatcher",
    "plan_graph_layer1",
]


class PlanCommandDispatcher:
    """Map ``command.kind`` to the matching :class:`PlanCommandHandler`."""

    def __init__(self) -> None:
        self._handlers: dict[str, PlanCommandHandler] = {}
        for handler_cls in (
            CreateTaskHandler,
            UpdateTaskFieldsHandler,
            AddDependencyHandler,
            RemoveDependencyHandler,
            ClaimTaskHandler,
            ReleaseTaskHandler,
            TransferTaskHandler,
            StartTaskHandler,
            CompleteTaskHandler,
            ReopenTaskHandler,
            DeleteTaskHandler,
            RevalidateHandler,
            PatchTaskHandler,
        ):
            instance = handler_cls()
            self._handlers[instance.kind] = instance

    def get(self, kind: str) -> PlanCommandHandler | None:
        return self._handlers.get(kind)

    def validate(self, command: GraphCommand, snapshot: GraphSnapshot) -> ValidationRun:
        handler = self._handlers.get(command.kind)
        if handler is None:
            return _run(
                command,
                accepted=False,
                issues=(
                    _denied_issue(
                        LkbErrorCode.UNKNOWN_COMMAND,
                        f"No handler for kind {command.kind!r}",
                    ),
                ),
            )
        # The plan_not_active (R-PG-001) and stale_revision (R-PG-002)
        # pre-gates are universal rules evaluated first by the Layer1
        # solver inside each handler's validate.
        return handler.validate(command, snapshot)

    def apply(
        self, command: GraphCommand, envelope: BoardEnvelope, validation: ValidationRun
    ) -> tuple[BoardEnvelope, CommandResult]:
        handler = self._handlers.get(command.kind)
        if handler is None:
            return envelope, CommandResult(
                decision="denied",
                command_id=command.command_id,
                error_code=LkbErrorCode.UNKNOWN_COMMAND,
                reason=str(LkbErrorCode.UNKNOWN_COMMAND),
            )
        graph = envelope.graphs.get(_plan_graph_id(command))
        if graph is not None:
            metadata = graph.get("plan")
            state = str(metadata.get("state") or "active") if isinstance(metadata, dict) else "active"
            if state != "active":
                return envelope, CommandResult(
                    decision="denied",
                    command_id=command.command_id,
                    error_code=LkbErrorCode.PLAN_NOT_ACTIVE,
                    reason=f"plan_not_active: Plan is {state}",
                )
        return handler.apply(command, envelope, validation)


_dispatcher: PlanCommandDispatcher | None = None


def plan_command_dispatcher() -> PlanCommandDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = PlanCommandDispatcher()
    return _dispatcher
