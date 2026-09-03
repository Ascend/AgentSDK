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

"""OrchestratordBackend protocol: bundled protocol implementations.

Groups the runtime protocols into one registrable backend. A default
``ClawcodexBackend`` can live in ``clawcodex_ext``; entry-point
discovery can come later.
"""

from __future__ import annotations
# pylint: disable=W2301

from typing import Any, Callable, Protocol, runtime_checkable

from .agent_runtime import AgentRuntime
from .coordinator import CoordinatorContextProvider
from .diagnostics import DiagnosticsProbe
from .git_backend import GitBackend
from .im_channel import ImChannel
from .intent_focus import IntentFocus
from .provider import LLMProvider
from .session_storage import SessionStorage
from .workspace_tooling import WorkspaceTooling


@runtime_checkable
class OrchestratordBackend(Protocol):
    """Bundles all Protocol implementations into one discoverable unit.

    Register via Python entry_points (``[orchestratord_runtime.backends]``)
    in ``pyproject.toml`` — Phase 5 wires this; today the orchestrator
    uses upstream implementations directly.

    The default loader picks the first registered backend (or a
    user-specified one via ``ORCHESTRATORD_BACKEND`` env var).
    """

    name: str

    @property
    def agent_runtime(self) -> AgentRuntime: ...

    @property
    def workspace_tooling(self) -> WorkspaceTooling: ...

    @property
    def session_storage(self) -> SessionStorage: ...

    @property
    def im_channel_factory(self) -> Callable[[str], ImChannel]: ...

    @property
    def git_backend(self) -> GitBackend: ...

    @property
    def llm_provider(self) -> Callable[[str], LLMProvider]: ...

    @property
    def diagnostics_probe(self) -> DiagnosticsProbe: ...

    @property
    def intent_focus(self) -> IntentFocus: ...

    @property
    def coordinator_context(self) -> CoordinatorContextProvider: ...

    def health_check(self) -> dict[str, Any]:
        """Verify backend reachability; called by orchestrator on startup.

        Returns a JSON-serialisable mapping (e.g. ``{"agent_runtime": "ok"}``).
        Raise ``BackendUnavailable`` if any component is unreachable.
        """
        ...


class BackendUnavailable(RuntimeError):
    """Raised by ``OrchestratordBackend.health_check`` when one or more
    backend components are unreachable. Phase 5 activates this exception.
    """


__all__ = ["BackendUnavailable", "OrchestratordBackend"]
