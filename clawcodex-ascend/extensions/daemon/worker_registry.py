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
#
"""Worker kind → factory registry.

The supervisor looks up workers by *kind* (e.g. ``"remoteControl"``)
and instantiates a fresh one per spawn cycle. This module owns the
mapping and the lookup helpers.

The registry is a process-global singleton. Tests call
:meth:`reset` to start with a clean slate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from extensions.daemon.errors import UnknownWorkerKindError

if TYPE_CHECKING:
    from extensions.capabilities.daemon_protocol import Worker

logger = logging.getLogger(__name__)


WorkerFactory = Callable[[], "Worker"]


class WorkerRegistry:
    """Process-global map from worker kind → factory."""

    _factories: dict[str, WorkerFactory] = {}
    _kinds_seen_order: list[str] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, kind: str, factory: WorkerFactory | None = None) -> WorkerFactory:
        """Register *factory* under *kind*.

        Two calling styles are supported::

            # Direct registration
            WorkerRegistry.register("cron", make_cron_worker)

            # Decorator form
            @WorkerRegistry.register("cron")
            def make_cron_worker(): ...

        Re-registration with the same kind overwrites the previous
        entry — this is intentional so plugins can override built-in
        workers during tests.
        """
        kind = kind.strip()
        if not kind:
            raise ValueError("worker kind must be non-empty")

        def _bind(fn: WorkerFactory) -> WorkerFactory:
            if not callable(fn):
                raise TypeError(f"worker factory for {kind!r} is not callable")
            cls._factories[kind] = fn
            if kind not in cls._kinds_seen_order:
                cls._kinds_seen_order.append(kind)
            logger.debug("WorkerRegistry: registered kind=%s", kind)
            return fn

        if factory is not None:
            return _bind(factory)

        def _decorator(fn: WorkerFactory) -> WorkerFactory:
            return _bind(fn)

        return _decorator

    @classmethod
    def unregister(cls, kind: str) -> bool:
        """Remove *kind* from the registry. Returns True if it existed."""
        existed = cls._factories.pop(kind, None) is not None
        if existed:
            try:
                cls._kinds_seen_order.remove(kind)
            except ValueError:
                pass
        return existed

    @classmethod
    def reset(cls) -> None:
        """Clear every registration. Test-only."""
        cls._factories.clear()
        cls._kinds_seen_order.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, kind: str) -> "Worker":
        """Instantiate the worker registered under *kind*.

        Raises :class:`UnknownWorkerKindError` if no factory exists.
        """
        factory = cls._factories.get(kind)
        if factory is None:
            raise UnknownWorkerKindError(kind)
        return factory()

    @classmethod
    def has_kind(cls, kind: str) -> bool:
        """True iff *kind* is registered."""
        return kind in cls._factories

    @classmethod
    def known_kinds(cls) -> list[str]:
        """All registered kinds, in registration order."""
        return list(cls._kinds_seen_order)


__all__ = ["WorkerFactory", "WorkerRegistry"]
