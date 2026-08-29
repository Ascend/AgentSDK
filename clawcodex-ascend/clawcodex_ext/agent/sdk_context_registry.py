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

"""Session-scoped ``contextvars.Context`` buckets for SDK standalone wrappers.

Standalone SDK helpers (e.g. ``agent_teams.context.set_session_id``) store
state in ``ContextVar`` or task-local context.  Running each tool in a fresh
bash subprocess resets that state; reusing one ``Context`` object per logical
clawcodex session preserves writes across sequential tool calls in-process.
"""

from __future__ import annotations

import contextvars
import threading

ContextKey = tuple[str, str]


class SdkContextRegistry:
    """``(session_id, agent_id)`` → reusable :class:`contextvars.Context`."""

    def __init__(self) -> None:
        self._contexts: dict[ContextKey, contextvars.Context] = {}
        self._locks: dict[ContextKey, threading.RLock] = {}
        self._meta_lock = threading.RLock()

    def context_key(
        self,
        *,
        session_id: str | None,
        agent_id: str | None,
    ) -> ContextKey:
        return (session_id or "__default__", agent_id or "")

    def get_context(self, key: ContextKey) -> contextvars.Context:
        with self._meta_lock:
            ctx = self._contexts.get(key)
            if ctx is None:
                ctx = contextvars.Context()
                self._contexts[key] = ctx
            return ctx

    def lock_for(self, key: ContextKey) -> threading.RLock:
        with self._meta_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    def clear_session(self, session_id: str) -> None:
        with self._meta_lock:
            drop = [k for k in self._contexts if k[0] == session_id]
            for key in drop:
                self._contexts.pop(key, None)
                self._locks.pop(key, None)

    def reset(self) -> None:
        with self._meta_lock:
            self._contexts.clear()
            self._locks.clear()


_registry: SdkContextRegistry | None = None
_registry_lock = threading.Lock()


def get_sdk_context_registry() -> SdkContextRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = SdkContextRegistry()
    return _registry


def reset_sdk_context_registry() -> SdkContextRegistry:
    global _registry
    with _registry_lock:
        _registry = SdkContextRegistry()
        return _registry
