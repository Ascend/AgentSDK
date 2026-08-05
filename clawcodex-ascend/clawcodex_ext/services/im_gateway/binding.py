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

"""Origin → unique-target binding policy.

A single IM ``origin`` may route to exactly one active target at a time.
The default target is a Gateway-hosted auto session. When a REPL or
orchestrator opts in, it overrides the default route; the override is
revoked on unbind or target termination, restoring the default. Binding
a second opt-in target overwrites the first and writes an audit record.
"""

# This split migration branch is linted before the aggregate package initializer is added.
# pylint: disable=relative-beyond-top-level

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Callable

from .models import (
    FEISHU_DM_ALL_ORIGIN,
    IM_DIRECT_ALL_ORIGIN,
    WECHAT_DIRECT_ALL_ORIGIN,
    OriginKey,
    SessionTarget,
)

logger = logging.getLogger(__name__)


@dataclass
class BindingEntry:
    origin: str
    target: SessionTarget
    registered_at: float = field(default_factory=time.time)
    connection_state: str = "active"  # active | offline | terminated

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "session_id": self.target.session_id,
            "host_type": self.target.host_type,
            "registered_at": self.registered_at,
            "connection_state": self.connection_state,
        }


BindingAuditor = Callable[[str, BindingEntry, BindingEntry | None], None]


class BindingPolicy:
    """Enforces unique opt-in binding per origin."""

    def __init__(self, auditor: BindingAuditor | None = None) -> None:
        self._bindings: dict[str, BindingEntry] = {}
        self._auditor = auditor or (lambda *a: None)

    def bind(self, origin: OriginKey | str, target: SessionTarget, *, now: float | None = None) -> BindingEntry:
        key = str(origin)
        ts = now if now is not None else time.time()
        entry = BindingEntry(origin=key, target=target, registered_at=ts)
        previous = self._bindings.get(key)
        for existing_key in list(self._bindings):
            if existing_key == key:
                continue
            if _same_exclusive_binding_group(key, existing_key):
                replaced = self._bindings.pop(existing_key)
                previous = previous or replaced
        self._bindings[key] = entry
        # Overwriting a previous binding is an auditable event. WeChat direct
        # origins are mutually exclusive at the channel level, so a wildcard
        # REPL binding and a specific-origin orchestrator binding replace each
        # other as well.
        if previous is not None and previous.target != target:
            self._auditor("binding_override", entry, previous)
            logger.info(
                "binding override: origin=%s session=%s (was session=%s)",
                key[:24],
                entry.target.session_id[:16],
                previous.target.session_id[:16],
            )
        elif previous is None:
            self._auditor("binding_created", entry, None)
            logger.info(
                "binding created: origin=%s session=%s host_type=%s",
                key[:24],
                entry.target.session_id[:16],
                entry.target.host_type,
            )
        return entry

    def unbind(self, origin: OriginKey | str) -> BindingEntry | None:
        key = str(origin)
        entry = self._bindings.pop(key, None)
        logger.info("binding unbound: origin=%s found=%s", key[:24], entry is not None)
        return entry

    def get(self, origin: OriginKey | str) -> BindingEntry | None:
        for candidate in _binding_candidates(str(origin)):
            entry = self._bindings.get(candidate)
            if entry is not None and entry.connection_state != "terminated":
                logger.debug(
                    "binding resolved: requested=%s matched=%s session=%s",
                    str(origin)[:24],
                    candidate[:24],
                    entry.target.session_id[:16],
                )
                return entry
        return None

    def is_opt_in(self, origin: OriginKey | str) -> bool:
        return self.get(origin) is not None

    def mark_offline(self, origin: OriginKey | str, *, session_id: str | None = None) -> None:
        matched = self._matched_binding(str(origin))
        entry = matched[1] if matched is not None else None
        if entry is not None and _matches_session(entry, session_id) and entry.connection_state == "active":
            entry.connection_state = "offline"
            self._auditor("binding_offline", entry, None)
            logger.info(
                "binding offline: origin=%s session=%s",
                str(origin)[:24],
                entry.target.session_id[:16],
            )

    def terminate(self, origin: OriginKey | str, *, session_id: str | None = None) -> None:
        matched = self._matched_binding(str(origin))
        entry = matched[1] if matched is not None else None
        if entry is not None and _matches_session(entry, session_id):
            entry.connection_state = "terminated"
            self._auditor("binding_terminated", entry, None)
            self._bindings.pop(matched[0], None)
            logger.info(
                "binding terminated: origin=%s session=%s",
                str(origin)[:24],
                entry.target.session_id[:16],
            )

    def _matched_binding(self, origin: str) -> tuple[str, BindingEntry] | None:
        for candidate in _binding_candidates(origin):
            entry = self._bindings.get(candidate)
            if entry is not None and entry.connection_state != "terminated":
                return candidate, entry
        return None

    def terminate_matching(self, origin: OriginKey | str) -> list[BindingEntry]:
        """Terminate all bindings in the same exclusive channel group."""
        key = str(origin)
        removed: list[BindingEntry] = []
        for existing_key in list(self._bindings):
            if not _same_exclusive_binding_group(key, existing_key):
                continue
            entry = self._bindings.pop(existing_key)
            entry.connection_state = "terminated"
            self._auditor("binding_terminated", entry, None)
            removed.append(entry)
            logger.info(
                "binding terminated (matching): origin=%s session=%s",
                existing_key[:24],
                entry.target.session_id[:16],
            )
        return removed

    def all_bindings(self) -> list[BindingEntry]:
        return list(self._bindings.values())


def _binding_candidates(origin: str) -> list[str]:
    """Return exact-to-broad binding keys for an inbound origin."""
    candidates = [origin]
    parts = origin.split(":")
    if origin == IM_DIRECT_ALL_ORIGIN:
        return candidates
    if len(parts) >= 4 and parts[0] == "wechat" and parts[1] == "direct":
        account = parts[2] or "*"
        candidates.append(f"wechat:direct:{account}:*")
        candidates.append(WECHAT_DIRECT_ALL_ORIGIN)
        candidates.append(IM_DIRECT_ALL_ORIGIN)
    elif len(parts) >= 4 and parts[0] == "feishu" and parts[1] == "dm":
        app_id = parts[2] or "*"
        candidates.append(f"feishu:dm:{app_id}:*")
        candidates.append(FEISHU_DM_ALL_ORIGIN)
        candidates.append(IM_DIRECT_ALL_ORIGIN)
    return candidates


def _exclusive_binding_group(origin: str) -> str:
    parts = origin.split(":")
    if origin == IM_DIRECT_ALL_ORIGIN:
        return "im:direct"
    if len(parts) >= 4 and parts[0] == "wechat" and parts[1] == "direct":
        return "wechat:direct"
    if len(parts) >= 4 and parts[0] == "feishu" and parts[1] == "dm":
        return "feishu:dm"
    return origin


def _same_exclusive_binding_group(left: str, right: str) -> bool:
    left_group = _exclusive_binding_group(left)
    right_group = _exclusive_binding_group(right)
    if "im:direct" in {left_group, right_group}:
        return left_group in {"im:direct", "wechat:direct", "feishu:dm"} and right_group in {
            "im:direct",
            "wechat:direct",
            "feishu:dm",
        }
    return left_group == right_group


def _matches_session(entry: BindingEntry, session_id: str | None) -> bool:
    return session_id is None or entry.target.session_id == session_id


__all__ = ["BindingAuditor", "BindingEntry", "BindingPolicy"]
