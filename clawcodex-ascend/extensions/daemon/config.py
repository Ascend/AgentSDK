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
"""Daemon configuration model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from .constants import (
    BACKOFF_CAP_MS,
    BACKOFF_INITIAL_MS,
    GRACEFUL_SHUTDOWN_TIMEOUT_MS,
)
from .errors import InvalidDaemonConfigError

DEFAULT_DAEMON_NAME: str = "remote-control"

DEFAULT_WORKER_KINDS: tuple[str, ...] = ("remoteControl",)

DEFAULT_SPAWN_MODE: str = "same-dir"

DEFAULT_CAPACITY: int = 4

DEFAULT_TIMEOUT_MS: int = GRACEFUL_SHUTDOWN_TIMEOUT_MS

DEFAULT_BACKOFF_INITIAL_MS: int = BACKOFF_INITIAL_MS

DEFAULT_BACKOFF_CAP_MS: int = BACKOFF_CAP_MS


@dataclass(frozen=True)
class DaemonConfig:
    """Immutable daemon configuration."""

    name: str = DEFAULT_DAEMON_NAME
    dir: Path = field(default_factory=Path.cwd)
    worker_kinds: tuple[str, ...] = DEFAULT_WORKER_KINDS
    spawn_mode: str = DEFAULT_SPAWN_MODE
    capacity: int = DEFAULT_CAPACITY
    permission_mode: str | None = None
    sandbox: bool = False
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    backoff_initial_ms: int = DEFAULT_BACKOFF_INITIAL_MS
    backoff_cap_ms: int = DEFAULT_BACKOFF_CAP_MS
    log_level: str = "INFO"
    extra_env: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        """Raise :class:`InvalidDaemonConfigError` on bad input."""
        if not self.name or not self.name.strip():
            raise InvalidDaemonConfigError("name must be non-empty")
        if "/" in self.name or "\\" in self.name or ".." in self.name:
            raise InvalidDaemonConfigError(f"name must not contain path separators or '..': {self.name!r}")
        if not self.worker_kinds:
            raise InvalidDaemonConfigError("worker_kinds must list at least one kind")
        if any(not k or not k.strip() for k in self.worker_kinds):
            raise InvalidDaemonConfigError("worker_kinds entries must be non-empty")
        if self.spawn_mode not in {"single-session", "worktree", "same-dir"}:
            raise InvalidDaemonConfigError(
                f"spawn_mode must be one of single-session/worktree/same-dir, got {self.spawn_mode!r}"
            )
        if self.capacity < 1:
            raise InvalidDaemonConfigError("capacity must be >= 1")
        if self.timeout_ms < 1_000:
            raise InvalidDaemonConfigError("timeout_ms must be >= 1000")
        if self.backoff_initial_ms < 1:
            raise InvalidDaemonConfigError("backoff_initial_ms must be >= 1")
        if self.backoff_cap_ms < self.backoff_initial_ms:
            raise InvalidDaemonConfigError("backoff_cap_ms must be >= backoff_initial_ms")

    def with_workers(self, kinds: Iterable[str]) -> "DaemonConfig":
        return replace(self, worker_kinds=tuple(kinds))

    def with_dir(self, dir_: Path) -> "DaemonConfig":
        return replace(self, dir=Path(dir_).resolve())

    def with_name(self, name: str) -> "DaemonConfig":
        return replace(self, name=name)
