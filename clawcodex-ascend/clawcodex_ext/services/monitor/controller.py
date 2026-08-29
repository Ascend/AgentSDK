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

"""Monitor controller.

High-level wrapper around ``spawn_background_bash`` that tags spawned tasks
with ``kind='monitor'`` and applies the Windows ``watch`` compatibility shim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.tasks.local_shell import LocalShellTaskState, is_local_shell_task

from .text_tail import TextTailFollower
from .watch_compat import normalize_watch_command

TaskKind = Literal["shell", "monitor"]


@dataclass(frozen=True)
class MonitorStartResult:
    """Result returned by :meth:`MonitorController.start`."""

    task_id: str
    output_path: Path
    kind: TaskKind
    interval_sec: int | None
    command: str


class MonitorController:
    """Start, stop, list, and tail background monitor tasks."""

    def __init__(self, context: Any) -> None:
        """Create a controller bound to a ToolContext.

        ``context`` must expose ``runtime_tasks`` (RuntimeTaskRegistry) and
        ``background_bash_tasks`` (legacy dict view), which is satisfied by
        ``ToolContext``.
        """
        self._ctx = context

    def start(
        self,
        command: str,
        *,
        kind: TaskKind = "monitor",
        interval_sec: int | None = None,
        cwd: Path | None = None,
        description: str | None = None,
    ) -> MonitorStartResult:
        """Start a background monitor task.

        Steps:
          1. If ``kind='monitor'`` and on Windows, normalise ``watch -n``.
          2. Spawn via ``spawn_background_bash``.
          3. Tag the registered ``LocalShellTaskState`` with ``kind`` and
             ``interval_sec``.
          4. Return a ``MonitorStartResult``.
        """
        if kind == "monitor":
            final_command = normalize_watch_command(command)
        else:
            final_command = command

        target_cwd = cwd or getattr(self._ctx, "cwd", None) or self._ctx.workspace_root

        from clawcodex_ext.tool_system.tools.bash.background import spawn_background_bash  # pylint: disable=no-name-in-module

        output = spawn_background_bash(
            command=final_command,
            cwd=target_cwd,
            description=description or command,
            context=self._ctx,
        )

        task_id = output["backgroundTaskId"]
        output_path = Path(output["outputFilePath"])

        # Tag the runtime state with monitor metadata.
        def _tag(prev: Any) -> Any:
            if not is_local_shell_task(prev):
                return prev
            from dataclasses import replace

            return replace(
                prev,
                kind=kind,
                interval_sec=interval_sec,
                command=command,
            )

        self._ctx.runtime_tasks.update(task_id, _tag)

        # Mirror the tag into the legacy dict so older readers see it.
        legacy = self._ctx.background_bash_tasks.get(task_id)
        if isinstance(legacy, dict):
            legacy["kind"] = kind
            legacy["interval_sec"] = interval_sec

        return MonitorStartResult(
            task_id=task_id,
            output_path=output_path,
            kind=kind,
            interval_sec=interval_sec,
            command=command,
        )

    def stop(self, task_id: str) -> bool:
        """Stop a monitor task.  Returns True if the stop signal was sent."""
        from clawcodex_ext.tool_system.tools.bash.background import stop_background_bash  # pylint: disable=no-name-in-module

        return stop_background_bash(self._ctx, task_id)

    def list_active(self) -> list[LocalShellTaskState]:
        """Return all running monitor tasks."""
        result: list[LocalShellTaskState] = []
        for state in self._ctx.runtime_tasks.all():
            if not is_local_shell_task(state):
                continue
            if getattr(state, "kind", "shell") != "monitor":
                continue
            if state.status != "running":
                continue
            result.append(state)
        return result

    def tail(
        self,
        task_id: str,
        *,
        max_bytes: int = 200_000,
        follow: bool = False,
    ) -> TextTailFollower:
        """Return a tail reader for *task_id*'s log file.

        The reader starts at the end of the file minus ``max_bytes`` so the
        consumer immediately sees the most recent output.
        """
        state = self._ctx.runtime_tasks.get(task_id)
        if not isinstance(state, LocalShellTaskState):
            raise TypeError(f"not a local shell task: {task_id}")

        output_path = Path(state.output_path or state.output_file)
        follower = TextTailFollower(output_path, ring_size=max_bytes)

        # Start from the current end of the file minus ``max_bytes`` so the
        # consumer immediately sees the most recent output.
        try:
            start_offset = max(0, output_path.stat().st_size - max_bytes)
        except OSError:
            start_offset = 0

        # ``start`` is async but only touches local state; drive it once.
        import asyncio

        try:
            asyncio.get_running_loop()
            asyncio.create_task(follower.start(from_offset=start_offset))
        except RuntimeError:
            asyncio.run(follower.start(from_offset=start_offset))

        return follower

    def read(
        self,
        task_id: str,
        *,
        max_bytes: int = 200_000,
    ) -> dict[str, Any] | None:
        """Return a one-time snapshot of *task_id*'s output and status."""
        from clawcodex_ext.tool_system.tools.bash.background import read_background_output  # pylint: disable=no-name-in-module

        return read_background_output(self._ctx, task_id, max_bytes=max_bytes)


__all__ = ["MonitorController", "MonitorStartResult"]
