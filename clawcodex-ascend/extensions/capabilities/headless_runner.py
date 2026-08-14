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
"""Headless runner registry.

Pluggable headless execution backend, defaulting to
``clawcodex_ext.entrypoints.headless.run_headless``. Callers in
``src.api.query`` use only :func:`run_headless_session`, keeping the
runtime import off the upstream path.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = ["HeadlessSessionOptions", "run_headless_session"]


@dataclass
class HeadlessSessionOptions:
    """Options for a headless session run.

    Mirrors the fields of ``clawcodex_ext.entrypoints.headless.HeadlessOptions``
    that ``QueryRunner`` actually uses.
    """

    prompt: str
    workspace_root: Path
    provider_name: str | None = None
    model: str | None = None
    max_turns: int = 20
    permission_mode: str = "default"
    stdout: io.StringIO = field(default_factory=io.StringIO)
    stderr: io.StringIO = field(default_factory=io.StringIO)
    on_event: Callable[[Any], None] = field(default=lambda e: None)
    # Env vars merged into the headless session's subprocess env.
    env: dict[str, str] = field(default_factory=dict)
    # Extra text appended to the effective system prompt, carrying the
    # constant workflow background so per-turn messages stay minimal.
    append_system_prompt: str | None = None
    # Forwarded to the headless session so the caller can cooperatively
    # cancel a session running on an executor thread.
    abort_controller: Any | None = None
    agent_id: str | None = None
    runtime_tasks: Any | None = None
    # When set, run_headless resumes the existing transcript.
    resume_session_id: str | None = None


def make_abort_controller() -> Any:
    """Create an AbortController without importing upstream at module load.

    The same instance is forwarded via ``HeadlessSessionOptions.abort_controller``.
    """
    from src.utils.abort_controller import AbortController

    return AbortController()


def run_headless_session(
    options: HeadlessSessionOptions,
) -> int:
    """Run a headless session and return the exit code.

    * Default / ``CLAW_HEADLESS_BACKEND=upstream``: lazily delegates to
      ``clawcodex_ext.entrypoints.headless.run_headless``.
    * ``CLAW_HEADLESS_BACKEND=stub``: returns 0 immediately (for tests
      that do not exercise the full agent loop).

    Custom backends can be registered via ``set_headless_backend()``.
    """
    backend = os.getenv("CLAW_HEADLESS_BACKEND", "").lower() or _active_backend[0]

    if backend == "stub":
        # Exercise the event bridge without running the agent.
        options.on_event(_make_stub_tool_event("tool_use", "bash", {}, False, "1"))
        options.on_event(_make_stub_tool_event("tool_result", "bash", {"output": "ok"}, False, "1"))
        return 0

    # Lazy import: forward on_event so tool events reach the orchestrator
    # stream, without importing upstream internals at module load.
    from clawcodex_ext.entrypoints.headless import HeadlessOptions, run_headless

    options_legacy = HeadlessOptions(
        prompt=options.prompt,
        output_format="text",
        provider_name=options.provider_name,
        model=options.model,
        max_turns=options.max_turns,
        permission_mode=options.permission_mode,
        workspace_root=options.workspace_root,
        stdout=options.stdout,
        stderr=options.stderr,
        on_event=options.on_event,
        env=options.env,
        append_system_prompt=options.append_system_prompt or "",
        abort_controller=options.abort_controller,
        agent_id=options.agent_id,
        runtime_tasks=options.runtime_tasks,
        resume_session_id=options.resume_session_id,
    )
    from clawcodex_ext.coordinator.mode import coordinator_mode_context

    coordinator_enabled = str(options.env.get("CLAUDE_CODE_COORDINATOR_MODE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    with coordinator_mode_context(coordinator_enabled):
        return run_headless(options_legacy)


def _make_stub_tool_event(
    kind: str,
    tool_name: str,
    tool_input: dict[str, Any],
    is_error: bool,
    tool_use_id: str,
) -> Any:
    """Build a minimal object duck-typing as a ToolEvent."""

    # The on_event callback receives a frozen ToolEvent dataclass; build
    # a lightweight object with the same attribute layout.
    class _StubEvent:
        __slots__ = (
            "kind",
            "tool_name",
            "tool_input",
            "tool_output",
            "tool_use_id",
            "is_error",
            "error",
        )

        def __init__(self):
            self.kind = kind
            self.tool_name = tool_name
            self.tool_input = tool_input
            self.tool_output: Any = None
            self.tool_use_id = tool_use_id
            self.is_error = is_error
            self.error: str | None = None

    return _StubEvent()


# Backend registry (allows tests / custom embeddings to override).
_active_backend: list[str] = ["upstream"]


def set_headless_backend(name: str) -> None:
    """Set the active headless runner backend ("upstream" or "stub")."""
    _active_backend[0] = name
