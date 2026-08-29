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

"""Dream consolidation runner.

Mirrors the ``runForkedAgent`` shape used upstream by
``claude-code-best/src/services/autoDream/autoDream.ts`` — i.e. a
function that takes a prompt, calls back to a progress watcher for
each assistant turn, and returns a final result.

**Phase A status:** the LLM invocation is intentionally a *stub*.
The actual agent-loop integration is a follow-up that depends on
clawcodex's own agent runner (which is not exposed as a public
library API in the way ``runForkedAgent`` is in TS).

The stub preserves the contract:

* takes a ``prompt`` string + ``on_message`` callback
* emits one assistant message with empty text + zero tool calls
  (so :func:`add_dream_turn` is exercised end-to-end)
* returns a :class:`DreamRunResult` with zero usage

This lets Phase A land the state machine + service main loop with
real tests; the runner swap is a one-line change in
:func:`run_dream_consolidation` once the agent runner is exposed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

_log = logging.getLogger(__name__)

__all__ = [
    "DreamRunResult",
    "DreamRunnerUnavailable",
    "run_dream_consolidation",
]


class DreamRunnerUnavailable(RuntimeError):
    """Raised when no dream runner is configured.

    The auto-dream service catches this and treats it as a soft skip
    (the time/session gates don't depend on the runner). The manual
    ``/dream`` skill surfaces it as a user-facing error.
    """


@dataclass
class DreamRunResult:
    """Final result of a dream consolidation run.

    Mirrors the relevant fields of upstream
    ``runForkedAgent``'s return value. ``usage`` defaults to an empty
    dict so the stub has a stable shape; the real runner fills it
    with token counters.
    """

    files_touched: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    summary: str = ""


class _ProgressCallback(Protocol):
    def __call__(self, *, text: str, tool_use_count: int, touched_paths: list[str]) -> None: ...


# Indirection so the service can swap implementations at startup
# (e.g. tests install a fake; production installs the real LLM-backed
# runner once it lands).
RunnerFactory = Callable[[], Callable[[str, _ProgressCallback], DreamRunResult]]
_runner_factory: RunnerFactory | None = None


def set_dream_runner_factory(factory: RunnerFactory | None) -> None:
    """Install or clear the runner factory.

    Pass ``None`` to fall back to the built-in stub.
    """
    global _runner_factory
    _runner_factory = factory


def run_dream_consolidation(
    prompt: str,
    on_message: _ProgressCallback | None = None,
) -> DreamRunResult:
    """Run a single dream consolidation pass.

    The Phase A stub:

    1. Logs the call (so it's visible in journald / dev logs).
    2. If ``on_message`` is set, emits one no-op turn.
    3. Returns a zero-usage :class:`DreamRunResult`.

    When a factory is installed via :func:`set_dream_runner_factory`,
    the factory's callable is invoked with the prompt + a thunk
    that wraps ``on_message`` into the factory's preferred shape.
    """
    if _runner_factory is not None:
        try:
            runner = _runner_factory()
        except Exception as e:  # pragma: no cover - defensive
            _log.warning("dream runner factory failed to construct: %s", e)
            raise DreamRunnerUnavailable(str(e)) from e
        try:
            return runner(prompt, on_message)
        except DreamRunnerUnavailable:
            raise
        except Exception as e:
            _log.warning("dream runner raised: %s", e)
            raise DreamRunnerUnavailable(str(e)) from e

    # Built-in stub path.
    _log.info(
        "dream consolidation stub: prompt_len=%d%s",
        len(prompt),
        " (with on_message)" if on_message is not None else "",
    )
    if on_message is not None:
        try:
            on_message(text="", tool_use_count=0, touched_paths=[])
        except Exception:  # pragma: no cover - defensive
            _log.debug("dream on_message callback raised", exc_info=True)
    return DreamRunResult(files_touched=[], usage={}, summary="(stub run)")


# ---------------------------------------------------------------------------
# Real LLM-backed runner
# ---------------------------------------------------------------------------

DREAM_MAX_TURNS: int = 25

DREAM_ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Glob",
        "Grep",
    }
)

DREAM_SYSTEM_PROMPT: str = (
    "You are running a dream \u2014 a background memory consolidation pass.\n\n"
    "Your PRIMARY goal is to WRITE memory files. You must produce or update "
    "at least one .md file in the memory directory during this run. "
    "Exploration without writing is a failed run.\n\n"
    "Follow the 4 phases in the user prompt in order:\n"
    "1. Orient \u2014 quickly scan what exists (2-3 tool calls max)\n"
    "2. Gather \u2014 search transcripts for new signal (3-5 tool calls max)\n"
    "3. Consolidate \u2014 WRITE new or updated memory files using the Write tool\n"
    "4. Prune \u2014 update MEMORY.md index to match actual files on disk\n\n"
    "Tool usage:\n"
    "- Bash: read-only commands only (ls, find, grep, cat, head, tail, wc, stat)\n"
    "- Read: read any file in the project or memory directory\n"
    "- Write/Edit: use these to CREATE and UPDATE memory files \u2014 this is your main job\n"
    "- Glob/Grep: search for files and content\n\n"
    "Memory file format: each .md file should have YAML frontmatter with "
    "name, description, and type fields. Keep files focused on one topic.\n\n"
    "Be decisive. If you find signal worth remembering, write it now. "
    "Do not defer or summarize without writing."
)


def _create_provider(model: str | None = None) -> Any:
    from clawcodex_ext.config import get_default_provider, get_provider_config  # pylint: disable=no-name-in-module
    from clawcodex_ext.providers.factory import get_provider_class

    provider_name = get_default_provider()
    provider_cfg = get_provider_config(provider_name)
    if not provider_cfg.get("api_key"):
        raise DreamRunnerUnavailable(
            f"API key for provider '{provider_name}' is not configured. Run `clawcodex login` to set it up."
        )
    provider_cls = get_provider_class(provider_name)
    return provider_cls(
        api_key=provider_cfg["api_key"],
        base_url=provider_cfg.get("base_url"),
        model=model or provider_cfg.get("default_model"),
    )


def _build_dream_tool_registry(provider: Any) -> Any:
    from clawcodex_ext.tool_system.registry import ToolRegistry
    from clawcodex_ext.tool_system.tools import ALL_STATIC_TOOLS

    registry = ToolRegistry()
    for tool in ALL_STATIC_TOOLS:
        if tool.name in DREAM_ALLOWED_TOOL_NAMES:
            registry.register(tool)
    return registry


def _build_dream_context(workspace_root: Path, abort_controller: Any) -> Any:
    from clawcodex_ext.tool_system.context import ToolContext
    from clawcodex_ext.permissions.types import ToolPermissionContext  # pylint: disable=no-name-in-module

    context = ToolContext(
        workspace_root=workspace_root,
        permission_context=ToolPermissionContext(
            mode="bypassPermissions",
            is_bypass_permissions_mode_available=True,
        ),
        abort_controller=abort_controller,
    )
    context.options.is_non_interactive_session = True
    context.ask_user = None
    return context


def _make_on_event_handler(
    on_message: Callable | None,
    files_touched: list[str],
) -> Callable | None:
    if on_message is None:
        return None

    def on_event(event: Any) -> None:
        if getattr(event, "kind", None) != "tool_use":
            return
        tool_name = getattr(event, "tool_name", "")
        tool_input = getattr(event, "tool_input", {}) or {}
        paths: list[str] = []
        if tool_name in ("Write", "Edit"):
            fp = tool_input.get("file_path") or tool_input.get("filePath")
            if fp:
                paths.append(fp)
                if fp not in files_touched:
                    files_touched.append(fp)
        try:
            on_message(
                text="",
                tool_use_count=1,
                touched_paths=paths,
            )
        except Exception:
            _log.debug("dream on_event callback raised", exc_info=True)

    return on_event


async def _run_dream_async(
    prompt: str,
    on_message: Callable | None,
    workspace_root: Path,
    max_turns: int,
) -> DreamRunResult:
    from clawcodex_ext.utils.abort_controller import AbortController  # pylint: disable=no-name-in-module
    from clawcodex_ext.types.messages import UserMessage
    from clawcodex_ext.query.agent_loop_compat import run_query_as_agent_loop

    try:
        provider = _create_provider()
    except DreamRunnerUnavailable:
        raise
    except Exception as e:
        raise DreamRunnerUnavailable(f"provider creation failed: {e}") from e

    try:
        tool_registry = _build_dream_tool_registry(provider)
    except Exception as e:
        raise DreamRunnerUnavailable(f"tool registry build failed: {e}") from e

    abort_controller = AbortController()
    tool_context = _build_dream_context(workspace_root, abort_controller)

    files_touched: list[str] = []
    on_event = _make_on_event_handler(on_message, files_touched)

    try:
        result = await run_query_as_agent_loop(
            initial_messages=[UserMessage(content=prompt)],
            system_prompt=DREAM_SYSTEM_PROMPT,
            provider=provider,
            tool_registry=tool_registry,
            tool_context=tool_context,
            max_turns=max_turns,
            on_event=on_event,
            abort_controller=abort_controller,
        )
    except Exception as e:
        raise DreamRunnerUnavailable(f"agent loop failed: {e}") from e

    return DreamRunResult(
        files_touched=list(files_touched),
        usage=dict(getattr(result, "usage", {}) or {}),
        summary=getattr(result, "response_text", "") or "",
    )


def run_dream_with_llm(
    prompt: str,
    on_message: Callable | None = None,
    *,
    workspace_root: Path | None = None,
    max_turns: int = DREAM_MAX_TURNS,
) -> DreamRunResult:
    """Run a dream consolidation pass using the real LLM agent loop."""
    ws = workspace_root or Path.cwd()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                _run_dream_async(prompt, on_message, ws, max_turns),
            )
            return future.result()
    else:
        return asyncio.run(_run_dream_async(prompt, on_message, ws, max_turns))


def create_real_dream_runner_factory() -> RunnerFactory:
    """Return a :data:`RunnerFactory` that produces the real LLM runner."""

    def factory() -> Callable:
        return run_dream_with_llm

    return factory


def wire_real_dream_runner() -> None:
    """Install the real LLM-backed dream runner globally."""
    set_dream_runner_factory(create_real_dream_runner_factory())
    _log.info("dream: real LLM runner installed")
