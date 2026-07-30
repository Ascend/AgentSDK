"""Stability-gate scoped pytest fixtures.

Provides reusable fixtures for the byte-level snapshot tests added in P0-3:

- ``pinned_message_factory`` — build Message dataclasses with deterministic
  ``uuid`` and ``timestamp`` so byte-level ``to_dict()`` comparisons don't
  fluctuate due to ``uuid4()`` / ``datetime.now().isoformat()``.

The ``isolated_tmp_repo`` fixture (minimal git repo on ``tmp_path``)
lives in ``tests/conftest.py`` so the orchestrator tests can reuse it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

_PINNED_UUID_PREFIX = "00000000-0000-0000-0000-"
_PINNED_TIMESTAMP = "2026-01-01T00:00:00"


@pytest.fixture
def pinned_message_factory() -> Callable[..., object]:
    """Build Message dataclasses with deterministic uuid/timestamp.

    Returns a callable ``_make(role, content, **overrides)`` that returns
    an instance of the matching Message subclass (``UserMessage`` /
    ``AssistantMessage`` / ``SystemMessage``). Default ``uuid`` is
    monotonic (counter-derived); default ``timestamp`` is the pinned
    ISO string ``2026-01-01T00:00:00``. Both can be overridden via
    ``overrides``.

    Why a counter and not a literal: each call gets a unique uuid so
    multi-turn snapshots don't share identifiers; the counter makes the
    sequence deterministic across test runs.
    """

    from clawcodex_ext.types.content_blocks import ToolResultBlock, ToolUseBlock
    from clawcodex_ext.types.messages import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )

    counter = {"n": 0}

    def _next_uuid() -> str:
        counter["n"] += 1
        return f"{_PINNED_UUID_PREFIX}{counter['n']:012d}"

    def _make(role: str, content, **overrides):
        uuid = overrides.pop("uuid", _next_uuid())
        timestamp = overrides.pop("timestamp", _PINNED_TIMESTAMP)
        if role == "user":
            return UserMessage(content=content, uuid=uuid, timestamp=timestamp, **overrides)
        if role == "assistant":
            # AssistantMessage defaults content to [] — accept str by wrapping
            # in a TextBlock, matching ``create_assistant_message`` semantics.
            from clawcodex_ext.types.content_blocks import TextBlock

            if isinstance(content, str):
                block_content = [TextBlock(text=content)]
            else:
                block_content = content
            return AssistantMessage(content=block_content, uuid=uuid, timestamp=timestamp, **overrides)
        if role == "system":
            return SystemMessage(content=content, uuid=uuid, timestamp=timestamp, **overrides)
        raise ValueError(f"unsupported role for pinned_message_factory: {role!r}")

    # Pre-bind helpers to the factory so callers can construct typed blocks
    # without reaching into ``clawcodex_ext.types.content_blocks`` directly.
    _make.ToolUseBlock = ToolUseBlock
    _make.ToolResultBlock = ToolResultBlock
    return _make
