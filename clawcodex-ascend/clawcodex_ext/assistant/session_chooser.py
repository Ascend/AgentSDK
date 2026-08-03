#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Stub mirror of ``typescript/src/assistant/AssistantSessionChooser.tsx``.

The upstream TS file is itself a stub annotated
``// Stub — AssistantSessionChooser not included in source snapshot``.
We mirror the stub here so the parity audit reflects that *both* trees
are intentionally empty. The Python REPL has no interactive picker for
``claude assistant`` yet; this module exists so future ports of
``dialogLaunchers`` / ``main.tsx`` have a real import target with the
same export name (``AssistantSessionChooser``, PascalCase) as the TS
source — no rename needed at the call sites.

Originally lived at ``src/assistant/session_chooser.py``; moved to
``clawcodex_ext/`` per the Decoupling Mandate. The ``src/`` shim is a
``sys.modules`` facade so ``from src.assistant.session_chooser import
...`` continues to work for the small number of legacy callers in
``extensions/`` and tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def AssistantSessionChooser(  # noqa: N802 — mirrors TS React component name
    sessions: Sequence[Any],
    on_select: Callable[[str], None],
    on_cancel: Callable[[], None],
) -> None:
    """Pick a session to attach to. No-op stub (returns ``None``).

    Matches the TS prop shape ``{ sessions, onSelect, onCancel }`` so
    callers can be ported without changing call sites. Name is PascalCase
    to match the TS React component export — a future port of
    ``dialogLaunchers.tsx::launchAssistantSessionChooser`` should be able
    to write ``from src.assistant.session_chooser import AssistantSessionChooser``
    without an alias.
    """


__all__ = ["AssistantSessionChooser"]
