#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Helpers that port the TaskV2 / TodoWrite gating logic from TypeScript.

This mirrors a small slice of ``typescript/src/utils/tasks.ts`` — specifically
``isTodoV2Enabled()`` — so that we expose the same tool set to the model:

* Interactive sessions (REPL / TUI) expose ``TaskCreate``, ``TaskGet``,
  ``TaskUpdate`` and ``TaskList`` and hide ``TodoWrite``.
* Non-interactive headless / SDK sessions expose ``TodoWrite`` and hide the
  TaskV2 tools (unless ``CLAUDE_CODE_ENABLE_TASKS`` is set, mirroring the
  env-based opt-in in the TypeScript reference).
* Enabling LKB (the merged ``LKB_PLAN_GRAPH`` feature flag) also exposes the
  TaskV2 tools in headless sessions: the Plan Graph is the persistent
  authority behind the TaskV2 tools, so the flag turns on both together.
"""

from __future__ import annotations

import os

from src.bootstrap.state import get_is_non_interactive_session

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f", ""}


def _env_truthy(name: str) -> bool:
    """Matches ``isEnvTruthy`` in ``typescript/src/utils/envUtils.ts``.

    Returns ``True`` only when the environment variable is set to a recognised
    truthy value. Unset variables return ``False``. The test is
    case-insensitive.
    """
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _lkb_enabled() -> bool:
    """True when the merged LKB flag (``LKB_PLAN_GRAPH``) is on.

    Imported lazily so this module stays usable before the optional LKB
    package / feature-gate stack is importable.
    """
    try:
        from lkb.flags import is_plan_graph_enabled

        return is_plan_graph_enabled()
    except Exception:
        return False


def is_todo_v2_enabled() -> bool:
    """Port of ``isTodoV2Enabled`` from the TypeScript implementation.

    * Force-enabled when ``CLAUDE_CODE_ENABLE_TASKS`` is truthy (e.g. SDK users
      who prefer the TaskV2 tools).
    * Force-enabled when LKB (``LKB_PLAN_GRAPH``) is on — the Plan Graph backs
      the TaskV2 tools, so a single flag turns on Task V2 + LKB together,
      including in headless sessions.
    * Otherwise enabled only for interactive sessions (the default for
      ``clawcodex`` REPL and TUI).
    """
    if _env_truthy("CLAUDE_CODE_ENABLE_TASKS"):
        return True
    if _lkb_enabled():
        return True
    return not get_is_non_interactive_session()


__all__ = ["is_todo_v2_enabled"]
