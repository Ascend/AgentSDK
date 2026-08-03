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

"""Environment-variable helpers — Chunk G / WI-8.1 prerequisite.

Mirrors ``typescript/src/utils/envUtils.ts``. ``is_env_truthy`` was
duplicated in ``src/agent/fork_subagent.py:_is_env_truthy``; per the
critic's Chunk-F note we hoist it here as the canonical location and
re-export from fork_subagent for back-compat.

The canonical truthy set matches the TS spirit (``"1"``, ``"true"``,
``"yes"``, ``"on"``, case-insensitive after stripping). Using a
constant frozenset rather than membership tests against a tuple gives
O(1) lookup and pins the set to "exactly these values."

Originally lived at ``src/utils/env.py``; moved to ``clawcodex_ext/`` per
the Decoupling Mandate. The ``src/`` shim is a ``sys.modules`` facade so
``from src.utils.env import is_env_truthy`` continues to work for the
small number of legacy callers in ``clawcodex_ext/`` and ``extensions/``.
"""

from __future__ import annotations

import os
from typing import Final

_TRUTHY_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def is_env_truthy(name: str) -> bool:
    """Return True iff ``os.environ[name]`` is set to a truthy value.

    Mirrors ``isEnvTruthy`` in ``typescript/src/utils/envUtils.ts``.
    Unset / empty / whitespace-only values are False; the canonical
    accepted truthy strings are ``"1"``, ``"true"``, ``"yes"``, ``"on"``
    after lowercasing and stripping.
    """
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY_VALUES


__all__ = [
    "is_env_truthy",
]
