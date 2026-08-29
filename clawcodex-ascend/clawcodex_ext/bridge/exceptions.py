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

"""Project-wide exceptions for the CCR bridge layer.

Phase-local exceptions (e.g., ``DirectConnectError``) live with their
respective modules; only cross-phase exceptions live here.

Originally lived at ``src/bridge/exceptions.py``; moved to
``clawcodex_ext/`` per the Decoupling Mandate. The ``src/`` shim is a
``sys.modules`` facade so ``from src.bridge.exceptions import ...``
continues to work for the small number of legacy callers in
``clawcodex_ext/`` and tests.
"""

from __future__ import annotations


class EpochSupersededError(Exception):
    """Raised by Phase 3's V2 transport when the server returns 409
    (worker_epoch superseded).

    Mirrors TS ``replBridgeTransport.ts:230`` which throws ``new Error(
    'epoch superseded')`` to unwind in-flight callers (``request()``).
    Catching this is how callers learn the epoch was bumped — they should
    drop the current transport and recreate it with a fresh epoch.
    """


class BridgeAuthError(Exception):
    """Raised when JWT cannot be decoded or refresh fails permanently.

    The ``TokenRefreshScheduler`` (WI-2.4) has a 3-failure cap; past that,
    the scheduler stops retrying and surfaces this exception to the caller.
    """


class BridgeFatalError(Exception):
    """Non-retryable error from the environments API.

    Mirrors TS ``BridgeFatalError`` exported from ``bridgeApi.ts:56-66``.
    Carries the HTTP status code and an optional server-provided error type
    (e.g. ``'environment_expired'``, ``'lifetime'``). Callers use
    ``is_expired_error_type(err.error_type)`` (Phase 3) to decide between
    teardown-and-recreate (expired) vs. fail-loudly (genuine 401/403).

    Args:
        message: Human-readable detail (typically ``f'{verb} {status}'``).
        status: HTTP status code.
        error_type: Optional server-provided error code from the response
            body's ``data.error.type`` field.
    """

    def __init__(
        self,
        message: str,
        status: int,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type

    def __repr__(self) -> str:
        return f"BridgeFatalError(status={self.status}, error_type={self.error_type!r}, message={str(self)!r})"


__all__ = ["BridgeAuthError", "BridgeFatalError", "EpochSupersededError"]
