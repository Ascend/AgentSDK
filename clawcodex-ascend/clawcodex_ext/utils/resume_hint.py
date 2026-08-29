#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""Print 'Resume this session with: clawcodex --resume <sid>' on exit.

Mirrors CCB's ``printResumeHint()`` (typescript/src/utils/gracefulShutdown.ts
L141-176). Centralised so the REPL, TUI upstream/downstream, and headless
text mode all share one implementation.

Gating rules (from FEATURE_PLAN §6.2.1):
  * Skip when ``session_id`` is empty (transient / non-resumable runs).
  * Skip when the stream is not a TTY (don't pollute piped output,
    machine-readable JSON, or stream-json consumers — those paths
    already carry the session id in their structured payload via
    ResultEvent.session_id / NDJSON ``session_id`` field).

Idempotency
-----------
``print_resume_hint`` is **process-wide idempotent**: the first call that
passes the gates flips a module-level latch and subsequent calls in the
same process are no-ops. This protects against the S-R1 double-print
scenario where two code paths both fire the hint — e.g. the REPL ``/exit``
inline print at ``repl/core.py:4746`` and the atexit cleanup registered
by ``frontend/repl_extensions.py:_register_signal_session_save``. With
the latch, the hint is emitted exactly once per process.
"""

from __future__ import annotations

import logging
import sys
from typing import IO, Optional

__all__ = ["print_resume_hint", "reset_resume_hint_for_test_only"]

logger = logging.getLogger(__name__)


_RESUME_HINT_TEMPLATE = "\nResume this session with: clawcodex --resume {session_id}\n"

# Process-wide latch: once a hint is emitted, subsequent calls are no-ops.
# Tests should call :func:`reset_resume_hint_for_test_only` in setup so the
# flag does not leak across test cases.
_resume_hint_printed: bool = False


def _stream_isatty(stream: object) -> bool:
    """Return ``stream.isatty()`` if available, else ``False``.

    Defensive against non-standard streams (test fixtures using
    ``io.StringIO``, custom wrappers, etc.) that may lack ``isatty``.
    """
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except Exception:  # noqa: BLE001 — best-effort probe
        return False


def print_resume_hint(
    session_id: Optional[str],
    *,
    stream: Optional[IO[str]] = None,
) -> None:
    """Print the standard resume hint to ``stream`` (default ``sys.stdout``).

    No-op when ``session_id`` is empty/None, when the stream is not a TTY,
    or when this function has already printed a hint earlier in the same
    process. The latch matches the atexit-callback semantics: at most one
    hint per process is sufficient to convey the resume instruction.
    """
    global _resume_hint_printed
    if _resume_hint_printed:
        return
    sid = (session_id or "").strip()
    if not sid:
        return
    out = stream if stream is not None else sys.stdout
    if not _stream_isatty(out):
        return
    out.write(_RESUME_HINT_TEMPLATE.format(session_id=sid))
    try:
        out.flush()
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("Could not flush resume hint", exc_info=exc)
    _resume_hint_printed = True


def reset_resume_hint_for_test_only() -> None:
    """Clear the process-wide printed latch. **Test-only.**

    Production callers must not call this; the latch is intentionally
    process-wide so that atexit-callback double-fires are suppressed.
    Tests should call this in ``setup_method`` / ``autouse`` fixtures
    to avoid bleed between cases.
    """
    global _resume_hint_printed
    _resume_hint_printed = False
