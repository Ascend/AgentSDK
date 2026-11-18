#!/usr/bin/env python3
# coding=utf-8

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


import enum
import threading
import time

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

DEFAULT_STATUS_PORT = 18080

ZERO_SESSIONS = {"active": 0, "finished": 0, "aborted": 0, "placeholder": 0}


class SessionState(enum.Enum):
    """Lifecycle state of one rollout session."""

    CREATED = "created"
    FINISHED = "finished"
    ABORTED = "aborted"


class SessionManager:
    """Thread-safe in-process session registry."""

    def __init__(self, enabled=True):
        self._enabled = enabled
        self._lock = threading.Lock()
        self._active = {}
        self._counters = dict(ZERO_SESSIONS)
        self._terminal = set()

    @property
    def enabled(self):
        """Whether logs are emitted."""
        return self._enabled

    def create_session(self, session_id, prompt_id=None, slot=None):
        """Register a new session before the interaction starts."""
        session_id = str(session_id)
        with self._lock:
            self._terminal.discard(session_id)
            self._active[session_id] = (prompt_id, time.time())
            self._counters["active"] += 1
        if self._enabled:
            logger.info(
                f"[AURA_ROLLOUT] session.create sid={session_id} prompt_id={prompt_id} slot={slot}"
            )
        return session_id

    def finalize_session(self, session_id, steps=None):
        """Mark a session as normally finished. """
        session_id = str(session_id)
        with self._lock:
            record = self._active.pop(session_id, None)
            if record is not None:
                elapsed = time.time() - record[1]
                prompt_id = record[0]
                self._counters["active"] -= 1
                self._counters["finished"] += 1
                self._terminal.add(session_id)
            else:
                elapsed = 0.0
                prompt_id = None
                known = session_id in self._terminal
        if not self._enabled:
            return
        if record is not None:
            logger.info(
                f"[AURA_ROLLOUT] session.finalize sid={session_id} prompt_id={prompt_id} "
                f"steps={steps} elapsed={elapsed:.3f}"
            )
        else:
            logger.info(
                f"[AURA_ROLLOUT] session.finalize.late sid={session_id} "
                f"known={str(known).lower()}"
            )

    def abort_session(self, session_id, reason=""):
        """Mark a session as aborted; the caller keeps the batch alive."""
        session_id = str(session_id)
        reason = str(reason or "unknown").replace(" ", "_")
        with self._lock:
            record = self._active.pop(session_id, None)
            if record is not None:
                elapsed = time.time() - record[1]
                prompt_id = record[0]
                self._counters["active"] -= 1
                self._counters["aborted"] += 1
                self._terminal.add(session_id)
            else:
                elapsed = 0.0
                prompt_id = None
                known = session_id in self._terminal
        if not self._enabled:
            return
        if record is not None:
            logger.warning(
                f"[AURA_ROLLOUT] session.abort sid={session_id} prompt_id={prompt_id} "
                f"reason={reason} elapsed={elapsed:.3f}"
            )
        else:
            logger.info(
                f"[AURA_ROLLOUT] session.abort.late sid={session_id} "
                f"reason={reason} known={str(known).lower()}"
            )

    def record_placeholder(self, prompt_id, index=None, session_id=None):
        """Record a zero-reward placeholder inserted for a failed session."""
        with self._lock:
            self._counters["placeholder"] += 1
        if self._enabled:
            logger.warning(
                f"[AURA_ROLLOUT] session.placeholder prompt_id={prompt_id} index={index} sid={session_id}"
            )

    def shutdown(self, timeout=0.0, poll_interval=0.05):
        """Drain active sessions, aborting whatever remains after ``timeout``. """
        start = time.time()
        with self._lock:
            start_active = len(self._active)
        if self._enabled:
            logger.info(f"[AURA_ROLLOUT] sessionmgr.drain.start active={start_active}")

        deadline = start + max(0.0, float(timeout))
        while time.time() < deadline:
            with self._lock:
                if not self._active:
                    break
            time.sleep(poll_interval)

        with self._lock:
            remaining = list(self._active.keys())
        if not remaining:
            if self._enabled:
                logger.info(
                    f"[AURA_ROLLOUT] sessionmgr.drain.done active=0 "
                    f"elapsed={time.time() - start:.3f}"
                )
            return 0

        if self._enabled:
            logger.warning(f"[AURA_ROLLOUT] sessionmgr.drain.timeout remain={len(remaining)}")
        for session_id in remaining:
            self.abort_session(session_id, reason="shutdown")
        return len(remaining)

    def get_status(self):
        """Return the session counters."""
        with self._lock:
            return self._status_locked()

    def log_status(self):
        """Emit one INFO summary line per iteration."""
        if not self._enabled:
            return
        status = self.get_status()
        logger.info(
            f"[AURA_ROLLOUT] sessionmgr.status active={status['active']} "
            f"finished={status['finished']} aborted={status['aborted']} "
            f"placeholder={status['placeholder']}"
        )

    def _status_locked(self):
        return dict(self._counters)


class NullSessionManager:
    """Pass-through manager used when ``rollout_manager.enabled`` is false."""

    enabled = False

    def create_session(self, session_id, prompt_id=None, slot=None):
        return session_id

    def finalize_session(self, session_id, steps=None):
        return None

    def abort_session(self, session_id, reason=""):
        return None

    def record_placeholder(self, prompt_id, index=None, session_id=None):
        return None

    def shutdown(self, timeout=0.0, poll_interval=0.05):
        return 0

    def get_status(self):
        return {key: 0 for key in ZERO_SESSIONS}

    def log_status(self):
        return None
