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

# pylint: disable=no-member,consider-using-with,subprocess-popen-preexec-fn

"""SessionViewerManager — manage a temporary Visualizer subprocess lifecycle.

The manager is used by the standalone Web Dashboard (``cli/dashboard.py``) to
launch an ephemeral ``clawcodex-dev viz`` instance on-demand when the user
clicks "View Session" on an issue.

Key design decisions (v2, per architecture review):

* **Port reuse fast path**: before spawning a new process, probe
  ``http://127.0.0.1:8765/api/viz/health``.  If a Visualizer is already
  running on the default port, return it directly — zero delay for
  developers who keep a Visualizer tab open.
* **Lazy spawn**: the first ``ensure_running()`` call spawns the subprocess.
  Subsequent calls reuse the same process.
* **Random port**: ``socket.bind(('127.0.0.1', 0))`` finds a free port.
* **Idle timeout**: the subprocess is killed after a configurable idle
  period (default 5 minutes).
* **Cleanup on exit**: ``atexit`` hook kills the subprocess when the
  parent process (Dashboard) exits.
* **No ``--reload``**: ``uvicorn.run(app, reload=True)`` raises
  ``RuntimeError`` when the app is passed as an instance (see
  ``visualizer/cli.py:156-162``).  We never pass ``--reload``.
* **stderr=DEVNULL**: avoids pipe-buffer deadlock when the subprocess
  writes stderr but nobody reads it.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default Visualizer port used for the fast-path probe.
_DEFAULT_VIZ_PORT = 8765
_DEFAULT_VIZ_HEALTH_URL = f"http://127.0.0.1:{_DEFAULT_VIZ_PORT}/api/viz/health"


class SessionViewerManager:
    """Manage a temporary Visualizer subprocess lifecycle.

    Parameters
    ----------
    idle_timeout_s:
        Seconds of inactivity after which the subprocess is killed.
        Default 300 (5 minutes).
    """

    def __init__(self, idle_timeout_s: int = 300) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._port: int = 0
        self._idle_timeout_s = idle_timeout_s
        self._last_access: float = 0.0
        self._lock = threading.Lock()
        self._spawn_time: float = 0.0
        self._stop_event = threading.Event()
        self._cleaner_thread: Optional[threading.Thread] = None
        atexit.register(self._force_stop)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_running(self) -> int:
        """Return the temp Visualizer port.

        **Fast path**: if a Visualizer is already running on port 8765,
        return 8765 without spawning a new process.

        **Slow path**: spawn a new Visualizer on a random port, wait for
        readiness, and start the idle-cleaner background thread.
        """
        # Fast path: reuse existing Visualizer on default port.
        if self._is_viz_alive(_DEFAULT_VIZ_PORT):
            logger.info("Reusing existing Visualizer on port %d", _DEFAULT_VIZ_PORT)
            with self._lock:
                self._port = _DEFAULT_VIZ_PORT
                self._last_access = time.monotonic()
            return _DEFAULT_VIZ_PORT

        # Slow path: spawn a new one.
        with self._lock:
            now = time.monotonic()
            self._last_access = now
            if self._is_alive_locked():
                return self._port
            return self._spawn_locked()

    def stop(self) -> None:
        """Stop the temp Visualizer process.

        If the current port is the default 8765 (reused), we simply
        clear the cached state without killing the process — it belongs
        to the user, not to us.
        """
        with self._lock:
            self._stop_locked()

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._port == _DEFAULT_VIZ_PORT:
                return self._is_viz_alive(_DEFAULT_VIZ_PORT)
            return self._is_alive_locked()

    @property
    def port(self) -> int:
        with self._lock:
            return self._port

    @property
    def uptime_s(self) -> float:
        if self._spawn_time == 0.0:
            return 0.0
        return time.monotonic() - self._spawn_time

    @property
    def pid(self) -> int | None:
        """Return the subprocess PID, or None if not spawned/not alive."""
        with self._lock:
            if self._process is not None and self._is_alive_locked():
                return self._process.pid
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_viz_alive(port: int) -> bool:
        """Check if a Visualizer is already running on the given port."""
        import urllib.request

        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/viz/health",
                timeout=1.0,
            )
            return resp.status == 200
        except Exception:
            return False

    def _is_alive_locked(self) -> bool:
        if self._process is None:
            return False
        ret = self._process.poll()
        if ret is not None:
            logger.info("Session viewer process exited with code %s", ret)
            self._process = None
            self._port = 0
            return False
        return True

    def _spawn_locked(self) -> int:
        port = self._find_free_port()

        # NOTE: We use "clawcodex-dev viz" (the console_scripts entry point)
        # rather than "python -m clawcodex_dev" because the package has no
        # __main__.py.  The console_scripts entry is defined in pyproject.toml
        # as: clawcodex-dev = "clawcodex_ext.cli.main:main".
        #
        # IMPORTANT: Do NOT pass --reload.  uvicorn.run(app, reload=True)
        # raises RuntimeError when the app is passed as an instance (it
        # expects a module path string).  See visualizer/cli.py:156-162.
        cmd = [
            "clawcodex-dev",
            "viz",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--no-open",
        ]
        logger.info("Spawning session viewer on port %d: %s", port, " ".join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  # Drain to avoid pipe-buffer deadlock.
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except FileNotFoundError:
            logger.error(
                "clawcodex-dev not found — cannot start session viewer. "
                "Is the package installed and the console_scripts entry available?"
            )
            raise
        self._port = port
        self._spawn_time = time.monotonic()
        # Wait for the server to start (cold bootstrap can take 2-15s).
        self._wait_for_ready(port)
        # Start the idle-cleaner background thread.
        if self._cleaner_thread is None or not self._cleaner_thread.is_alive():
            self._stop_event.clear()
            self._cleaner_thread = threading.Thread(
                target=self._idle_cleaner_loop,
                daemon=True,
            )
            self._cleaner_thread.start()
        return port

    def _wait_for_ready(self, port: int, timeout_s: float = 15.0) -> None:
        """Poll the health endpoint until the server is ready.

        The cold-start time is dominated by ``clawcodex_ext/__init__.py``
        which eagerly installs permission extensions, memory extensions,
        and provider patches.  A timeout of 15s is conservative.
        """
        import urllib.request

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                resp = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/viz/health",
                    timeout=1.0,
                )
                if resp.status == 200:
                    logger.info("Session viewer ready on port %d", port)
                    return
            except Exception:  # nosec B110
                pass
            time.sleep(0.3)
        logger.warning("Session viewer did not respond within %ss", timeout_s)

    def _stop_locked(self) -> None:
        self._stop_event.set()
        # If we're reusing the default Visualizer, don't kill it — it
        # belongs to the user, not to us.
        if self._port == _DEFAULT_VIZ_PORT:
            self._port = 0
            self._spawn_time = 0.0
            return
        if self._process is not None:
            pgid = None
            try:
                pgid = os.getpgid(self._process.pid)
            except Exception:  # nosec B110
                pass
            # Try graceful SIGTERM first.
            try:
                self._process.terminate()
                self._process.wait(timeout=3.0)
            except Exception:
                # Force kill the process group.
                try:
                    if pgid:
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        self._process.kill()
                except Exception:  # nosec B110
                    pass
            self._process = None
            self._port = 0
            self._spawn_time = 0.0
            logger.info("Session viewer stopped")

    def _force_stop(self) -> None:
        """atexit hook — best-effort cleanup."""
        with self._lock:
            self._stop_locked()

    def _idle_cleaner_loop(self) -> None:
        """Background thread: kill the process if idle for too long."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=30.0)
            if self._stop_event.is_set():
                break
            with self._lock:
                if not self._is_alive_locked():
                    continue
                idle = time.monotonic() - self._last_access
                if idle > self._idle_timeout_s:
                    logger.info(
                        "Session viewer idle for %ds (timeout=%ds), stopping",
                        int(idle),
                        self._idle_timeout_s,
                    )
                    self._stop_locked()

    @staticmethod
    def _find_free_port() -> int:
        """Find a free TCP port.

        Binds a socket to ``127.0.0.1`` on port 0 (OS-assigned),
        extracts the assigned port, and closes the socket immediately.
        There is a tiny TOCTOU window between close and use, but it is
        negligible in practice.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
