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


import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

STATUS_PATH = "/session_status"
DEFAULT_MAX_PORT_TRIES = 10


def is_train_master_process():
    """Whether the current process runs on the training master node."""
    task_index = os.environ.get("VC_TASK_INDEX", "").strip()
    master_index = os.environ.get("MASTER_TRAIN_INDEX", "").strip()
    if not task_index or not master_index:
        return True
    try:
        return int(task_index) == int(master_index)
    except ValueError:
        return task_index == master_index


class _SessionStatusHandler(BaseHTTPRequestHandler):
    """Serve the JSON status payload and stay silent in the console."""

    def do_GET(self):
        if self.path.split("?", 1)[0] != STATUS_PATH:
            self.send_error(404)
            return
        try:
            payload = self.server.status_cb()
        except Exception as exp:
            payload = {"error": type(exp).__name__}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, log_format, *args):
        return None


class SessionStatusServer:
    """Threaded HTTP server for ``GET /session_status``."""

    def __init__(self, port, status_cb, host="0.0.0.0", max_port_tries=DEFAULT_MAX_PORT_TRIES,
                 mode="executor"):
        self._port = int(port)
        self._status_cb = status_cb
        self._host = host
        self._max_port_tries = int(max_port_tries)
        self._mode = mode
        self._httpd = None
        self._thread = None
        self._addr = None
        self._bound_port = None

    @property
    def addr(self):
        """``host:port`` actually bound, or None before start."""
        return self._addr

    def start(self):
        """Bind and serve in a daemon thread; return ``host:port`` or None."""
        last_error = None
        for offset in range(self._max_port_tries):
            port = self._port + offset
            try:
                self._httpd = ThreadingHTTPServer((self._host, port), _SessionStatusHandler)
            except OSError as exp:
                last_error = exp
                continue
            self._httpd.status_cb = self._status_cb
            self._httpd.daemon_threads = True

            self._thread = threading.Thread(
                target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
            )
            self._thread.start()
            bound_port = self._httpd.server_address[1]
            self._bound_port = bound_port
            self._addr = f"{self._host}:{bound_port}"
            if offset:
                logger.warning(f"[AURA_ROLLOUT] status.port.fallback port={bound_port} tries={offset + 1}")
            logger.info(f"[AURA_ROLLOUT] status.server.start addr={self._addr} path={STATUS_PATH}")
            return self._addr
        logger.error(
            f"[AURA_ROLLOUT] status.server.failed port={self._port} tries={self._max_port_tries} "
            f"error={type(last_error).__name__}"
        )
        return None

    def stop(self):
        """Stop serving and release the socket."""
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info(
            f"[AURA_ROLLOUT] status.server.stop port={self._bound_port} mode={self._mode}"
        )
        self._httpd = None
        self._thread = None
        self._addr = None
        self._bound_port = None
