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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""Generic remote task execution worker.

Replaces the old ``remoteControl`` worker and no longer depends on the
Anthropic Cloud bridge. Listens on a Unix Domain Socket and runs agent
tasks in subprocesses.

Design
------
* Listens on ``~/.clawcodex/task_server/task.sock`` (env-overridable).
* JSON-lines protocol (one JSON object per line).
* A connection may send multiple TaskRequests (kept open).
* Tasks run via a ``clawcodex-dev -p <prompt>`` subprocess.
* Results are written back on the same socket connection.

Example
-------
Send a task::

    echo '{"id":"t1","command":"run_agent","payload":{"prompt":"check code"}}' \\
      | socat - UNIX-CONNECT:~/.clawcodex/task_server/task.sock

"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess  # noqa: S404 — controlled subprocess spawn
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from extensions.capabilities.task_protocol import (
    TaskRequest,
    TaskResult,
)

logger = logging.getLogger(__name__)

# Default paths

DEFAULT_STATE_DIR = Path.home() / ".clawcodex" / "task_server"
ENV_SOCK_PATH = "CLAWCODEX_TASK_SOCK"
ENV_STATE_DIR = "CLAWCODEX_TASK_STATE_DIR"


# TaskRequest JSON parsing


def _parse_task_request(raw: dict[str, Any]) -> TaskRequest:
    """Parse a TaskRequest from a JSON dict, defaulting missing fields."""
    payload = raw.get("payload")
    metadata = raw.get("metadata")
    return TaskRequest(
        id=str(raw.get("id", uuid.uuid4().hex)),
        command=str(raw.get("command", "run_agent")),
        payload=payload if isinstance(payload, dict) else {},
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _task_result_to_json(result: TaskResult) -> str:
    """Serialize a TaskResult as a JSON line."""
    return json.dumps(
        {
            "task_id": result.task_id,
            "status": result.status,
            "output": result.output,
            "error": result.error,
            "exit_code": result.exit_code,
            "metadata": result.metadata,
        },
        ensure_ascii=False,
    )


# Task execution


async def _execute_task(request: TaskRequest) -> TaskResult:
    """Execute one task request.

    Supported commands:
    * ``run_agent`` — run an agent via a ``clawcodex-dev -p <prompt>`` subprocess
    * ``exec`` — run a shell command (from payload["cmd"])
    * ``ping`` — return a liveness acknowledgement
    """
    logger.info("[task_worker] executing task id=%s command=%s", request.id, request.command)

    if request.command == "ping":
        return TaskResult(
            task_id=request.id,
            status="completed",
            output="pong",
            exit_code=0,
            metadata={"executed_at": time.time()},
        )

    if request.command == "exec":
        cmd = request.payload.get("cmd", "")
        if not cmd:
            return TaskResult(
                task_id=request.id,
                status="failed",
                error="payload.cmd is required for 'exec' command",
                exit_code=1,
            )
        return await _run_subprocess(request.id, cmd, shell=True)

    if request.command == "run_agent":
        prompt = request.payload.get("prompt", "")
        if not prompt:
            return TaskResult(
                task_id=request.id,
                status="failed",
                error="payload.prompt is required for 'run_agent' command",
                exit_code=1,
            )
        cwd = request.payload.get("cwd") or os.getcwd()
        model = request.payload.get("model") or ""
        max_turns = request.payload.get("max_turns", 20)
        agent_cli = [sys.executable, "-m", "clawcodex"]
        if model:
            agent_cli += ["--model", str(model)]
        agent_cli += ["-p", prompt, "--max-turns", str(max_turns)]
        return await _run_subprocess(
            request.id,
            agent_cli,
            shell=False,
            cwd=cwd,
            timeout=request.payload.get("timeout"),
        )

    return TaskResult(
        task_id=request.id,
        status="failed",
        error=f"unknown command: {request.command!r}",
        exit_code=1,
    )


async def _run_subprocess(
    task_id: str,
    cmd: str | list[str],
    *,
    shell: bool = False,
    cwd: str | None = None,
    timeout: float | None = None,
) -> TaskResult:
    """Run a subprocess and capture its output."""
    proc: asyncio.subprocess.Process | None = None
    start = time.monotonic()
    try:
        if shell:
            proc = await asyncio.create_subprocess_shell(
                str(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
            )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            elapsed = time.monotonic() - start
            return TaskResult(
                task_id=task_id,
                status="failed",
                error=f"timed out after {timeout}s",
                exit_code=-1,
                metadata={"elapsed_s": round(elapsed, 3)},
            )

        elapsed = time.monotonic() - start
        out_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        err_text = stderr.decode("utf-8", errors="replace") if stderr else ""

        if proc.returncode == 0:
            return TaskResult(
                task_id=task_id,
                status="completed",
                output=out_text,
                exit_code=0,
                metadata={"elapsed_s": round(elapsed, 3)},
            )

        return TaskResult(
            task_id=task_id,
            status="failed" if proc.returncode != 78 else "permanent_failure",
            output=out_text,
            error=err_text,
            exit_code=proc.returncode or -1,
            metadata={"elapsed_s": round(elapsed, 3)},
        )

    except FileNotFoundError as exc:
        return TaskResult(
            task_id=task_id,
            status="failed",
            error=f"executable not found: {exc}",
            exit_code=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[task_worker] subprocess error task_id=%s", task_id)
        return TaskResult(
            task_id=task_id,
            status="failed",
            error=str(exc),
            exit_code=1,
        )
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()


# Socket server


def _get_sock_path(state_dir: Path | None = None) -> Path:
    """Return the Unix socket path.

    Precedence: ``CLAWCODEX_TASK_SOCK`` env var, then ``state_dir/task.sock``.
    """
    env_sock = os.environ.get(ENV_SOCK_PATH)
    if env_sock:
        return Path(env_sock)

    base = state_dir or Path(os.environ.get(ENV_STATE_DIR, DEFAULT_STATE_DIR))
    base = Path(base).expanduser().resolve()
    return base / "task.sock"


def _is_socket_alive(path: Path) -> bool:
    """Return ``True`` if another live server is listening on *path*."""
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(path))
            return True
        except OSError:
            return False
        finally:
            probe.close()
    except OSError:
        return False


class TaskServerWorker:
    """Generic remote task worker.

    Listens on a Unix Domain Socket, accepts JSON-lines task requests,
    and returns results.
    """

    kind = "task_server"

    def __init__(self, state_dir: str | Path | None = None) -> None:
        if state_dir is not None:
            self.state_dir = Path(state_dir).expanduser().resolve()
        else:
            self.state_dir = Path(os.environ.get(ENV_STATE_DIR, DEFAULT_STATE_DIR)).expanduser().resolve()
        self.sock_path = _get_sock_path(self.state_dir)
        self._server: asyncio.AbstractServer | None = None
        self._cancel_event: asyncio.Event | None = None
        self._started_at: float = 0.0

    # Worker lifecycle (RemoteTaskWorker Protocol)

    async def run(self, env: dict[str, str]) -> int:
        """Start the socket listen loop.

        Implements :class:`extensions.capabilities.task_protocol.RemoteTaskWorker`.
        """
        self._started_at = time.time()
        cancel = asyncio.Event()
        self._cancel_event = cancel

        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, cancel.set)
            except (NotImplementedError, RuntimeError):
                pass

        # Ensure the dir exists and clear a stale socket
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.sock_path.exists() and not _is_socket_alive(self.sock_path):
            self.sock_path.unlink(missing_ok=True)

        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self.sock_path),
            )
        except OSError as exc:
            logger.error("[task_worker] failed to bind socket %s: %s", self.sock_path, exc)
            return 78  # permanent

        # Restrict the socket to owner-only read/write
        try:
            self.sock_path.chmod(0o600)
        except OSError:
            pass

        logger.info(
            "[task_worker] listening on %s (pid=%d)",
            self.sock_path,
            os.getpid(),
        )

        try:
            async with self._server:
                await cancel.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self._cleanup()

        logger.info("[task_worker] shutdown complete")
        return 0

    def health_check(self) -> dict[str, Any] | None:
        """Return a health-check snapshot."""
        if not self._started_at:
            return None
        return {
            "kind": self.kind,
            "uptime_s": round(time.time() - self._started_at, 3),
            "socket": str(self.sock_path),
            "listening": self._server is not None and self._server.is_serving(),
        }

    # Internals

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection.

        Each connection may send multiple requests (one JSON object per
        line). The connection stays open until the client closes its
        write end or an error occurs.
        """
        peer = writer.get_extra_info("peername") or "unknown"
        logger.debug("[task_worker] client connected: %s", peer)

        try:
            while not self._cancel_event or not self._cancel_event.is_set():
                line = await reader.readline()
                if not line:
                    break  # client closed the connection

                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    err_result = TaskResult(
                        task_id="unknown",
                        status="failed",
                        error=f"invalid JSON: {exc}",
                        exit_code=1,
                    )
                    writer.write((_task_result_to_json(err_result) + "\n").encode("utf-8"))
                    await writer.drain()
                    continue
                if not isinstance(data, dict):
                    err_result = TaskResult(
                        task_id="unknown",
                        status="failed",
                        error="request must be a JSON object",
                        exit_code=1,
                    )
                    writer.write((_task_result_to_json(err_result) + "\n").encode("utf-8"))
                    await writer.drain()
                    continue

                request = _parse_task_request(data)
                result = await _execute_task(request)
                writer.write((_task_result_to_json(result) + "\n").encode("utf-8"))
                await writer.drain()

        except (ConnectionResetError, BrokenPipeError):
            logger.debug("[task_worker] client disconnected: %s", peer)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("[task_worker] handler error peer=%s", peer)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001  # nosec
                pass

    async def _cleanup(self) -> None:
        """Remove the socket file."""
        if self.sock_path.exists():
            try:
                self.sock_path.unlink(missing_ok=True)
            except OSError:
                pass


# WorkerRegistry factories


def build_task_server_worker() -> TaskServerWorker:
    """Factory for ``WorkerRegistry.register("task_server", ...)``."""
    return TaskServerWorker()


__all__ = [
    "DEFAULT_STATE_DIR",
    "TaskServerWorker",
    "build_task_server_worker",
]
