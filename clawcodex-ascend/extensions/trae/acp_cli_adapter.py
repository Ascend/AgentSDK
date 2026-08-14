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
"""Trae Agent CLI wrapper — pseudo-ACP adapter.

Wraps ByteDance's open-source ``trae-agent`` (``trae-cli``) as a
*pseudo ACP server*, so clawcodex can drive Trae Agent capabilities
(code editing, command execution, trajectory recording) through the
unified :class:`ACPTransport` / :class:`ACPServer` interfaces.

Motivation:
  As of 2026-07 ``trae-agent`` is still a pure CLI tool (no stdio
  JSON-RPC server, no ACP implementation, see trae-agent #344). Its
  ``trae-cli run`` subcommand's input/output/intermediate trajectory
  already resembles the ACP event stream, so a thin adapter projects
  the CLI process plus its trajectory JSONL onto the ACP message stream
  (see the mapping table in :class:`TraeCliACPAdapter`).

Lives in ``extensions/trae/acp_cli_adapter.py`` — Layer 2 decoupling,
wired through the ``extensions/capabilities/acp_protocol`` Protocols.

Risk mitigation:
  - trae-agent interface changes -> fix ``_build_run_cmd`` in one place
  - Trajectory JSONL field changes -> ``_trajectory_to_acp`` degrades gracefully
  - subprocess on Windows -> marked experimental; prefer Mac/Linux
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from extensions.capabilities.acp_protocol import (
    ACPMessage,
    ACPMessageType,
    ACPSession,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TraeCliConfig",
    "TraeCliACPAdapter",
    "TrajectoryParseError",
]

# Maximum polling rounds while waiting for the trajectory file (50 x 0.1s = 5s)
_TRAJ_WAIT_ROUNDS = 50
_TRAJ_POLL_INTERVAL_S = 0.1


class TrajectoryParseError(Exception):
    """Raised when a trajectory JSONL line cannot be parsed (non-fatal by default)."""


@dataclass
class TraeCliConfig:
    """trae-cli launch config (deserializable from trae_config.yaml).

    Pin-by-version strategy: ``trae_cli_path`` defaults to ``trae-cli``
    (PATH lookup). If the CLI interface changes, fix
    :meth:`TraeCliACPAdapter._build_run_cmd` in one place.
    """

    trae_cli_path: str = "trae-cli"
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    extra_flags: list[str] = field(default_factory=list)


class TraeCliACPAdapter:
    """Wrap trae-cli as a pseudo ACP server.

    Implements the key lifecycle methods of the :class:`ACPServer`
    Protocol (create/resume/process/end) but **not** :class:`ACPTransport`
    (it plays the server role, not the transport role).

    Internal mapping:
      session/create  -> generate session_id + trajectory file path
                         (no CLI call)
      message/stream  -> tail <jsonl> and parse each line into an ACP message
      session/end     -> subprocess.terminate() + clean up trajectory file
      session/resume  -> trae-cli interactive --resume-trajectory <jsonl>

    Cleanup guarantee: :meth:`end_session` terminates first, then kills,
    to avoid orphan processes.
    """

    def __init__(
        self,
        config: TraeCliConfig,
        workspace: str,
        *,
        process_factory: Any | None = None,
    ) -> None:
        """
        Args:
            config: trae-cli launch config.
            workspace: working directory (trae-cli --working-dir).
            process_factory: optional ``subprocess.Popen`` replacement
                (unit tests inject a mock). ``None`` on the production
                path, which uses the real ``subprocess.Popen``.
        """
        self._cfg = config
        self._workspace = workspace
        self._procs: dict[str, subprocess.Popen] = {}
        self._trajectories: dict[str, Path] = {}
        self._traj_offsets: dict[str, int] = {}
        self._sessions: dict[str, ACPSession] = {}
        # Unit-test injection: callable replacing subprocess.Popen
        self._process_factory = process_factory or subprocess.Popen

    # ===== ACPServer interface =====

    async def create_session(self, workspace_path: str) -> ACPSession:
        """Maps to ACP session/create — generate sid + trajectory path, no CLI call.

        Difference from the doc: returns :class:`ACPSession` (not a bare
        sid) to match the :class:`ACPServer` Protocol signature. The sid
        is still available via ``session.id``.
        """
        sid = str(uuid.uuid4())
        traj_dir = Path(workspace_path) / ".trae" / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj = traj_dir / f"{sid}.jsonl"
        self._trajectories[sid] = traj
        session = ACPSession(
            id=sid,
            workspace_path=workspace_path,
            metadata={"trajectory": str(traj)},
        )
        self._sessions[sid] = session
        return session

    async def resume_session(self, session_id: str) -> ACPSession | None:
        """Maps to ACP session/resume — continue a saved trajectory in interactive mode.

        Returns ``None`` when the trajectory file or session is unknown.
        """
        traj = self._trajectories.get(session_id)
        session = self._sessions.get(session_id)
        if traj is None or session is None:
            return None
        if not traj.exists():
            return None
        session = self._sessions[session_id]
        cmd = [
            self._cfg.trae_cli_path,
            "interactive",
            "--resume-trajectory",
            str(traj),
            "--working-dir",
            session.workspace_path,
            *self._cfg.extra_flags,
        ]
        await self._spawn(session_id, cmd, env=self._env())
        return session

    async def process_message(self, msg: ACPMessage) -> AsyncIterator[ACPMessage]:
        """Maps to ACP message/stream — start trae-cli run, tail the trajectory line by line.

        The first call for a session launches ``trae-cli run``; later
        calls (same session) reuse the process and keep tailing.
        ``msg.content`` is passed as the task text to ``trae-cli run "<task>"``.
        """
        sid = msg.session_id
        if not sid:
            return
            yield  # noqa: E701 — make this an async generator for type checker
        task = msg.content if isinstance(msg.content, str) else json.dumps(msg.content or "")
        if sid not in self._procs:
            cmd = self._build_run_cmd(sid, task)
            await self._spawn(sid, cmd, env=self._env())
        async for evt in self._tail_trajectory(sid):
            yield self._trajectory_to_acp(sid, evt)

    async def end_session(self, session_id: str) -> None:
        """Maps to ACP session/end — terminate the process and remove the trajectory file.

        ``terminate()`` first, wait 5s, then ``kill()`` on timeout.
        """
        proc = self._procs.pop(session_id, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                logger.warning("trae-cli pid=%s killed after terminate timeout", proc.pid)
        # Clean up the trajectory file (optional — failure does not block)
        traj = self._trajectories.pop(session_id, None)
        if traj and traj.exists():
            try:
                traj.unlink()
            except OSError as exc:
                logger.warning("failed to remove trajectory %s: %s", traj, exc)
        self._sessions.pop(session_id, None)
        self._traj_offsets.pop(session_id, None)

    async def invoke_skill(self, skill_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Stub — placeholder for the skill bridge, injected by the upper ACPServer assembly."""
        return {"error": "skill bridge not wired in the ACP adapter", "skill": skill_name}

    async def handle_session(self, transport: Any) -> None:
        """Stub — the transport-driven main loop lives in the framework layer; this is only a backend."""
        raise NotImplementedError("TraeCliACPAdapter is a backend; transport loop is provided by the framework.")

    # ===== Internals =====

    def _build_run_cmd(self, sid: str, task: str) -> list[str]:
        """Build the ``trae-cli run`` command line (interface changes converge here).

        ``task`` is positional; trajectory/provider/model are flags.
        """
        traj = self._trajectories[sid]
        session = self._sessions[sid]
        return [
            self._cfg.trae_cli_path,
            "run",
            task,
            "--working-dir",
            session.workspace_path,
            "--trajectory-file",
            str(traj),
            "--provider",
            self._cfg.provider,
            "--model",
            self._cfg.model,
            *self._cfg.extra_flags,
        ]

    async def _spawn(self, sid: str, cmd: list[str], env: dict[str, str]) -> bool:
        """Launch the trae-cli subprocess (unit tests can inject a mock via process_factory)."""
        self._procs[sid] = self._process_factory(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        logger.info("trae-cli spawned: sid=%s pid=%s cmd=%s", sid, self._procs[sid].pid, cmd)
        return True

    async def _tail_trajectory(self, sid: str) -> AsyncIterator[dict[str, Any]]:
        """Tail the trajectory JSONL like ``tail -F``, one dict per line.

        Waits up to 5s for the file, then readline-loops until the
        process exits; unparseable lines are skipped.
        """
        traj = self._trajectories[sid]
        proc = self._procs[sid]
        # Wait for the trajectory file to appear
        for _ in range(_TRAJ_WAIT_ROUNDS):
            if traj.exists():
                break
            if proc.poll() is not None:
                # Process exited and the file never appeared — give up
                return
            await asyncio.sleep(_TRAJ_POLL_INTERVAL_S)
        if not traj.exists():
            logger.warning("trajectory file never appeared: %s", traj)
            return
        with traj.open("r", encoding="utf-8") as f:
            f.seek(self._traj_offsets.get(sid, 0))
            while True:
                line = f.readline()
                if not line:
                    if proc.poll() is not None:
                        # Process exited with no more data
                        return
                    await asyncio.sleep(_TRAJ_POLL_INTERVAL_S)
                    continue
                self._traj_offsets[sid] = f.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate and skip bad lines instead of raising
                    logger.debug("skip unparseable trajectory line: %r", line[:200])
                    continue
                if not isinstance(evt, dict):
                    logger.debug("skip non-object trajectory line: %r", line[:200])
                    continue
                yield evt

    def _trajectory_to_acp(self, sid: str, evt: dict[str, Any]) -> ACPMessage:
        """Map a trajectory event to an ACP message (graceful degradation).

        ``tool_name`` present -> :attr:`ACPMessageType.TOOL_CALL`;
        otherwise -> :attr:`ACPMessageType.MESSAGE_STREAM`. Missing
        fields degrade to a generic message instead of raising.
        """
        tool = evt.get("tool_name")
        content = evt.get("content", "")
        msg_id = evt.get("id") or str(uuid.uuid4())
        if tool:
            return ACPMessage(
                type=ACPMessageType.TOOL_CALL,
                id=msg_id,
                session_id=sid,
                tool_calls=[{"name": tool, "arguments": evt.get("tool_input", {})}],
                content=content,
                metadata={
                    "step": evt.get("step", "unknown"),
                    "model": evt.get("model"),
                },
            )
        return ACPMessage(
            type=ACPMessageType.MESSAGE_STREAM,
            id=msg_id,
            session_id=sid,
            content=content,
            metadata={
                "step": evt.get("step", "unknown"),
                "model": evt.get("model"),
            },
        )

    def _env(self) -> dict[str, str]:
        """Build the subprocess environment (provider/model/mcp_servers)."""
        env = dict(os.environ)
        env["TRAE_PROVIDER"] = self._cfg.provider
        env["TRAE_MODEL"] = self._cfg.model
        if self._cfg.mcp_servers:
            env["TRAE_MCP_SERVERS"] = json.dumps(self._cfg.mcp_servers)
        # Interop: if mcp_servers include the clawcodex bridge, env carries it
        return env
