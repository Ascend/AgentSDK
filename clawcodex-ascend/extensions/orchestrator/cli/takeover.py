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

"""F-49 takeover: read-only snapshot viewer for a running orchestrator agent.

Spawns a ``--resume`` REPL against the agent's ``run_id`` so the operator
can inspect the current conversation history. The orchestrator agent is
**not** paused — it continues running unaffected. When the REPL exits,
the orchestrator proceeds normally. There is no handback: takeover is a
pure read-only snapshot of the on-disk ``transcript.jsonl``.

    clawcodex orchestrator issue takeover --id ISSUE-1
        │
        ├─ IssueRegistry.get_by_issue_ref(issue_id)
        │   → (run_id, workspace_path)             # authoritative
        │
        └─ subprocess: python3 -m src.cli --resume <run_id>
              │   (cwd = workspace_path)
              │
              └─ ClawCodexExtREPL(resume_session_id=run_id)
                    → Session.resume(run_id)            # directory format
                    → REPL displays the conversation history

Reads only; no orchestrator coupling beyond :class:`IssueRegistry`.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _TakeoverTarget:
    """Resolved target for the takeover flow."""

    run_id: str
    workspace_path: Path
    issue_id: str  # for header / log lines


def _resolve_target(
    registry_path: Path | None,
    workspace_root: Path | None,
    issue_id: str | None,
    run_id: str | None,
) -> _TakeoverTarget | None:
    """Look up ``(run_id, workspace_path)`` via :class:`IssueRegistry`
    or accept ``--run`` + ``--workspace`` directly.

    Lookup priority:
      1. ``--id <issue_id>`` via
         :meth:`IssueRegistry.get_by_issue_ref`. Returns the
         record's ``run_id`` and ``workspace_path``.
      2. ``--run <run_id>`` + ``--workspace <path>`` (or resolved
         ``workspace_root``). The caller is responsible for the
         workspace because there is no inverse index from
         ``run_id`` → ``workspace_path`` in the registry.
      3. Otherwise ``None`` — caller emits a usage / lookup error.
    """
    from extensions.orchestrator.issue_registry import IssueRegistry

    if issue_id:
        if registry_path is None or not registry_path.exists():
            return None
        try:
            registry = IssueRegistry(registry_path)
        except Exception:
            return None
        record = registry.get_by_issue_ref(issue_id)
        if record is None or record.run_id is None:
            return None
        if record.workspace_path is None:
            return None
        return _TakeoverTarget(
            run_id=record.run_id,
            workspace_path=Path(record.workspace_path),
            issue_id=record.issue_identifier or record.issue_id,
        )

    if run_id and workspace_root is not None:
        # ``--run`` mode: workspace comes from ``--workspace`` or
        # the resolved ``workspace_root``. We don't have the issue
        # identifier here, so use the run_id as a label.
        return _TakeoverTarget(
            run_id=run_id,
            workspace_path=Path(workspace_root),
            issue_id=f"run:{run_id}",
        )

    return None


def _spawn_resume_repl(
    run_id: str,
    workspace_path: Path,
) -> int:
    """Spawn ``python3 -m src.cli --resume <run_id>`` with
    ``cwd=workspace_path`` and block on its exit code. The REPL
    inherits stdout/stderr so the operator sees the same UX as a
    direct ``--resume`` call. The workspace is conveyed via
    ``cwd`` — the CLI parser has no ``--workspace`` flag.

    Returns the REPL's exit code.
    """
    try:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "src.cli",
                "--resume",
                run_id,
            ],
            cwd=str(workspace_path),
        )
    except FileNotFoundError as exc:
        print(
            f"error: failed to spawn REPL — {exc}. Check that `python3 -m src.cli` is on PATH.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"error: REPL spawn failed — {exc}", file=sys.stderr)
        return 1


def _run_takeover(
    registry_path: Path | None,
    workspace_root: Path | None,
    args: argparse.Namespace,
) -> int:
    """Sync wrapper around :func:`_run_takeover_async` for the CLI
    dispatcher.

    The dispatcher at ``extensions.orchestrator/cli/issue.py``
    runs in the top-level (non-async) ``run()`` context, so it
    cannot ``await`` directly. This wrapper calls
    :func:`asyncio.run` on the async core. Tests that need to
    drive the full flow inside an existing event loop can
    ``await _run_takeover_async(...)`` instead — same logic, no
    ``asyncio.run`` conflict.
    """
    return asyncio.run(
        _run_takeover_async(registry_path, workspace_root, args),
    )


async def _run_takeover_async(
    registry_path: Path | None,
    workspace_root: Path | None,
    args: argparse.Namespace,
) -> int:
    """Async core of the takeover flow. See :func:`_run_takeover`
    for the contract; this is the version that callers inside an
    existing event loop (e.g. async tests) should ``await``.
    """
    issue_id = getattr(args, "id", None) or getattr(args, "issue_id", None)
    run_id = getattr(args, "run", None) or getattr(args, "run_id", None)
    workspace_arg = getattr(args, "workspace", None)

    if not issue_id and not run_id:
        print(
            "error: --id <issue_id> or --run <run_id> is required",
            file=sys.stderr,
        )
        return 2

    if run_id and not workspace_root and not workspace_arg:
        print(
            "error: --run requires --workspace (or a resolved workspace root)",
            file=sys.stderr,
        )
        return 2

    effective_workspace = Path(workspace_arg) if workspace_arg else workspace_root

    target = _resolve_target(
        registry_path,
        effective_workspace,
        issue_id,
        run_id,
    )
    if target is None:
        if issue_id:
            print(
                f"error: no active run found for issue {issue_id!r}. Nothing to take over.",
                file=sys.stderr,
            )
        else:
            print(
                f"error: could not resolve target for run {run_id!r}",
                file=sys.stderr,
            )
        return 1

    print(
        f"Starting takeover REPL for {target.issue_id} (run {target.run_id}) in {target.workspace_path}",
        file=sys.stderr,
    )

    # Ask the running agent to flush its in-memory transcript buffer to
    # disk so the REPL can display the latest conversation history. The
    # agent is NOT paused — it keeps running. If the control socket is
    # unavailable (agent already ended or not started), skip and rely on
    # whatever is already on disk.
    sock_path = target.workspace_path / ".run_control" / f"{target.run_id}.sock"
    if sock_path.exists():
        flushed = await _send_flush_transcript(sock_path)
        if flushed:
            # Give the agent a moment to process the command and flush.
            await asyncio.sleep(1.0)

    # Wait for transcript.jsonl to land so the REPL can display the
    # conversation history. If it doesn't appear within the timeout
    # (agent still warming up), fall back to a minimal session.json stub
    # so the REPL starts cleanly. The agent is NOT paused — it keeps
    # running while the operator inspects the snapshot. If the transcript
    # already exists, no stub is written (it would shadow the real data).
    _wait_for_transcript(target.run_id, timeout=5.0)
    _ensure_session_stub(target.run_id)

    return _spawn_resume_repl(target.run_id, target.workspace_path)


async def _send_flush_transcript(sock_path: Path) -> bool:
    """Send a ``flush_transcript`` command over the control socket.

    This asks the running agent to flush its in-memory transcript buffer
    to ``transcript.jsonl`` so the REPL can read the latest conversation
    history. The agent is NOT paused — it keeps running.

    Returns ``True`` if the command was sent, ``False`` if the socket is
    unavailable (agent already ended or not started).
    """
    from extensions.orchestrator.control_socket import send_cmd

    try:
        _reader, writer = await asyncio.open_unix_connection(str(sock_path))
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False

    try:
        await send_cmd(writer, "flush_transcript")
    except Exception:
        return False
    finally:
        try:
            writer.close()
        except Exception:  # nosec B110 - best-effort writer cleanup
            pass
    return True


def _wait_for_transcript(session_id: str, timeout: float = 3.0) -> None:
    """Wait up to *timeout* seconds for ``transcript.jsonl`` to appear.

    The orchestrator's agent flushes its transcript asynchronously during
    the turn loop.  Waiting briefly here avoids "Session not found" when
    the REPL starts.

    Best-effort: returns immediately on timeout rather than blocking the
    takeover flow.  Session.load() will return None and the REPL starts
    a fresh session (acceptable for agents that had no conversation).
    """
    import time as _time

    try:
        from clawcodex_ext.services.session_storage import resolve_sessions_dir

        transcript_path = resolve_sessions_dir() / session_id / "transcript.jsonl"
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if transcript_path.exists():
                return
            _time.sleep(0.2)
    except Exception:  # nosec B110 - best-effort
        pass  # Best-effort


def _ensure_session_stub(session_id: str) -> None:
    """Write a minimal ``session.json`` for session_id if none exists.

    Used as fallback when ``transcript.jsonl`` hasn't been flushed yet.
    The stub includes a single chat message so ``Conversation.add_message()``
    does not crash with ``IndexError`` (``pop from empty list``).
    Best-effort: failures are silently swallowed.

    IMPORTANT: if ``transcript.jsonl`` already exists, the session is real
    — this function returns without writing anything. A stub would shadow
    the transcript in ``Session.load()`` (Branch 2 trusts ``session.json``
    unconditionally), hiding the actual conversation history from the REPL.
    """
    import json as _json
    import time as _time

    try:
        from clawcodex_ext.services.session_storage import resolve_sessions_dir

        session_dir = resolve_sessions_dir() / session_id
        session_file = session_dir / "session.json"
        if session_file.exists():
            return
        # If the transcript already exists, the session is real — don't
        # shadow it with a stub. Session.load() will read the transcript
        # directly (either via _load_from_enhanced_transcript or the
        # metadata+transcript fallback). The stub is only for agents that
        # haven't written any session data yet.
        transcript_path = session_dir / "transcript.jsonl"
        if transcript_path.exists():
            return
        session_dir.mkdir(parents=True, exist_ok=True)
        now = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        stub = {
            "session_id": session_id,
            "provider": "",
            "model": "",
            "conversation": {
                "messages": [{"role": "chat", "content": "Takeover session."}],
                "max_history": 100,
            },
            "created_at": now,
            "updated_at": now,
        }
        session_file.write_text(_json.dumps(stub, indent=2), encoding="utf-8")
    except Exception:  # nosec B110 - best-effort, see docstring
        pass
