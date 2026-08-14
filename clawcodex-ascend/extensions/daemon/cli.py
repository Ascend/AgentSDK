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
"""``clawcodex-dev daemon`` CLI — supervisor lifecycle subcommands.

Verbs
-----
* ``start``  — fork the supervisor into the background and return.
* ``stop``   — ask the supervisor (via signal) to shut down.
* ``status`` — print a one-line summary of the supervisor + workers.
* ``ps``     — alias for ``status``.
* ``bg``     — same as ``start`` (alias kept for parity with CCB).
* ``attach`` — print a notice explaining the MVP limitation.
* ``logs``   — print the supervisor log file (or a stub if no logs yet).
* ``kill``   — force ``SIGKILL`` the supervisor (skip the graceful path).

Decoupling
----------
The verbs are registered as a single ``"daemon"`` subcommand on the
downstream CLI registry, NOT modified into ``src/cli.py``. The
:func:`register_daemon_subcommand` helper below installs the entry —
see ``clawcodex_ext.cli.subcommand_registry``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

from extensions.daemon.config import (
    DEFAULT_DAEMON_NAME,
    DEFAULT_SPAWN_MODE,
    DaemonConfig,
)
from extensions.daemon.errors import (
    DaemonAlreadyRunningError,
    DaemonNotRunningError,
)
from extensions.daemon.state import (
    DaemonState,
    DaemonStatus,
    get_state_dir,
    is_process_alive,
    query_daemon_status,
)
from extensions.daemon.supervisor import Supervisor

logger = logging.getLogger(__name__)

EXIT_FORK_FAILED = 78
STARTUP_POLL_ATTEMPTS = 40
STARTUP_POLL_INTERVAL_S = 0.1


# ---------------------------------------------------------------------------
# Argparse builders
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``daemon`` parser."""
    parser = argparse.ArgumentParser(
        prog="clawcodex daemon",
        description=(
            "Long-running daemon supervisor for ClawCodex workers. "
            "Use 'start' to fork the supervisor, 'stop' to shut it "
            "down, 'status' to inspect it."
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=(f"Override the daemon state directory (default: {get_state_dir()}). Useful for tests."),
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    p_start = sub.add_parser("start", help="Start the supervisor")
    p_start.add_argument("--name", default=DEFAULT_DAEMON_NAME)
    p_start.add_argument("--dir", type=Path, default=None)
    p_start.add_argument(
        "--workers",
        default="remoteControl",
        help="Comma-separated worker kinds (default: remoteControl)",
    )
    p_start.add_argument(
        "--spawn-mode",
        default=DEFAULT_SPAWN_MODE,
        choices=["single-session", "worktree", "same-dir"],
    )
    p_start.add_argument("--capacity", type=int, default=4)
    p_start.add_argument(
        "--permission-mode",
        default=None,
        choices=["bypassPermissions", "dontAsk", "plan"],
    )
    p_start.add_argument(
        "--sandbox",
        action="store_true",
        help="Run workers with sandbox enabled.",
    )
    p_start.add_argument(
        "--foreground",
        action="store_true",
        help=(
            "Don't fork — run the supervisor in the current process. "
            "Useful for systemd-style supervisors and E2E tests."
        ),
    )

    p_stop = sub.add_parser("stop", help="Stop the supervisor")
    p_stop.add_argument("--name", default=DEFAULT_DAEMON_NAME)
    p_stop.add_argument(
        "--timeout-ms",
        type=int,
        default=30_000,
        help="Max wait for graceful exit before escalating to SIGKILL.",
    )
    p_stop.add_argument(
        "--force",
        action="store_true",
        help="Send SIGKILL immediately (skip graceful shutdown).",
    )

    p_status = sub.add_parser("status", help="Print daemon + worker status")
    p_status.add_argument("--name", default=DEFAULT_DAEMON_NAME)
    p_status.add_argument("--json", action="store_true")

    p_ps = sub.add_parser("ps", help="Alias for 'status'")
    p_ps.add_argument("--name", default=DEFAULT_DAEMON_NAME)
    p_ps.add_argument("--json", action="store_true")

    p_bg = sub.add_parser("bg", help="Alias for 'start'")
    p_bg.add_argument("--name", default=DEFAULT_DAEMON_NAME)
    p_bg.add_argument("--dir", type=Path, default=None)
    p_bg.add_argument("--workers", default="remoteControl")
    p_bg.add_argument(
        "--spawn-mode",
        default=DEFAULT_SPAWN_MODE,
        choices=["single-session", "worktree", "same-dir"],
    )
    p_bg.add_argument("--capacity", type=int, default=4)
    p_bg.add_argument(
        "--permission-mode",
        default=None,
        choices=["bypassPermissions", "dontAsk", "plan"],
    )
    p_bg.add_argument("--sandbox", action="store_true")
    p_bg.add_argument(
        "--foreground",
        action="store_true",
        help="Don't fork — run the supervisor in the current process.",
    )

    p_attach = sub.add_parser("attach", help="Attach to the supervisor log (MVP: prints a notice)")
    p_attach.add_argument("--name", default=DEFAULT_DAEMON_NAME)

    p_logs = sub.add_parser("logs", help="Print recent supervisor log lines")
    p_logs.add_argument("--name", default=DEFAULT_DAEMON_NAME)
    p_logs.add_argument("--tail", type=int, default=200)

    p_kill = sub.add_parser("kill", help="Force-kill the supervisor")
    p_kill.add_argument("--name", default=DEFAULT_DAEMON_NAME)

    return parser


# ---------------------------------------------------------------------------
# Verb implementations
# ---------------------------------------------------------------------------


def _build_config_from_args(args: argparse.Namespace) -> DaemonConfig:
    workers = tuple(w.strip() for w in (args.workers or "").split(",") if w.strip())
    dir_ = args.dir if getattr(args, "dir", None) else Path.cwd()
    return DaemonConfig(
        name=args.name,
        dir=dir_,
        worker_kinds=workers or ("remoteControl",),
        spawn_mode=args.spawn_mode,
        capacity=args.capacity,
        permission_mode=getattr(args, "permission_mode", None),
        sandbox=getattr(args, "sandbox", False),
    )


def _format_status(
    status: DaemonStatus,
    state: DaemonState | None,
    name: str,
    as_json: bool = False,
) -> str:
    if as_json:
        return json.dumps(
            {
                "name": name,
                "status": status.value,
                "state": state.to_dict() if state else None,
            },
            indent=2,
        )

    lines: list[str] = ["=== Daemon Supervisor ==="]
    if status == DaemonStatus.STOPPED:
        lines.append(f"  Status:  stopped (no state file for {name!r})")
        lines.append("  Workers: -")
        return "\n".join(lines)
    if status == DaemonStatus.STALE:
        lines.append("  Status:  stale (PID no longer alive; state file cleaned)")
        lines.append("  Workers: -")
        return "\n".join(lines)
    assert state is not None
    uptime = _humanize_uptime(state.started_at)
    lines += [
        f"  Status:  {status.value}",
        f"  Name:    {state.name}",
        f"  PID:     {state.pid}",
        f"  CWD:     {state.cwd}",
        f"  Started: {state.started_at}  (up {uptime})",
        f"  Workers: {', '.join(state.worker_kinds)}",
    ]
    return "\n".join(lines)


def _humanize_uptime(started_at_iso: str) -> str:
    try:
        from datetime import datetime, timezone

        started = datetime.strptime(started_at_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - started
        total = int(delta.total_seconds())
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except (ValueError, TypeError):
        logger.warning("failed to parse started_at %r; falling back to 'unknown'", started_at_iso)
        return "unknown"


async def cmd_start(args: argparse.Namespace) -> int:
    state_dir = getattr(args, "state_dir", None)
    name = args.name
    status, _ = query_daemon_status(name, state_dir=state_dir)
    if status == DaemonStatus.RUNNING:
        print(
            f"daemon: {name!r} is already running; refusing to start.",
            file=sys.stderr,
        )
        return 1

    cfg = _build_config_from_args(args)
    cfg.validate()

    if args.foreground:
        # Stay attached — used for systemd-style supervisors and tests.
        sup = Supervisor(cfg, state_dir=state_dir)
        try:
            return await sup.run()
        except DaemonAlreadyRunningError as exc:
            print(f"daemon: {exc}", file=sys.stderr)
            return 1

    # Fork into the background. Re-exec this CLI with --foreground.
    _fork_supervisor(cfg, state_dir=state_dir)
    # Give the child a moment to write its state file.
    for _ in range(STARTUP_POLL_ATTEMPTS):
        await asyncio.sleep(STARTUP_POLL_INTERVAL_S)
        status, _ = query_daemon_status(name, state_dir=state_dir)
        if status == DaemonStatus.RUNNING:
            print(f"daemon: started {name!r}")
            return 0
    total_s = STARTUP_POLL_ATTEMPTS * STARTUP_POLL_INTERVAL_S
    print(
        f"daemon: forked but {name!r} did not report running within {total_s:.1f}s; check `daemon status`",
        file=sys.stderr,
    )
    return 1


def _fork_supervisor(cfg: DaemonConfig, *, state_dir: Path | None) -> None:
    """Re-exec the CLI with ``--foreground`` to detach."""
    argv = [
        sys.executable,
        "-m",
        "extensions.daemon.cli",
        "start",
        "--name",
        cfg.name,
        "--workers",
        ",".join(cfg.worker_kinds),
        "--spawn-mode",
        cfg.spawn_mode,
        "--capacity",
        str(cfg.capacity),
        "--foreground",
    ]
    if cfg.permission_mode:
        argv += ["--permission-mode", cfg.permission_mode]
    if cfg.sandbox:
        argv += ["--sandbox"]
    argv += ["--dir", str(cfg.dir)]
    if state_dir:
        argv += ["--state-dir", str(state_dir)]

    env = dict(os.environ)
    # Detach on POSIX via double-fork semantics. On Windows, ``close_fds``
    # plus CREATE_NEW_PROCESS_GROUP keeps the child alive past our exit.
    if hasattr(os, "fork"):
        pid = os.fork()
        if pid == 0:
            # First child: become session leader, fork again.
            try:
                os.setsid()
            except OSError:
                pass
            pid2 = os.fork()
            if pid2 == 0:
                # Grandchild: actually re-exec.
                os.execvpe(argv[0], argv, env)
                os._exit(EXIT_FORK_FAILED)
            os._exit(0)
        # Parent waits for first child to exit (it forks and exits immediately).
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    else:  # pragma: no cover — Windows
        # Best-effort detached spawn.
        import subprocess

        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        # Intentionally NOT using a ``with`` block here: Popen.__exit__
        # calls wait(), which would block this parent until the detached
        # supervisor exits. Stdio is DEVNULL and the object is released
        # when this function returns, so nothing is leaked long-term.
        subprocess.Popen(  # noqa: S603  # pylint: disable=consider-using-with
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
        )


async def cmd_stop(args: argparse.Namespace) -> int:
    state_dir = getattr(args, "state_dir", None)
    name = args.name
    status, state = query_daemon_status(name, state_dir=state_dir)
    if status != DaemonStatus.RUNNING:
        print(f"daemon: {name!r} is not running (status={status.value})")
        return 1
    assert state is not None
    pid = state.pid
    if args.force:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        print(f"daemon: SIGKILL sent to {name!r} (pid={pid})")
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"daemon: {name!r} already gone (pid={pid})")
        return 0
    print(f"daemon: SIGTERM sent to {name!r} (pid={pid})")

    deadline = time.monotonic() + (args.timeout_ms / 1000.0)
    while time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        if not is_process_alive(pid):
            print(f"daemon: {name!r} stopped cleanly")
            return 0
    print(
        f"daemon: {name!r} did not stop within {args.timeout_ms}ms; escalating to SIGKILL",
        file=sys.stderr,
    )
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state_dir = getattr(args, "state_dir", None)
    name = args.name
    status, state = query_daemon_status(name, state_dir=state_dir)
    print(_format_status(status, state, name, as_json=bool(args.json)))
    if status == DaemonStatus.RUNNING:
        return 0
    return 1


def cmd_attach(args: argparse.Namespace) -> int:
    print(
        "daemon: attach is not yet implemented in the Python port. "
        "Use `daemon logs` to inspect recent supervisor output, "
        "or run with --foreground for live output.",
        file=sys.stderr,
    )
    return 1


def cmd_logs(args: argparse.Namespace) -> int:
    state_dir = getattr(args, "state_dir", None) or get_state_dir()
    log_path = state_dir / f"{args.name}.log"
    if not log_path.exists():
        print(f"daemon: no log file at {log_path}", file=sys.stderr)
        return 1
    try:
        # ``tail -n`` is platform-specific; we read + slice in pure Python.
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-args.tail :]:
            print(line)
        return 0
    except OSError as exc:
        print(f"daemon: failed to read log: {exc}", file=sys.stderr)
        return 1


def cmd_kill(args: argparse.Namespace) -> int:
    args.force = True
    return asyncio.run(cmd_stop(args))


# ---------------------------------------------------------------------------
# Top-level dispatcher — invoked by ``subcommand_registry.register("daemon")``
# ---------------------------------------------------------------------------


def run_daemon(args: Sequence[str] | None = None) -> int:
    """Entry point for ``clawcodex-dev daemon <verb> ...``."""
    parser = build_parser()
    parsed = parser.parse_args(list(args if args is not None else sys.argv[1:]))
    verb = parsed.verb

    try:
        if verb in ("start", "bg"):
            return asyncio.run(cmd_start(parsed))
        if verb == "stop":
            return asyncio.run(cmd_stop(parsed))
        if verb in ("status", "ps"):
            return cmd_status(parsed)
        if verb == "attach":
            return cmd_attach(parsed)
        if verb == "logs":
            return cmd_logs(parsed)
        if verb == "kill":
            return cmd_kill(parsed)
    except DaemonNotRunningError as exc:
        print(f"daemon: {exc}", file=sys.stderr)
        return 1
    except DaemonAlreadyRunningError as exc:
        print(f"daemon: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(f"daemon: unknown verb {verb!r}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Subcommand registration
# ---------------------------------------------------------------------------


def register_daemon_subcommand() -> None:
    """Register the ``daemon`` CLI verb on the downstream registry."""
    from clawcodex_ext.cli.subcommand_registry import register

    @register("daemon")
    def _daemon_handler(args: list[str]) -> int:
        return run_daemon(args)


__all__ = [
    "build_parser",
    "cmd_attach",
    "cmd_kill",
    "cmd_logs",
    "cmd_start",
    "cmd_status",
    "cmd_stop",
    "register_daemon_subcommand",
    "run_daemon",
]


if __name__ == "__main__":
    raise SystemExit(run_daemon(sys.argv[1:]))
