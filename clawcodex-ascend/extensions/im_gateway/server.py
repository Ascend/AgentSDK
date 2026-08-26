#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Gateway daemon lifecycle: PID/lock, stale-socket cleanup, health.

The daemon process runs :func:`serve` under ``asyncio.run``: it creates
a :class:`MessageGateway`, opens a POSIX UDS listener (``UdsPipeServer``
as the line-protocol base; ``GatewayIpcProtocol`` semantics land in
P2/P3), writes a PID file + health file, installs SIGTERM/SIGINT
handlers for graceful shutdown, and serves until stopped.

``clawcodex-dev gateway server start|stop|status|restart`` (in
``gateway_cmd/commands.py``) calls :class:`GatewayDaemon` which spawns
``python -m extensions.im_gateway.server serve`` as a detached
subprocess. ``status`` reads the PID + health file; ``stop`` sends
SIGTERM. v1 acceptance is POSIX/WSL/Git Bash only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clawcodex_ext.services.channels.feishu_settings import FeishuAppSettings
from clawcodex_ext.services.channels.models import ChannelType
from clawcodex_ext.services.im_gateway.config import (
    DEFAULT_STATE_DIR as _DEFAULT_STATE_DIR,
    load_config,
    migrate_legacy_state_dir,
)
from clawcodex_ext.services.im_gateway.gateway import MessageGateway
from clawcodex_ext.services.im_gateway.retention import run_retention_sweep
from clawcodex_ext.utils.file_lock import HAS_FLOCK, flock_exclusive

logger = logging.getLogger(__name__)
# lark_oapi.channel imports a very large generated dispatcher. On WSL cold
# starts this can take around two minutes before the websocket timeout begins.
FEISHU_SDK_STARTUP_BUFFER_SECONDS = 150.0
DEFAULT_STARTUP_HEALTH_WAIT_SECONDS = 45.0
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_READY_CHANNEL_STATUSES = frozenset({"connected", "logged_in", "websocket:connected"})
DEFAULT_STATE_DIR = _DEFAULT_STATE_DIR

# ``DEFAULT_STATE_DIR`` is re-exported from
# :mod:`clawcodex_ext.services.im_gateway.config` (imported above) so the
# historical ``from extensions.im_gateway.server import DEFAULT_STATE_DIR``
# import keeps working.


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    if os.name != "nt":
        path.chmod(_PRIVATE_DIRECTORY_MODE)


def _open_private_file(path: Path, flags: int) -> int:
    _ensure_private_directory(path.parent)
    fd = os.open(path, flags | os.O_CREAT, _PRIVATE_FILE_MODE)
    if os.name != "nt":
        os.fchmod(fd, _PRIVATE_FILE_MODE)
    return fd


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Keep the active log and every newly rotated successor private."""

    def _open(self):
        fd = _open_private_file(Path(self.baseFilename), os.O_WRONLY | os.O_APPEND)
        return os.fdopen(fd, self.mode, encoding=self.encoding, errors=self.errors)


@dataclass
class DaemonPaths:
    state_dir: Path
    pid_file: Path
    lock_file: Path
    sock_file: Path
    health_file: Path
    log_file: Path

    @classmethod
    def for_state_dir(cls, state_dir: str | Path | None = None) -> DaemonPaths:
        if state_dir is None:
            # Move a pre-rename ~/.clawcodex/im-gateway install forward the
            # first time the default path is resolved.
            base = migrate_legacy_state_dir()
        else:
            base = Path(state_dir).expanduser()
        _ensure_private_directory(base)
        return cls(
            state_dir=base,
            pid_file=base / "gateway.pid",
            lock_file=base / "gateway.lock",
            sock_file=base / "gateway.sock",
            health_file=base / "health.json",
            log_file=base / "gateway.log",
        )


# -- process helpers -----------------------------------------------------


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def read_pid(paths: DaemonPaths) -> int | None:
    if not paths.pid_file.exists():
        return None
    try:
        return int(paths.pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def write_pid(paths: DaemonPaths, pid: int) -> None:
    fd = _open_private_file(paths.pid_file, os.O_WRONLY | os.O_TRUNC)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{pid}\n")


def cleanup_stale(paths: DaemonPaths) -> bool:
    """Remove stale PID + socket if the recorded PID is dead.

    Returns True if any stale artifact was removed.
    """
    removed = False
    pid = read_pid(paths)
    if pid is not None and not is_pid_alive(pid):
        with contextlib.suppress(FileNotFoundError):
            paths.pid_file.unlink()
        removed = True
    if paths.sock_file.exists():
        # A socket without a live PID is stale.
        if pid is None or not is_pid_alive(pid):
            with contextlib.suppress(FileNotFoundError):
                paths.sock_file.unlink()
            removed = True
    return removed


def acquire_lock(paths: DaemonPaths) -> int | None:
    """Try to acquire the single-instance lock. Returns fd or None."""
    if not HAS_FLOCK:
        return None
    fd = _open_private_file(paths.lock_file, os.O_RDWR)
    try:
        flock_exclusive(fd, non_blocking=True)
        return fd
    except OSError:
        os.close(fd)
        return None


def write_health(paths: DaemonPaths, **fields) -> None:
    """Publish a complete private health document with atomic replacement."""
    data = {
        "running": True,
        "pid": os.getpid(),
        "started_at": fields.get("started_at", time.time()),
        "channels": fields.get("channels", []),
        "channel_status": fields.get("channel_status", {}),
        "state_dir": str(paths.state_dir),
        "socket": str(paths.sock_file),
    }
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    _ensure_private_directory(paths.health_file.parent)
    fd, temp_name = tempfile.mkstemp(prefix=f".{paths.health_file.name}.", suffix=".tmp", dir=paths.health_file.parent)
    temp_path: Path | None = Path(temp_name)
    try:
        if os.name != "nt":
            os.fchmod(fd, _PRIVATE_FILE_MODE)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        assert temp_path is not None
        os.replace(temp_path, paths.health_file)
        temp_path = None
        _fsync_directory(paths.health_file.parent)
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()


def _fsync_directory(path: Path) -> None:
    """Durably publish a replacement directory entry on POSIX."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _valid_health_data(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    started_at = data.get("started_at")
    if started_at is not None and (isinstance(started_at, bool) or not isinstance(started_at, (int, float))):
        return False
    channels = data.get("channels")
    if channels is not None and (not isinstance(channels, list) or not all(isinstance(item, str) for item in channels)):
        return False
    channel_status = data.get("channel_status")
    if channel_status is not None and (
        not isinstance(channel_status, dict)
        or not all(isinstance(channel, str) and isinstance(status, str) for channel, status in channel_status.items())
    ):
        return False
    return True


def read_health(paths: DaemonPaths) -> dict[str, Any] | None:
    if not paths.health_file.exists():
        return None
    try:
        data = json.loads(paths.health_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if _valid_health_data(data) else None


def startup_health_wait_seconds(paths: DaemonPaths) -> float:
    """CLI wait budget for daemon health after spawning the child process."""
    try:
        config = load_config(paths.state_dir / "channels.yaml")
    except Exception:  # noqa: BLE001
        return DEFAULT_STARTUP_HEALTH_WAIT_SECONDS
    timeout = 0.0
    for channel in config.channels:
        if not channel.enabled or channel.type is not ChannelType.FEISHU:
            continue
        extra = channel.extra or {}
        mode = str(extra.get("connection_mode") or "websocket").lower()
        if mode != "websocket":
            continue
        settings = FeishuAppSettings.from_config(channel)
        timeout = max(timeout, settings.startup_connect_timeout_seconds)
    if timeout <= 0.0:
        return DEFAULT_STARTUP_HEALTH_WAIT_SECONDS
    return timeout + FEISHU_SDK_STARTUP_BUFFER_SECONDS


def _channel_status_ready(status: object) -> bool:
    return str(status) in _READY_CHANNEL_STATUSES


def _channel_status_retrying(status: object) -> bool:
    return str(status) == "websocket:retrying"


# -- daemon process ------------------------------------------------------


def _resolve_log_level(verbose: bool, env: str | None) -> int:
    """Pick the daemon log level.

    Priority: ``CLAWCODEX_GATEWAY_LOG_LEVEL`` env > ``CLAWCODEX_DEBUG=1``
    (DEBUG) > ``--verbose`` flag (DEBUG) > default (WARNING). Pass
    ``--verbose`` or set ``CLAWCODEX_DEBUG=1`` during diagnosis; set
    ``CLAWCODEX_GATEWAY_LOG_LEVEL`` to pin a specific level.
    """
    if env:
        env = env.strip().upper()
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        if env in levels:
            return levels[env]
    if os.environ.get("CLAWCODEX_DEBUG", "").lower() in ("1", "true", "yes"):
        return logging.DEBUG
    if verbose:
        return logging.DEBUG
    return logging.WARNING


def _warm_feishu_sdk() -> None:
    """Trigger the slow ``import lark_oapi.channel`` in a worker thread.

    ``lark_oapi.channel`` pulls in the generated event dispatcher (every API
    processor), a multi-second import on a cold cache. Running it here (before
    the adapter constructs its ``FeishuChannel``) overlaps it with daemon
    startup so the WS connect isn't serialized behind the import. No-op if the
    SDK isn't installed.
    """
    try:
        import lark_oapi.channel  # noqa: F401 — import side effect is the point
    except Exception as exc:  # noqa: BLE001
        # Pre-warm is best-effort; the connect loop will surface real errors.
        logger.debug(
            "Feishu SDK pre-warm skipped: exception_type=%s",
            type(exc).__name__,
        )


async def serve(paths: DaemonPaths, *, log_level: int = logging.WARNING) -> int:
    """Run the gateway daemon (called from the spawned subprocess)."""
    if not HAS_FLOCK:
        print(
            "error: gateway daemon requires POSIX flock support (POSIX/WSL/Git Bash only)",
            file=sys.stderr,
        )
        return 1
    # Configure logging FIRST with a RotatingFileHandler (10 MiB per file,
    # keep up to 3 backups so the log never exceeds ~40 MiB). The subprocess
    # stderr is still redirected to the same log file by GatewayDaemon.start,
    # capturing any unhandled crash dumps that bypass the logging framework.
    _ensure_private_directory(paths.log_file.parent)
    root = logging.getLogger()
    root.setLevel(log_level)
    handler = _PrivateRotatingFileHandler(
        str(paths.log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MiB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(log_level)
    # Replace any existing handlers so the daemon subprocess owns its logging.
    root.handlers.clear()
    root.addHandler(handler)
    # Channel + gateway internals follow the resolved level; keep noisy libs
    # at WARNING so urllib/asyncio don't flood the log.
    logging.getLogger("clawcodex_ext.services.channels").setLevel(log_level)
    logging.getLogger("clawcodex_ext.services.im_gateway").setLevel(log_level)
    # The lark_oapi SDK forces its own logger to INFO in Client.__init__,
    # which floods the log with WS connect/ping INFO lines even when the user
    # didn't pass --verbose. Pin it to the resolved level so non-verbose runs
    # stay quiet (WARNING+), and --verbose still gets SDK DEBUG detail.
    logging.getLogger("Lark").setLevel(log_level)
    logging.getLogger("lark_oapi").setLevel(log_level)
    logging.getLogger("websockets").setLevel(log_level)

    fd = acquire_lock(paths)
    if fd is None:
        print("error: another gateway daemon holds the lock", file=sys.stderr)
        return 1
    cleanup_stale(paths)

    # Pre-warm the Feishu SDK import in a background thread. ``import
    # lark_oapi.api.*`` is a ~75s synchronous import of a large auto-generated
    # package; doing it lazily inside the adapter's connect loop delays WS
    # connect by the same amount. Kicking it off here (before gateway.start)
    # overlaps the import with the rest of daemon startup so the channel comes
    # up as soon as the import + WS handshake finish.
    import threading

    from clawcodex_ext.services.channels.feishu_sdk import feishu_dependencies_available

    if feishu_dependencies_available():
        threading.Thread(target=_warm_feishu_sdk, daemon=True, name="feishu-sdk-warm").start()

    # Write the PID file BEFORE starting channel adapters. A slow / hanging /
    # crashing adapter start (e.g. the Feishu WS connect loop) used to block
    # `await gateway.start()` below, so write_pid was never reached and the
    # daemon became invisible to `stop()`/`restart()` — leaving an orphaned
    # process holding the flock. Recording the PID up front guarantees the
    # daemon is always stoppable even when an adapter start misbehaves.
    started_at = time.time()
    write_pid(paths, os.getpid())

    try:
        config = load_config(paths.state_dir / "channels.yaml")
        # Force the gateway's reliability store to live under the daemon's
        # state_dir (the YAML default points at ~/.clawcodex/gateway).
        config.state_dir = str(paths.state_dir)
        gateway = MessageGateway(config)
        await gateway.start()
        # Adapter.start() performs the blocking initial connection attempt.
        # Once MessageGateway.start() returns, collect a single status snapshot:
        # connected channels are ready, retrying channels are degraded but have
        # already been queued for background reconnect.
        channel_status = await gateway.wait_channels_ready(timeout=0.0)
        for cid, status in channel_status.items():
            if _channel_status_ready(status):
                logger.info("gateway channel ready: %s -> %s", cid, status)
            elif _channel_status_retrying(status):
                logger.warning(
                    "gateway channel degraded after initial connect failure: %s -> %s "
                    "(retrying in background; see log)",
                    cid,
                    status,
                )
            else:
                logger.warning(
                    "gateway channel NOT ready after startup window: %s -> %s "
                    "(degraded; see log; messages may be dropped until it connects)",
                    cid,
                    status,
                )
        # Register the unbound-origin handler so authorized messages get explicit
        # REPL/orchestrator connection guidance instead of being silently dropped.
        from clawcodex_ext.services.im_gateway.stub_agent import make_stub_handler

        gateway.set_handler(make_stub_handler(gateway.outbound))
        logger.info("gateway inbound handler registered: unbound guidance handler")

        # Open the GatewayIpcProtocol UDS listener (register/heartbeat/deliver/ack
        # + control reload/status). P2/P3 frames are handled by GatewayIpcServer.
        from clawcodex_ext.services.im_gateway.ipc_server import GatewayIpcServer

        server = GatewayIpcServer(paths.sock_file, gateway)
        await server.start()

        # Wire the opt-in push path: when an origin is bound to a REPL/orchestrator
        # peer, the dispatcher pushes inbound messages over IPC instead of the
        # stub handler. The push callback delegates to the IPC server.
        async def _push_to_opt_in(message) -> bool:
            return await server.push_deliver(
                origin=message.origin,
                delivery_id=message.message_id,
                text=message.text,
                semantic=message.semantic.value if message.semantic else None,
                context_token=message.context_token,
            )

        gateway.set_push_handler(_push_to_opt_in)
        logger.info("gateway opt-in push handler registered")

        write_health(
            paths,
            started_at=started_at,
            channels=gateway.registry.names(),
            channel_status=channel_status,
        )
    except BaseException:
        # Adapter/IPC startup failed. Remove the PID we wrote up front so the
        # next `start`/`restart` doesn't see a stale PID pointing at a dead
        # process, then release the lock and re-raise so the parent reports the
        # early exit.
        with contextlib.suppress(FileNotFoundError):
            paths.pid_file.unlink()
        os.close(fd)
        raise

    # Start the persistent-file retention loop (every 24 hours by default).
    retention_task = asyncio.create_task(
        _retention_loop(
            gateway.store,
            config.reliability.retention_cron_interval_seconds,
            config.reliability,
        )
    )
    logger.info(
        "gateway retention loop started: interval=%ss",
        config.reliability.retention_cron_interval_seconds,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _schedule_shutdown() -> None:
        asyncio.create_task(_shutdown(server, gateway, paths, stop_event, retention_task))

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _schedule_shutdown)

    await stop_event.wait()
    os.close(fd)
    return 0


async def _retention_loop(store, interval: int, reliability) -> None:
    """Run retention every ``interval`` seconds and continue after failures."""
    while True:
        await asyncio.sleep(interval)
        try:
            run_retention_sweep(store, reliability)
        except Exception:  # noqa: BLE001
            logger.exception("im_gateway retention loop iteration failed")


async def _shutdown(
    server,
    gateway,
    paths: DaemonPaths,
    stop_event: asyncio.Event,
    retention_task: asyncio.Task | None = None,
) -> None:
    if retention_task is not None and not retention_task.done():
        retention_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await retention_task
    await server.close()
    await gateway.stop()
    with contextlib.suppress(FileNotFoundError):
        paths.pid_file.unlink()
    with contextlib.suppress(FileNotFoundError):
        paths.health_file.unlink()
    with contextlib.suppress(FileNotFoundError):
        paths.sock_file.unlink()
    stop_event.set()


# -- CLI-facing controller ----------------------------------------------


class GatewayDaemon:
    """Controls the gateway daemon subprocess from the CLI."""

    def __init__(self, paths: DaemonPaths | None = None) -> None:
        self.paths = paths or DaemonPaths.for_state_dir()

    def status(self) -> int:
        pid = read_pid(self.paths)
        if pid is None or not is_pid_alive(pid):
            print("Gateway daemon: NOT RUNNING")
            if pid is not None:
                print(f"  (stale PID {pid}; cleaned up)")
                cleanup_stale(self.paths)
            return 0
        health = read_health(self.paths) or {}
        uptime_s = int(time.time() - health.get("started_at", time.time()))
        print("Gateway daemon: RUNNING")
        print(f"  PID            : {pid}")
        print(f"  Uptime         : {uptime_s}s")
        print(f"  Socket         : {self.paths.sock_file}")
        print(f"  Log            : {self.paths.log_file}")
        print(f"  Channels       : {', '.join(health.get('channels') or []) or '(none)'}")
        print(f"  State dir      : {self.paths.state_dir}")
        return 0

    def start(self, *, verbose: bool = False) -> int:
        pid = read_pid(self.paths)
        if pid is not None and is_pid_alive(pid):
            print(f"Gateway daemon already running (PID {pid}).")
            return self.status()
        cleanup_stale(self.paths)
        # Attach the log file as the subprocess's stdout+stderr so any
        # unhandled crash dumps that bypass the logging framework are captured.
        # RotatingFileHandler inside serve() handles normal log rotation.
        log_fd = _open_private_file(self.paths.log_file, os.O_WRONLY | os.O_APPEND)
        log_fh = os.fdopen(log_fd, "a", encoding="utf-8")
        serve_args = [
            sys.executable,
            "-m",
            "extensions.im_gateway.server",
            "serve",
            "--state-dir",
            str(self.paths.state_dir),
        ]
        if verbose:
            serve_args.append("--verbose")
        try:
            # This detached daemon must outlive the short-lived ``start`` command.
            # pylint: disable=consider-using-with
            proc = subprocess.Popen(
                serve_args,
                stdout=log_fh,
                stderr=log_fh,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            log_fh.close()
        # Wait for the daemon to write its PID file (best-effort, bounded).
        # Then wait for the health file, which is written only after
        # ``serve()`` has waited for inbound adapters to connect — so a
        # "started" daemon has actually confirmed channel connectivity (or
        # timed out and recorded a degraded status). Total budget must exceed
        # the serve() channel-ready window (15s) plus startup overhead.
        pid_deadline = time.time() + 10.0
        new_pid: int | None = None
        while time.time() < pid_deadline:
            new_pid = read_pid(self.paths)
            if new_pid is not None and is_pid_alive(new_pid):
                break
            if proc.poll() is not None:
                print(
                    f"error: gateway daemon exited early (code {proc.returncode}); see {self.paths.log_file}",
                    file=sys.stderr,
                )
                return 1
            time.sleep(0.1)
        else:
            print("error: gateway daemon did not write PID in 10s", file=sys.stderr)
            return 1

        # Wait for the health file so we can report per-channel connectivity.
        # Feishu performs a blocking SDK import + initial websocket connection
        # inside adapter.start(), so this follows the configured startup timeout
        # plus the measured cold-import/bootstrap buffer.
        health_deadline = time.time() + startup_health_wait_seconds(self.paths)
        health = None
        while time.time() < health_deadline:
            if proc.poll() is not None:
                print(
                    f"error: gateway daemon exited early (code {proc.returncode}); see {self.paths.log_file}",
                    file=sys.stderr,
                )
                return 1
            health = read_health(self.paths)
            if health is not None:
                break
            time.sleep(0.3)

        print(f"Gateway daemon started · pid {new_pid}")
        channel_status = (health or {}).get("channel_status") or {}
        if channel_status:
            for cid, status in channel_status.items():
                if _channel_status_ready(status):
                    print(f"  channel {cid}: {status}")
                elif _channel_status_retrying(status):
                    print(
                        f"  channel {cid}: {status} — initial connect failed; "
                        f"retrying in background; see {self.paths.log_file}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  channel {cid}: NOT connected ({status}) — see "
                        f"{self.paths.log_file}; messages may be dropped",
                        file=sys.stderr,
                    )
        elif health is None:
            print(
                f"  warning: daemon still starting (no health yet); see {self.paths.log_file}",
                file=sys.stderr,
            )
        return 0

    def stop(self, *, timeout: float = 5.0) -> int:
        pid = read_pid(self.paths)
        if pid is None or not is_pid_alive(pid):
            print("Gateway daemon: already stopped")
            cleanup_stale(self.paths)
            return 0
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            cleanup_stale(self.paths)
            return 0
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not is_pid_alive(pid):
                break
            time.sleep(0.1)
        if is_pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
        cleanup_stale(self.paths)
        print("Gateway daemon stopped.")
        return 0

    def restart(self, *, verbose: bool = False) -> int:
        rc_stop = self.stop()
        if rc_stop != 0:
            return rc_stop
        return self.start(verbose=verbose)


# -- entry point ---------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="extensions.im_gateway.server")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve_p = sub.add_parser("serve", help="run the daemon (foreground)")
    serve_p.add_argument("--state-dir", default=None)
    serve_p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable DEBUG-level IM logging (default: WARNING; use --verbose or set "
        "CLAWCODEX_GATEWAY_LOG_LEVEL=INFO|DEBUG)",
    )
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        paths = DaemonPaths.for_state_dir(args.state_dir)
        log_level = _resolve_log_level(args.verbose, os.environ.get("CLAWCODEX_GATEWAY_LOG_LEVEL"))
        try:
            return asyncio.run(serve(paths, log_level=log_level))
        except KeyboardInterrupt:
            return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
