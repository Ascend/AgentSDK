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

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _run_connect_gateway(args: argparse.Namespace) -> int:
    """Submit an IM gateway connect request to the running orchestrator daemon."""
    from .server import _find_metadata, _is_pid_alive  # delayed import to avoid circular dependency

    origin = _resolve_gateway_origin(args)

    meta_path, meta = _find_metadata(args)
    pid = meta.get("pid") if meta else None
    try:
        alive = bool(pid and _is_pid_alive(int(pid)))
    except (TypeError, ValueError):
        alive = False
    if not alive:
        print("Connection failed: orchestrator is not running", file=sys.stderr)
        return 1

    sock = _resolve_gateway_sock(args)
    if not _gateway_socket_available(sock):
        print("IM gateway daemon is not running", file=sys.stderr)
        print(f"  Requested socket: {sock}", file=sys.stderr)
        return 1

    workspace = Path(meta.get("workspace_root", os.getcwd())) if meta else Path.cwd()
    response_path = _gateway_control_response_path(workspace, "gateway_connect")
    control_path = _write_gateway_control(
        workspace,
        "gateway_connect",
        {
            "origin": origin,
            "sock": sock,
            "response_path": str(response_path),
        },
    )
    result = _wait_gateway_control_result(response_path)
    if result is not None:
        if result.get("ok"):
            print(f"gateway connected: origin={origin} sock={sock}")
            return 0
        print(f"gateway connect failed: {result.get('message') or 'unknown error'}", file=sys.stderr)
        return 1

    print("gateway connect request submitted; waiting for orchestrator next poll")
    print(f"  Control: {control_path}")
    print(f"  Running daemon PID: {pid}")
    if meta_path:
        print(f"  Metadata: {meta_path}")
    return 0


def _run_disconnect_gateway(args: argparse.Namespace) -> int:
    """Submit an IM gateway disconnect request to the running orchestrator daemon."""
    from .server import _find_metadata, _is_pid_alive  # delayed import to avoid circular dependency

    meta_path, meta = _find_metadata(args)
    pid = meta.get("pid") if meta else None
    try:
        alive = bool(pid and _is_pid_alive(int(pid)))
    except (TypeError, ValueError):
        alive = False
    if not alive:
        print("Connection failed: orchestrator is not running", file=sys.stderr)
        return 1

    workspace = Path(meta.get("workspace_root", os.getcwd())) if meta else Path.cwd()
    response_path = _gateway_control_response_path(workspace, "gateway_disconnect")
    control_path = _write_gateway_control(
        workspace,
        "gateway_disconnect",
        {
            "response_path": str(response_path),
        },
    )
    result = _wait_gateway_control_result(response_path)
    if result is not None:
        if result.get("ok"):
            print("gateway disconnected")
            return 0
        print(
            f"gateway disconnect failed: {result.get('message') or 'unknown error'}",
            file=sys.stderr,
        )
        return 1

    print("gateway disconnect request submitted; waiting for orchestrator next poll")
    print(f"  Control: {control_path}")
    print(f"  Running daemon PID: {pid}")
    if meta_path:
        print(f"  Metadata: {meta_path}")
    return 0


def _resolve_gateway_origin(args: argparse.Namespace) -> str:
    explicit = getattr(args, "gateway", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    origin = os.environ.get("CLAWCODEX_GATEWAY_ORIGIN") or os.environ.get("CLAWCODEX_IM_ORIGIN")
    if origin:
        return origin
    from clawcodex_ext.services.im_gateway.models import IM_DIRECT_ALL_ORIGIN

    return IM_DIRECT_ALL_ORIGIN


def _resolve_gateway_sock(args: argparse.Namespace) -> str:
    sock = (
        getattr(args, "gateway_sock", None)
        or os.environ.get("CLAWCODEX_GATEWAY_SOCK")
        or os.environ.get("CLAWCODEX_IM_GATEWAY_SOCK")
    )
    return str(sock or os.path.expanduser("~/.clawcodex/gateway/gateway.sock"))


def _gateway_socket_available(sock: str) -> bool:
    if not Path(sock).exists():
        return False
    try:
        asyncio.run(_probe_gateway_socket(sock))
        return True
    except Exception:  # noqa: BLE001
        return False


async def _probe_gateway_socket(sock: str) -> None:
    from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient

    client = GatewayIpcClient(sock, instance_id="orchestrator-control-probe")
    try:
        await client.connect()
    finally:
        await client.close()


def _gateway_control_response_path(workspace: Path, command: str) -> Path:
    control_dir = workspace / ".orchestrator_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    return control_dir / f"{command}_{uuid.uuid4().hex}.result.json"


def _write_gateway_control(workspace: Path, command: str, payload: dict) -> Path:
    control_dir = workspace / ".orchestrator_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    control_path = control_dir / f"{command}_{request_id}.control"
    body = f"{command}\n\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
    control_path.write_text(body, encoding="utf-8")
    return control_path


def _wait_gateway_control_result(response_path: Path, timeout_seconds: float = 0.2) -> dict | None:
    deadline = time.time() + max(0.0, timeout_seconds)
    while time.time() < deadline:
        if response_path.exists():
            try:
                return json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"ok": False, "message": "invalid gateway control result"}
        time.sleep(0.02)
    return None


def _run_start(args: argparse.Namespace) -> int:
    """Start the orchestrator daemon. Idempotent — already-running → show status."""
    from .server import _find_metadata, _is_pid_alive, _run_status  # delayed import to avoid circular dependency

    # Check if already running
    meta_path, meta = _find_metadata(args)
    if meta:
        pid = meta.get("pid")
        if pid and _is_pid_alive(pid):
            print(f"Orchestrator daemon is already running (PID {pid}).")
            print("Showing current status:")
            return _run_status(args)
        # Clean up stale metadata from dead PID before starting fresh
        if meta_path and meta_path.exists():
            meta_path.unlink(missing_ok=True)
            print(f"  Cleaned stale metadata from dead PID {pid or 'N/A'}")

    # Launch the orchestrator directly
    return _run_orchestrator(
        workflow_path=args.workflow,
        dashboard=getattr(args, "dashboard", False),
        port=getattr(args, "port", None),
        workflow_yaml_path=getattr(args, "workflow_yaml", None),
        gateway=getattr(args, "gateway", False),
        gateway_origin=getattr(args, "gateway_origin", None),
        gateway_sock=getattr(args, "gateway_sock", None),
    )


# ---------------------------------------------------------------------------
# orchestrator launch
# ---------------------------------------------------------------------------


def _mount_gateway_opt_in(
    subsystem,
    config,
    *,
    enabled: bool = False,
    origin: str | None = None,
    sock: str | None = None,
    feishu_adapter: Any | None = None,
):
    """Connect the orchestrator daemon to the IM gateway (opt-in via env).

    Enabled when ``enabled`` is true or ``CLAWCODEX_GATEWAY_ORIGIN`` is set.
    Without a specific origin, this binds all supported direct/private IM messages.
    Returns the
    :class:`OrchestratorGatewayClient` (for heartbeat scheduling) or None.

    Inbound IM messages for the origin are pushed over IPC and dispatched
    to existing orchestrator entry points; orchestrator events flow back to
    IM via OUTBOUND frames (``build_ipc_deliver``). No behavior change
    when the env var is unset.
    """
    origin = origin or os.environ.get("CLAWCODEX_GATEWAY_ORIGIN") or os.environ.get("CLAWCODEX_IM_ORIGIN")
    if not origin and enabled:
        from clawcodex_ext.services.im_gateway.models import IM_DIRECT_ALL_ORIGIN

        origin = IM_DIRECT_ALL_ORIGIN
    if not origin:
        return None
    sock = sock or os.environ.get("CLAWCODEX_GATEWAY_SOCK") or os.environ.get("CLAWCODEX_IM_GATEWAY_SOCK")
    if not sock:
        sock = os.path.expanduser("~/.clawcodex/gateway/gateway.sock")

    from extensions.orchestrator.im_gateway_client import (
        OrchestratorGatewayClient,
        OrchestratorHandlers,
    )

    def _orch():
        # subsystem._orchestrator is built during run(); resolve lazily.
        return getattr(subsystem, "_orchestrator", None)

    def _control_verb(verb, issue_id):
        o = _orch()
        if o is not None and hasattr(o, "_apply_control_command"):
            try:
                o._apply_control_command(verb, issue_id or "", "")
                logger.info("IM control_verb: %s issue=%s", verb, issue_id)
                return
            except Exception:  # noqa: BLE001
                logger.exception("IM control_verb failed")
        logger.warning("IM control_verb: orchestrator not ready (%s %s)", verb, issue_id)

    def _issue_inject(issue_id, hint):
        # Write to the workspace's .operator_hints.md via the orchestrator.
        _orch()
        ws_root = getattr(getattr(config, "workspace", None), "root", "")
        if ws_root:
            try:
                hints_file = Path(ws_root) / ".operator_hints.md"
                hints_file.parent.mkdir(parents=True, exist_ok=True)
                with hints_file.open("a", encoding="utf-8") as f:
                    f.write(f"\n{hint}\n")
                logger.info("IM issue_inject: issue=%s hint_len=%d", issue_id, len(hint))
                return
            except Exception:  # noqa: BLE001
                logger.exception("IM issue_inject failed")
        logger.warning("IM issue_inject: no workspace root")

    def _operator_hints(issue_id, text):
        _issue_inject(issue_id, text)

    def _queue_pending(issue_id, text):
        # Pending-message queue lives on RuntimeTaskRegistry; without an
        # active task for this issue we record the intent for the next run.
        logger.info("IM followup queued: issue=%s text_len=%d", issue_id, len(text))

    def _agent_intent(verb, issue_id):
        _control_verb(verb, issue_id)

    def _issue_cli(verb, issue_id, payload):
        logger.info("IM issue_cli: %s issue=%s", verb, issue_id)

    def _bridge_interrupt(issue_id, payload):
        _control_verb("stop", issue_id)

    handlers = OrchestratorHandlers(
        queue_pending_message=_queue_pending,
        control_verb=_control_verb,
        issue_inject=_issue_inject,
        operator_hints=_operator_hints,
        agent_intent=_agent_intent,
        issue_cli=_issue_cli,
        bridge_interrupt=_bridge_interrupt,
    )

    from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient

    session_id = f"orchestrator-{os.getpid()}"
    ipc = GatewayIpcClient(sock, instance_id=session_id)
    wrapper = OrchestratorGatewayClient(
        handlers, ipc_client=ipc, origin=origin, command_router=None, control_bridge=None
    )

    async def _connect_and_register() -> bool:
        """Connect to the gateway and register. Returns True on success.

        Never raises — the gateway and orchestrator are decoupled and
        either may be stopped independently. When the gateway is
        unavailable, returns False so the caller can retry on the next
        heartbeat without printing a traceback.
        """
        try:
            response = await ipc.reconnect_until_registered(
                session_id=session_id,
                origin=origin,
                capabilities=["outbound_text"],
            )
        except Exception:  # noqa: BLE001
            logger.debug("orchestrator IM reconnect raised (gateway unavailable)")
            return False
        if response is None or response.ack_layer != "accepted":
            logger.warning("orchestrator IM gateway unavailable; will retry on next heartbeat")
            return False
        flush_pending = getattr(wrapper, "_flush_pending_outbound", None)
        if callable(flush_pending):
            await flush_pending()
        logger.info("orchestrator IM opt-in connected: origin=%s sock=%s", origin[:32], sock)
        return True

    async def _heartbeat_loop():
        # Connect first, then heartbeat every 30s.  Startup can race the
        # gateway daemon, so keep trying instead of silently disabling IM.
        while not await _connect_and_register():
            await asyncio.sleep(30.0)
        missed_heartbeats = 0
        while True:
            try:
                response = await ipc.heartbeat()
                if response is None:
                    missed_heartbeats += 1
                    if missed_heartbeats < 2:
                        logger.warning(
                            "orchestrator IM heartbeat ACK timed out; "
                            "keeping the current registration until the next check"
                        )
                    else:
                        logger.warning("orchestrator IM heartbeat timed out twice; reconnecting")
                        await _connect_and_register()
                        missed_heartbeats = 0
                elif response.ack_layer != "accepted":
                    logger.warning("orchestrator IM heartbeat was not accepted; reconnecting")
                    await _connect_and_register()
                    missed_heartbeats = 0
                else:
                    missed_heartbeats = 0
                    maybe_flush = getattr(wrapper, "_flush_pending_outbound", None)
                    if callable(maybe_flush):
                        await maybe_flush()
            except Exception:  # noqa: BLE001
                logger.warning("orchestrator IM heartbeat failed; reconnecting")
                await _connect_and_register()
            await asyncio.sleep(30.0)

    wrapper._heartbeat_loop = _heartbeat_loop

    # Outbound: orchestrator events → WeChat via OUTBOUND frames.
    # _build_session_sink reads im_event_deliver at sink-build time inside
    # Orchestrator.run(); set it on the orchestrator instance right after
    # subsystem.run() constructs it, before it starts polling.
    def _sync_deliver(event, text):
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(wrapper.send_outbound(text))
        except RuntimeError:
            logger.warning("orchestrator IM: no loop; dropping event")

    _orig_run = subsystem.run

    async def _run_with_im():
        # subsystem.run constructs self._orchestrator then calls its run().
        # Patch run() so we set im_event_deliver on the orchestrator before
        # it starts polling / building session sinks.
        from extensions.orchestrator.orchestrator import Orchestrator as _Orch

        _orig_orch_run = _Orch.run

        async def _orch_run_patched(self, *a, **kw):
            self._im_gateway_wrapper = wrapper
            self._im_gateway_session_id = session_id
            self._im_gateway_heartbeat_task = getattr(wrapper, "_heartbeat_task", None)
            self.im_event_deliver = _sync_deliver
            self.im_event_channel = "wechat"
            # F-??? Feishu activity-sink wiring: when the caller passes
            # a FeishuAppChannelAdapter, propagate it through to the
            # orchestrator so :meth:`Orchestrator._build_session_sink`
            # can attach a :class:`FeishuActivitySink` per session. Stays
            # a no-op when ``feishu_adapter`` is None.
            if feishu_adapter is not None:
                self.im_channel_adapter = feishu_adapter
            if hasattr(self, "_emit_im_event"):
                from extensions.orchestrator.events import EventLevel

                self._emit_im_event(
                    "",
                    "orchestrator.started",
                    EventLevel.INFO,
                    "IM notifications enabled",
                )
            return await _orig_orch_run(self, *a, **kw)

        _Orch.run = _orch_run_patched
        try:
            await _orig_run()
        finally:
            _Orch.run = _orig_orch_run

    subsystem.run = _run_with_im

    return wrapper


def _run_orchestrator(
    workflow_path: str | None,
    dashboard: bool = False,
    port: int | None = None,
    workflow_yaml_path: str | None = None,
    gateway: bool = False,
    gateway_origin: str | None = None,
    gateway_sock: str | None = None,
) -> int:
    """Launch the orchestrator with a workflow file.

    This is the core launch entry point. Supports optional embedded
    dashboard status printing.
    """
    # Start the opt-in freeze-detection watchdog when the
    # env var is set. The orchestrator daemon is long-running and spawns
    # agent loops in subprocesses/threads; a hung worker can deadlock the
    # parent. Layer-1 dumps thread stacks so postmortem analysis can
    # attribute the hang even when the agent loop itself is unresponsive.
    try:
        from clawcodex_ext.diagnostics import FreezeDetector

        FreezeDetector.maybe_start_from_env()
    except Exception:  # nosec B110
        pass

    from extensions.orchestrator.tracker import TrackerConfigError, validate_tracker_config
    from extensions.orchestrator.workflow import WorkflowLoader, WorkflowParseError

    if not workflow_path:
        print("error: --workflow is required", file=sys.stderr)
        return 2

    try:
        config, prompt = WorkflowLoader.load(workflow_path)
    except WorkflowParseError as exc:
        print(f"error: failed to parse workflow: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"error: workflow file not found: {workflow_path}", file=sys.stderr)
        return 2

    # Load prompt into WorkflowStore so PromptBuilder can use it
    from ..workflow_store import get_workflow_store

    get_workflow_store().load(workflow_path)

    try:
        validate_tracker_config(config.tracker)
    except TrackerConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The orchestrator daemon is a long-running process whose INFO logs
    # (poll ticks, issue lifecycle, retries) are its primary diagnostic
    # surface. Use the centralized logging setup for consistent format,
    # timezone-aware timestamps, MDC context injection, and optional
    # JSON output for log aggregators.
    _ws_root = getattr(config.workspace, "root", "") or ""
    _json_log = str(Path(_ws_root) / ".reports" / "orchestrator.ndjson") if _ws_root else None
    from ..logging_setup import configure_orchestrator_logging

    configure_orchestrator_logging(
        level=logging.INFO,
        json_path=_json_log,
    )

    # Build repo slug for the startup banner
    _tracker_kind = getattr(config.tracker, "kind", "?")
    _owner = getattr(config.tracker, "owner", None) or ""
    _repo = getattr(config.tracker, "repo", None) or ""
    _repo_slug = f"{_owner}/{_repo}" if _owner and _repo else ""
    _pid = os.getpid()
    _agent = getattr(config, "agent", None)

    print(f"\u2713 orchestrator daemon started \u00b7 pid {_pid}", end="")
    if _tracker_kind and _tracker_kind != "?":
        print(f" \u00b7 tracker={_tracker_kind}", end="")
        if _repo_slug:
            print(f" \u00b7 repo={_repo_slug}", end="")
    print()
    if _agent is not None:
        print(
            f"\u2713 max_concurrent_agents={getattr(_agent, 'max_concurrent_agents', '?')}"
            f" \u00b7 permission_mode={getattr(_agent, 'permission_mode', '?')}"
        )

    from extensions.api.orchestration import OrchestrationSubsystem

    subsystem = OrchestrationSubsystem(config, workflow_yaml_path=workflow_yaml_path)

    # Wire the orchestrator into the process-wide dashboard store.
    # The provider reads subsystem._orchestrator which is constructed inside
    # subsystem.run(); until then the source returns an empty snapshot.
    try:
        from extensions.agent_dashboard import register_dashboard_source
        from extensions.agent_dashboard.sources.orchestrator_source import OrchestratorDashboardSource

        register_dashboard_source(
            OrchestratorDashboardSource(
                orchestrator_provider=lambda: getattr(subsystem, "_orchestrator", None),
            )
        )
    except Exception:
        logger.debug("Failed to register orchestrator dashboard source", exc_info=True)

    # Fix 2: write the real daemon PID to <workspace>/daemon.pid
    # so external tools (cron monitor, stop scripts) can locate the
    # running daemon.  The previous shell-wrapper pattern
    # ``nohup ... & disown; echo $! > pidfile`` captured the nohup
    # wrapper PID which sometimes did not match the python process
    # that ultimately ran the orchestrator (chain-exec races, signal
    # forwarding).  Writing the pidfile in-process via ``os.getpid()``
    # makes the value authoritative and removes the dependency on
    # the shell launcher's PID semantics.
    try:
        import atexit

        _ws_root = Path(getattr(config.workspace, "root", "") or "")
        if str(_ws_root):
            _pidfile = _ws_root / "daemon.pid"
            _pidfile.parent.mkdir(parents=True, exist_ok=True)
            _pidfile.write_text(f"{os.getpid()}\n", encoding="utf-8")

            def _cleanup_pidfile() -> None:
                try:
                    _pidfile.unlink(missing_ok=True)
                except Exception:  # nosec B110
                    pass

            atexit.register(_cleanup_pidfile)
    except Exception as exc:  # noqa: BLE001
        # Never block daemon start on pidfile failures (read-only
        # workspace, missing dir, etc.) — just warn and continue.
        print(
            f"warning: failed to write pidfile: {exc}",
            file=sys.stderr,
        )

    # Register signal handlers for graceful shutdown on SIGTERM/SIGINT.
    # Without these, a plain `kill <pid>` or Ctrl+C sends SIGTERM/SIGINT
    # which Python asyncio does not handle — the process dies immediately
    # without running any cleanup (shutdown(), _cancel_all_tasks(), atexit).
    # With signal handlers, the event loop catches the signal and calls
    # subsystem.shutdown(), which sets _shutdown_event so the polling loop
    # exits cleanly, then _cancel_all_tasks() cancels running issues.
    #
    # SIGKILL (-9) cannot be caught and will still cause abrupt death;
    # the pdeath_sig PR_SET_PDEATHSIG in subprocesses mitigates orphan
    # children for that case.
    #
    # IMPORTANT: signal handlers MUST be registered on the loop that
    # actually runs subsystem.run(). Using asyncio.get_event_loop() before
    # asyncio.run() grabs a stale/ghost loop (asyncio.run creates a new
    # one internally), so the handler would never fire. We register inside
    # the coroutine via get_running_loop() to bind to the real running loop.

    def _schedule_shutdown(sig_name: str) -> None:
        """Callback registered via loop.add_signal_handler."""
        logger.info("Received %s — scheduling graceful shutdown...", sig_name)
        # Schedule the async shutdown as a task; add_signal_handler
        # only accepts synchronous callables.
        asyncio.create_task(subsystem.shutdown())

    # IM gateway opt-in: when configured, register the orchestrator as the
    # opt-in target for that WeChat origin so inbound messages drive
    # orchestrator actions, and orchestrator events flow back to WeChat via
    # OUTBOUND IPC frames. No-op otherwise.
    im_client_wrapper = _mount_gateway_opt_in(
        subsystem,
        config,
        enabled=gateway,
        origin=gateway_origin,
        sock=gateway_sock,
    )

    async def _run() -> None:
        # Bind signal handlers to the loop that is actually running this
        # coroutine. asyncio.run() creates a fresh loop, so registration
        # must happen here, not outside asyncio.run().
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda sig_name=signal.Signals(sig).name: _schedule_shutdown(sig_name),
            )
        im_task = None
        if im_client_wrapper is not None:
            im_task = asyncio.create_task(im_client_wrapper._heartbeat_loop())
            im_client_wrapper._heartbeat_task = im_task
        try:
            await subsystem.run()
        except (asyncio.CancelledError, KeyboardInterrupt):
            await subsystem.shutdown()
            raise
        finally:
            if im_task is not None and not im_task.done():
                im_task.cancel()
                with __import__("contextlib").suppress(asyncio.CancelledError):
                    await im_task

    if dashboard:

        async def _run_with_dashboard() -> None:
            """Run orchestrator with a concurrent dashboard status loop."""
            dashboard_task = asyncio.create_task(_dashboard_loop(subsystem.status_dashboard, port))
            try:
                await _run()
            finally:
                dashboard_task.cancel()

        asyncio.run(_run_with_dashboard())
    else:
        asyncio.run(_run())

    return 0


async def _dashboard_loop(dashboard, port: int | None) -> None:
    """Periodic dashboard status print loop."""

    while True:
        await asyncio.sleep(5)
        try:
            state = dashboard.state()
            running_ids = list(state.get("running", {}).keys())
            print(
                f"[dashboard] running={len(running_ids)} "
                f"completed={state.get('completed_count', 0)} "
                f"failed={state.get('failed_count', 0)}",
                file=sys.stderr,
            )
        except Exception:  # nosec B110
            pass
