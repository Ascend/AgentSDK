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

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


from .agent_runner import AgentSession, RetryItem
from .events import EventLevel
from .issue import Issue
from .review_feedback import ReviewFollowup
from .tracker import (
    PullRequestFeedback,
    PullRequestFeedbackCapability,
    PullRequestRef,
    supports,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CONTINUATION_RETRY_DELAY_MS = 1_000
_FAILURE_RETRY_BASE_MS = 10_000


def _operator_failure_detail(exc: BaseException) -> str:
    """Return a concise failure detail suitable for IM and registry records."""

    raw = " ".join(str(exc).split())
    body_detail = _extract_error_message_from_body(raw)
    if body_detail:
        status_code = _extract_status_code(raw)
        if raw.startswith("request_failed") and status_code:
            return f"request_failed status={status_code}: {body_detail}"
        return body_detail
    return raw or exc.__class__.__name__


def _extract_status_code(text: str) -> str | None:
    for part in text.split():
        if part.startswith("status="):
            status = part.removeprefix("status=").strip()
            if status:
                return status
    return None


def _extract_error_message_from_body(text: str) -> str | None:
    marker = "body="
    marker_index = text.find(marker)
    if marker_index < 0:
        return None
    body = text[marker_index + len(marker) :].strip()
    if not body:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    return _extract_error_message(payload)


def _extract_error_message(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in (
            "error_message",
            "message",
            "error_description",
            "detail",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return " ".join(error.split())
        nested = _extract_error_message(error)
        if nested:
            return nested
        errors = payload.get("errors")
        if isinstance(errors, list):
            for item in errors:
                nested = _extract_error_message(item)
                if nested:
                    return nested
    return None


class OrchestratorOpsMixin:
    async def _handle_review_followup_control(self, issue_id: str, extra: str) -> None:
        """Handle a CLI-approved review_followup control command."""
        if not issue_id:
            return
        record = self._registry.get(issue_id)
        if record is None:
            logger.warning("review_followup control: issue %s not in registry", issue_id)
            return
        if issue_id in self._state.running:
            logger.info("review_followup control: issue %s already running, skipping", issue_id)
            return

        feedback_ids = (
            [fid.strip() for fid in extra.split(",") if fid.strip()] if extra else list(record.pending_feedback_ids)
        )
        if not feedback_ids:
            logger.info("review_followup control: no feedback IDs for issue %s", issue_id)
            return

        pull_request = PullRequestRef(
            number=record.pr_number,
            url=record.pr_url,
        )
        issue = Issue(
            id=record.issue_id,
            identifier=record.issue_identifier,
            title=record.issue_identifier,
            branch_name=record.branch_name,
        )
        feedback_items: list[PullRequestFeedback] = []
        if not supports(self.tracker, PullRequestFeedbackCapability):
            return
        try:
            all_feedback = await self.tracker.fetch_pull_request_feedback(
                pull_request=pull_request,
                issue_id=record.issue_id,
                include_ci_failures=self.workflow.review_feedback.include_ci_failures,
            )
            feedback_by_id = {item.id: item for item in all_feedback}
            for fid in feedback_ids:
                if fid in feedback_by_id:
                    feedback_items.append(feedback_by_id[fid])
        except Exception as exc:
            logger.error("review_followup control: failed to fetch feedback for issue %s: %s", issue_id, exc)
            return

        if not feedback_items:
            logger.info("review_followup control: no matching feedback found for issue %s", issue_id)
            return

        followup = ReviewFollowup(
            issue=issue,
            record=record,
            pull_request=pull_request,
            feedback=feedback_items,
            prompt="",
        )
        self._state.claimed.add(issue_id)
        await self._launch_review_followup(followup)
        logger.info(
            "review_followup control: launched follow-up for issue %s with %d feedback items",
            issue_id,
            len(feedback_items),
        )

    async def _schedule_retry(
        self,
        session: AgentSession,
        *,
        delay_base_ms: int | None = None,
    ) -> None:
        """Schedule a retry for a failed session.

        ``delay_base_ms`` overrides the base delay for the exponential backoff
        curve. When ``None`` the default ``_FAILURE_RETRY_BASE_MS`` is used
        (10s). The orchestrator passes ``workflow.agent.max_turns_retry_delay_ms``
        for ``max_turns_exceeded`` sessions so the longer wait default kicks in
        without forcing all retries to share it.
        """
        issue_id = session.issue.id or ""
        attempt = self._state.retry_attempts.get(issue_id, 0) + 1
        self._state.retry_attempts[issue_id] = attempt

        # Retry context: persist the just-failed run_id so the next
        # attempt's agent can Read() the previous transcript to understand
        # what was tried and where it failed.
        if session.run_id:
            record = self._registry.get(issue_id)
            if record is not None:
                if record.previous_run_ids is None:
                    record.previous_run_ids = []
                if session.run_id not in record.previous_run_ids:
                    record.previous_run_ids.append(session.run_id)
                    self._registry._save()

        max_attempts = self.workflow.agent.max_retry_attempts
        if max_attempts and attempt > max_attempts:
            logger.warning(
                "Retry limit reached issue_id=%s attempts=%d max=%d — giving up",
                issue_id,
                attempt,
                max_attempts,
            )
            self._state.claimed.discard(issue_id)
            self._registry.mark_abandoned(issue_id)
            await self._sync_tracker_issue_state(issue_id, "abandoned")
            return

        # Exponential backoff capped at max_retry_backoff_ms
        base_ms = delay_base_ms if delay_base_ms is not None else _FAILURE_RETRY_BASE_MS
        max_ms = self.workflow.agent.max_retry_backoff_ms
        delay_ms = min(base_ms * (1 << (attempt - 1)), max_ms)

        retry = RetryItem(
            issue_id=issue_id,
            attempt=attempt,
            delay_seconds=delay_ms / 1000.0,
            identifier=session.issue.identifier or "",
            error=f"agent failed: {session.status}",
        )
        self._state.retry_queue.append(retry)
        logger.info(
            "Scheduled retry issue_id=%s attempt=%s delay=%sms",
            issue_id,
            attempt,
            delay_ms,
        )
        self._emit_im_event(
            issue_id,
            "intent.retry",
            EventLevel.INFO,
            f"retry scheduled in {delay_ms}ms",
            {"attempt": attempt, "delay_ms": delay_ms},
        )

    def _broadcast_clarification_status(self) -> None:
        """Collect clarification status for all issues and push to the dashboard."""
        if self.status_dashboard is None:
            return
        from .status_dashboard import ClarificationEntry

        now = time.time()
        max_rounds = getattr(
            getattr(self.workflow, "clarifier", None),
            "max_rounds",
            2,
        )
        entries: list[ClarificationEntry] = []
        for issue_id, record in self._registry._records.items():
            status = record.clarification_status
            if status in ("awaiting_author", "awaiting_local", "manual_required", "resolved"):
                elapsed = now - (record.updated_at or now)
                entries.append(
                    ClarificationEntry(
                        issue_id=issue_id,
                        status=status or "",
                        open_questions=list(record.open_questions),
                        round_num=record.clarification_round,
                        max_rounds=max_rounds,
                        elapsed_seconds=elapsed,
                        author_login=record.author_login,
                    )
                )
        self.status_dashboard.on_clarification_update(entries)

    def _compute_workspace_focus_for_clarifier(self, issue: "Issue") -> list[dict]:
        """Compute workspace focus as clarification context enrichment.

        Only called when a follow-up branch exists. For new issues
        (no branch yet) returns [].
        """
        branch = getattr(issue, "branch_name", None) or getattr(issue, "linked_branch", None)
        if not branch:
            return []
        try:
            changed = self._git_changed_files(branch)
            if not changed:
                return []
            from clawcodex_ext.intent_forecast.focus import compute_workspace_focuses

            return compute_workspace_focuses(changed_files=changed, recent_messages=[])
        except Exception as exc:
            logger.warning("Workspace focus computation failed for issue %s: %s", issue.id, exc)
            return []

    async def _process_escalated_issues(self) -> None:
        """Check for clarification-exhausted issues and apply escalation policy.

        When a clarification item is marked EXHAUSTED, the escalation policy
        determines what happens next:
          - skip: mark as ABANDONED so orchestrator skips it on next poll
          - mark_failed: mark as FAILED
          - notify: mark as FAILED + send notification
        """
        sentinel_path = self._workspace_root / ".escalated_issues.json"
        if not sentinel_path.exists():
            return

        try:
            data = json.loads(sentinel_path.read_text())
        except Exception:
            return

        if not data:
            return

        # Collect IDs to remove from sentinel
        to_remove = []

        for issue_id in data:
            if issue_id in self._state.completed or issue_id in self._state.claimed:
                to_remove.append(issue_id)
                continue

            policy = self._clarification_resolver._config.escalation
            if policy == "mark_failed":
                self._registry.mark_failed(issue_id)
                await self._sync_tracker_issue_state(issue_id, "failed")
                self._state.completed.add(issue_id)
                self._emit_im_event(
                    issue_id,
                    "clarification.exhausted",
                    EventLevel.WARN,
                    "clarification exhausted",
                )
            elif policy == "notify":
                self._registry.mark_failed(issue_id)
                await self._sync_tracker_issue_state(issue_id, "failed")
                self._state.completed.add(issue_id)
                logger.warning("Escalation notify for issue %s", issue_id)
                self._emit_im_event(
                    issue_id,
                    "clarification.exhausted",
                    EventLevel.ERROR,
                    "clarification exhausted",
                )
            else:  # skip → mark as abandoned
                self._registry.mark_abandoned(issue_id)
                await self._sync_tracker_issue_state(issue_id, "abandoned")
                self._state.completed.add(issue_id)
                logger.info("Escalation skip for issue %s", issue_id)
                self._emit_im_event(
                    issue_id,
                    "clarification.exhausted",
                    EventLevel.WARN,
                    "clarification exhausted",
                )

            to_remove.append(issue_id)

        # Prune processed entries from sentinel
        if to_remove:
            for issue_id in to_remove:
                data.pop(issue_id, None)
            sentinel_path.write_text(json.dumps(data, indent=2))

    async def _process_retry_queue(self) -> None:
        """Process retry queue with exponential backoff.

        Retries are processed before new candidate issues so that
        previously-failed work gets priority.
        """
        now = time.time()
        ready: list[Any] = []
        remaining: list[Any] = []

        for retry in self._state.retry_queue:
            if now >= retry.scheduled_at + retry.delay_seconds:
                ready.append(retry)
            else:
                remaining.append(retry)

        self._state.retry_queue = remaining

        for retry in ready:
            # Skip if already running or completed
            if retry.issue_id in self._state.running or retry.issue_id in self._state.completed:
                logger.debug("Retry skipped issue_id=%s already running/completed", retry.issue_id)
                continue

            # Check concurrency slot
            if len(self._state.running) >= self._state.max_concurrent_agents:
                logger.debug("Retry deferred issue_id=%s no concurrency slots", retry.issue_id)
                remaining.append(retry)
                continue

            # Re-fetch issue state from tracker
            try:
                issues = await self.tracker.fetch_issue_states_by_ids([retry.issue_id])
                issue = issues.get(retry.issue_id)
                if issue is None:
                    logger.warning("Retry issue not found issue_id=%s", retry.issue_id)
                    continue
            except Exception as exc:
                logger.error("Failed to fetch retry issue %s: %s", retry.issue_id, exc)
                # Put back at end of queue with extended delay
                retry.delay_seconds = min(retry.delay_seconds * 2, self.workflow.agent.max_retry_backoff_ms / 1000.0)
                retry.scheduled_at = now
                remaining.append(retry)
                continue

            # Check if issue is still in active states
            active_states = [s.strip().lower() for s in (getattr(self.tracker, "active_states", None) or [])]
            if issue.state and issue.state.strip().lower() not in active_states:
                logger.info(
                    "Retry issue %s no longer active (state=%s), dropping",
                    retry.issue_id,
                    issue.state,
                )
                continue

            self._state.claimed.add(retry.issue_id)
            await self._launch_issue(issue)
            logger.info(
                "Retry launched issue_id=%s attempt=%s",
                retry.issue_id,
                retry.attempt,
            )

    async def _process_control_commands(self) -> None:
        """Process lifecycle control commands from CLI.

        Checks the control directory for pause/resume/stop/takeover commands
        written by `clawcodex orchestrator pause/resume/stop/takeover`.
        """

        control_dir = self._workspace_root / ".orchestrator_control"
        if not control_dir.exists():
            return

        try:
            for control_file in control_dir.iterdir():
                if not control_file.name.endswith(".control"):
                    continue
                parts = control_file.read_text(encoding="utf-8").strip().split("\n")
                if not parts:
                    continue
                cmd = parts[0].strip()
                issue_id = parts[1].strip() if len(parts) > 1 else ""
                extra = "\n".join(parts[2:]).strip() if len(parts) > 2 else ""

                try:
                    if cmd == "review_followup":
                        await self._handle_review_followup_control(issue_id, extra)
                    elif cmd == "rebase":
                        # Route CLI-written rebase control files to
                        # the built-in rebase path. Format::
                        #   rebase\n<id>\nforce=0|1\n<reason>
                        await self._handle_rebase_control(issue_id, extra)
                    elif cmd in {"gateway_connect", "gateway_disconnect"}:
                        await self._handle_gateway_control(cmd, extra)
                    elif cmd == "review_approve":
                        await self._handle_review_approve_control(issue_id, extra)
                    elif cmd == "review_retry":
                        await self._handle_review_retry_control(issue_id, extra)
                    elif cmd == "retry":
                        await self._handle_retry_control(issue_id, extra)
                    else:
                        self._apply_control_command(cmd, issue_id, extra)
                finally:
                    # Clean up control file after processing
                    try:
                        control_file.unlink()
                    except Exception:  # nosec B110
                        pass
        except Exception as exc:
            logger.warning("Failed to process control commands: %s", exc)

    async def _handle_gateway_control(self, cmd: str, extra: str) -> None:
        """Handle CLI-written IM gateway connect/disconnect control files."""
        payload: dict[str, Any] = {}
        if extra:
            try:
                payload = json.loads(extra)
            except json.JSONDecodeError as exc:
                logger.warning("gateway control: invalid payload: %s", exc)
                return
        response_path = payload.get("response_path")
        if cmd == "gateway_connect":
            result = await self._connect_gateway_runtime(
                origin=str(payload.get("origin") or ""),
                sock=str(payload.get("sock") or ""),
            )
        else:
            result = await self._disconnect_gateway_runtime()
        if response_path:
            self._write_gateway_control_result(Path(str(response_path)), result)

    def _write_gateway_control_result(self, path: Path, result: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.debug("gateway control: failed to write result %s", path, exc_info=True)

    async def _connect_gateway_runtime(self, *, origin: str, sock: str) -> dict[str, Any]:
        if not origin:
            return {"ok": False, "message": "gateway origin is required"}
        if not sock:
            return {"ok": False, "message": "gateway socket is required"}

        current = getattr(self, "_im_gateway_wrapper", None)
        current_ipc = getattr(current, "_ipc", None)
        current_origin = getattr(current, "_origin", None)
        current_sock = str(getattr(current_ipc, "socket_path", getattr(current_ipc, "sock", "")))
        if current is not None and current_origin == origin and current_sock == sock:
            return {"ok": True, "message": "already connected"}

        from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient
        from extensions.orchestrator.im_gateway_client import (
            OrchestratorGatewayClient,
            OrchestratorHandlers,
        )

        def _control_verb(verb, issue_id):
            self._apply_control_command(verb, issue_id or "", "")

        def _issue_inject(issue_id, hint):
            hints_file = self._workspace_root / ".operator_hints.md"
            hints_file.parent.mkdir(parents=True, exist_ok=True)
            with hints_file.open("a", encoding="utf-8") as f:
                f.write(f"\n{hint}\n")

        handlers = OrchestratorHandlers(
            queue_pending_message=lambda issue_id, text: logger.info(
                "IM followup queued: issue=%s text_len=%d", issue_id, len(text)
            ),
            control_verb=_control_verb,
            issue_inject=_issue_inject,
            operator_hints=_issue_inject,
            agent_intent=_control_verb,
            issue_cli=lambda verb, issue_id, payload: logger.info("IM issue_cli: %s issue=%s", verb, issue_id),
            bridge_interrupt=lambda issue_id, payload: _control_verb("stop", issue_id),
        )
        session_id = f"orchestrator-{os.getpid()}-{int(time.time() * 1000)}"
        ipc = GatewayIpcClient(sock, instance_id=session_id)
        wrapper = OrchestratorGatewayClient(
            handlers, ipc_client=ipc, origin=origin, command_router=None, control_bridge=None
        )
        try:
            await ipc.connect()
            response = await ipc.register(
                session_id=session_id,
                origin=origin,
                capabilities=["outbound_text"],
            )
            if response is None or response.ack_layer != "accepted":
                await ipc.close()
                return {"ok": False, "message": "gateway registration failed"}
        except FileNotFoundError:
            await ipc.close()
            return {"ok": False, "message": "IM gateway daemon is not running"}
        except Exception as exc:  # noqa: BLE001
            await ipc.close()
            logger.warning("gateway control connect failed", exc_info=True)
            return {"ok": False, "message": str(exc)}

        old_wrapper = getattr(self, "_im_gateway_wrapper", None)
        old_task = getattr(self, "_im_gateway_heartbeat_task", None)
        old_session_id = getattr(self, "_im_gateway_session_id", None)
        deliver = self._build_gateway_ipc_deliver(wrapper)
        self._im_gateway_wrapper = wrapper
        self._im_gateway_session_id = session_id
        self._im_gateway_heartbeat_task = asyncio.create_task(self._gateway_runtime_heartbeat_loop(wrapper, session_id))
        self.im_event_deliver = deliver
        self.im_event_channel = "wechat"
        self._attach_gateway_sink_to_existing_emitters(deliver)
        if callable(getattr(wrapper, "_flush_pending_outbound", None)):
            await wrapper._flush_pending_outbound()
        if old_task is not None and not old_task.done():
            old_task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await old_task
        if old_wrapper is not None and old_wrapper is not wrapper:
            await self._close_gateway_wrapper(old_wrapper, old_session_id)
        self._emit_im_event(
            "",
            "orchestrator.started",
            EventLevel.INFO,
            "IM notifications enabled",
        )
        return {"ok": True, "message": "connected"}

    def _build_gateway_ipc_deliver(self, wrapper) -> Any:
        loop = asyncio.get_running_loop()

        def _sync_deliver(event, text):
            loop.create_task(wrapper.send_outbound(text))

        return _sync_deliver

    def _attach_gateway_sink_to_existing_emitters(self, deliver) -> None:
        emitters = getattr(self, "_im_emitters", {}) or {}
        if not emitters:
            return
        from .channel_sink import ChannelProgressSink

        for emitter in list(emitters.values()):
            add_sink = getattr(emitter, "add_sink", None)
            if callable(add_sink):
                add_sink(ChannelProgressSink(deliver))

    async def _gateway_runtime_heartbeat_loop(self, wrapper, session_id: str) -> None:
        ipc = getattr(wrapper, "_ipc", None)
        if ipc is None:
            return
        while True:
            try:
                await ipc.heartbeat()
            except Exception:  # noqa: BLE001
                logger.debug("orchestrator IM runtime heartbeat failed", exc_info=True)
            await asyncio.sleep(30.0)

    async def _disconnect_gateway_runtime(self) -> dict[str, Any]:
        wrapper = getattr(self, "_im_gateway_wrapper", None)
        task = getattr(self, "_im_gateway_heartbeat_task", None)
        session_id = getattr(self, "_im_gateway_session_id", None)
        if task is not None and not task.done():
            task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task
        if wrapper is not None:
            await self._close_gateway_wrapper(wrapper, session_id)
        self._im_gateway_wrapper = None
        self._im_gateway_heartbeat_task = None
        self._im_gateway_session_id = None
        self.im_event_deliver = None
        self.im_event_channel = ""
        return {"ok": True, "message": "disconnected"}

    async def _close_gateway_wrapper(self, wrapper, session_id: str | None) -> None:
        ipc = getattr(wrapper, "_ipc", None)
        if ipc is None:
            return
        with __import__("contextlib").suppress(RuntimeError, ConnectionError, OSError):
            await ipc.unregister(session_id)
        await ipc.close()
