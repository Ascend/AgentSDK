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
"""Run lifecycle, continuation and verification helpers for AgentRunner."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .issue import Issue

logger = logging.getLogger(__name__)


class AgentLifecycleMixin:
    """Provide run ids, continuation checks, artifact export and verification."""

    def _export_events_for_viz(self, session: "AgentSession") -> None:
        """Mirror audit and worker artifacts into the visualizer session dir."""
        try:
            run_id = session.run_id
            workspace_path = getattr(getattr(session, "workspace", None), "path", None)
            if not run_id or not workspace_path:
                return
            destination = Path.home() / ".clawcodex" / "sessions" / str(run_id)
            destination.mkdir(parents=True, exist_ok=True)

            events = Path(workspace_path) / ".reports" / f"{run_id}.events.ndjson"
            if events.is_file():
                shutil.copyfile(events, destination / "events.ndjson")
            spawns = Path(workspace_path) / ".reports" / "agent_spawns.ndjson"
            if spawns.is_file():
                shutil.copyfile(spawns, destination / "agent_spawns.ndjson")

            headless_session_id = ""
            try:
                from src.bootstrap.state import get_session_id

                headless_session_id = str(get_session_id() or "")
            except Exception:
                headless_session_id = ""
            if headless_session_id and headless_session_id != str(run_id):
                source = Path.home() / ".clawcodex" / "sessions" / headless_session_id / "subagents"
                if source.is_dir():
                    target = destination / "subagents"
                    target.mkdir(parents=True, exist_ok=True)
                    for worker_file in source.glob("agent-*.jsonl"):
                        shutil.copyfile(worker_file, target / worker_file.name)
        except Exception:
            logger.exception("viz events mirror failed run_id=%s", getattr(session, "run_id", None))

    def _build_run_id(self, session: "AgentSession") -> str:
        """Build the stable run id used by reports and session artifacts."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt = getattr(session, "attempt", 1)
        if session.run_kind == "review_followup":
            issue_attempt = getattr(session, "issue_attempt", attempt)
            followup_attempt = getattr(session, "followup_attempt", 1)
            return f"run-{issue_attempt}-followup-{followup_attempt}-{timestamp}"
        return f"run-{attempt:02d}-{timestamp}"

    async def _post_summary_placeholder(
        self,
        session: "AgentSession",
        comment_tracker: Any,
    ) -> None:
        """Create the in-progress summary comment and remember its id."""
        body = "## ClawCodex Run Summary\n\n⏳ Run in progress."
        try:
            created = await comment_tracker.create_comment(session.issue.id, body)
        except Exception as exc:
            logger.warning(
                "Failed to post summary placeholder issue_id=%s: %s",
                session.issue.id,
                exc,
            )
            return
        if created is not None and getattr(created, "id", None):
            session.summary_comment_id = created.id

    async def _should_continue(
        self,
        issue: "Issue",
        tracker: Any,
        session: "AgentSession | None" = None,
    ) -> tuple[bool, "Issue"]:
        """Return whether an active issue still needs another agent turn."""
        if not issue.id:
            return False, issue

        cache = getattr(session, "state_cache", None) if session is not None else None
        if cache is not None:
            turn = int(getattr(session, "turn_count", 0) or 0)
            user_interrupted = bool(getattr(session, "user_interrupted", False))
            recent_inactive = cache.has_recent_inactive(issue.id, turn - 1)
            if turn > 0 and not user_interrupted and not recent_inactive and cache.should_skip_poll(issue.id, turn):
                return True, issue

        refreshed = await tracker.fetch_issue_states_by_ids([issue.id])
        refreshed_issue = refreshed.get(issue.id)
        if refreshed_issue is None:
            return False, issue
        active_states = [state.strip().lower() for state in (getattr(tracker, "active_states", None) or [])]
        is_active = refreshed_issue.state is not None and refreshed_issue.state.strip().lower() in active_states
        if cache is not None and is_active and session is not None:
            cache.record(
                issue_id=issue.id,
                is_active=True,
                state=getattr(refreshed_issue, "state", None),
                observed_at_turn=int(getattr(session, "turn_count", 0) or 0),
            )
        if not is_active:
            return False, refreshed_issue

        if session is not None and getattr(session, "turn_count", 0) > 0:
            completion = await self._workspace_completion_state(session)
            if completion is not None:
                head_changed, has_uncommitted, has_user_changes = completion
                if head_changed or has_user_changes or session.status in ("completed", "task_complete"):
                    if head_changed or has_user_changes:
                        session.has_made_progress = True
                    logger.info(
                        "Issue %s work appears done in workspace "
                        "(turn_count=%d, head_changed=%s, "
                        "has_uncommitted=%s, has_user_uncommitted=%s)",
                        issue.id,
                        session.turn_count,
                        head_changed,
                        has_uncommitted,
                        has_user_changes,
                    )
                    return False, refreshed_issue

        if (
            session is not None
            and not getattr(session, "has_made_progress", False)
            and getattr(session, "turn_count", 0) >= 2
            and getattr(session, "tool_count", 0) > 0
            and await self._recent_commits_are_empty(session)
        ):
            return False, refreshed_issue
        return is_active, refreshed_issue

    async def _workspace_completion_state(self, session: "AgentSession") -> tuple[bool, bool, bool] | None:
        workspace_path = getattr(getattr(session, "workspace", None), "path", None)
        if workspace_path is None:
            return None
        try:
            from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
                get_file_status,
            )
            from .agent_session import _has_user_visible_status_changes

            status_entries = await asyncio.to_thread(get_file_status, str(workspace_path))
            has_uncommitted = bool(status_entries)
            has_user_changes = _has_user_visible_status_changes(status_entries)
            head_changed = False
            start_commit = getattr(session, "start_commit_sha", None)
            if start_commit:
                process = await asyncio.to_thread(
                    self._git_capture,
                    ["git", "rev-parse", "HEAD"],
                    str(workspace_path),
                    self._build_subprocess_env(),
                )
                current_head = process.stdout.strip()
                head_changed = bool(current_head and current_head != start_commit)
            return head_changed, has_uncommitted, has_user_changes
        except Exception:
            logger.warning(
                "Workspace completion state check failed issue_id=%s path=%s",
                getattr(getattr(session, "issue", None), "id", "unknown"),
                workspace_path,
                exc_info=True,
            )
            return None

    async def _recent_commits_are_empty(self, session: "AgentSession") -> bool:
        workspace_path = getattr(getattr(session, "workspace", None), "path", None)
        if workspace_path is None:
            return False
        try:
            process = await asyncio.to_thread(
                self._git_capture,
                ["git", "diff", "--stat", "HEAD~3..HEAD"],
                str(workspace_path),
                self._build_subprocess_env(),
            )
            return process.returncode == 0 and not process.stdout.strip()
        except Exception:
            logger.warning(
                "Recent commit check failed issue_id=%s path=%s",
                getattr(getattr(session, "issue", None), "id", "unknown"),
                workspace_path,
                exc_info=True,
            )
            return False

    def _build_subprocess_env(self) -> dict[str, str] | None:
        """Merge workflow-configured environment values over the daemon env."""
        custom_env = getattr(self.agent_config, "env", None)
        if not custom_env:
            return None
        base = os.environ.copy()
        for key, value in custom_env.items():
            if key == "PATH" and value:
                base["PATH"] = value.replace("$PATH", base.get("PATH", ""))
            else:
                base[key] = value
        return base

    @staticmethod
    def _git_capture(args: list[str], cwd: str, env: dict[str, str] | None) -> subprocess.CompletedProcess[str]:
        """Run git synchronously; async callers offload this to a thread."""
        return subprocess.run(  # nosec B603 - args are fixed internal git commands
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            check=False,
        )

    async def _run_verification(self, session: "AgentSession") -> bool:
        """Run the configured test command in the issue workspace."""
        test_command = getattr(self.agent_config, "test_command", None)
        if not test_command:
            return True
        workspace_path = getattr(getattr(session, "workspace", None), "path", None)
        if not workspace_path:
            return False
        timeout_ms = getattr(
            getattr(self.agent_config, "verification", None),
            "timeout_ms",
            600_000,
        )
        process: asyncio.subprocess.Process | None = None
        try:
            from .agent_session import _set_pdeathsig

            process = await asyncio.create_subprocess_shell(
                test_command,
                cwd=str(workspace_path),
                preexec_fn=_set_pdeathsig,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_subprocess_env(),
            )
            await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_ms / 1000.0,
            )
            return process.returncode == 0
        except asyncio.TimeoutError:
            logger.warning("Verification timed out for issue_id=%s", session.issue.id)
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                    await process.communicate()
                except ProcessLookupError:
                    pass
                except Exception:
                    logger.warning(
                        "Failed to terminate timed-out verification issue_id=%s",
                        session.issue.id,
                        exc_info=True,
                    )
            return False
        except Exception as exc:
            logger.warning("Verification error for issue_id=%s: %s", session.issue.id, exc)
            return False


__all__ = ["AgentLifecycleMixin"]
