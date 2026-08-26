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

"""Polling engine — GenServer equivalent in Python.

Port of Symphony's Orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
    CardUpdateCapability,
    ChannelCapability,
)
from extensions.orchestrator_runtime.adapters.clawcodex_compat import ToolContext

from .agent_runner import AgentRunner, AgentSession, RetryItem
from .config.schema import WorkflowConfig
from .events import EventLevel
from .git_sync import (
    GitSyncService,
)
from .issue import Issue
from .issue_registry import IssueRegistry
from .mode_router import HeuristicRouter, LLMRouter, Router
from .mode_selector import ModeSelector
from . import modes as _modes
from .modes.coordinator import CoordinatorModeRunner
from .modes.debate import DebateModeRunner
from .modes.pipeline import PipelineModeRunner
from .modes.single import SingleModeRunner
from .modes.swarm import SwarmModeRunner
from .status_dashboard import StatusDashboard
from .tracker import (
    TrackerAdapter,
)
from .workspace import WorkspaceManager

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


@dataclass
class OrchestratorState:
    """Runtime state for the orchestrator polling loop."""

    poll_interval_ms: int = 30_000
    max_concurrent_agents: int = 10
    next_poll_due_at_ms: float | None = None
    poll_check_in_progress: bool = False
    running: dict[str, AgentSession] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    pending_review: set[str] = field(default_factory=set)  # awaiting human review
    claimed: set[str] = field(default_factory=set)
    retry_queue: list[RetryItem] = field(default_factory=list)
    retry_attempts: dict[str, int] = field(default_factory=dict)
    # Throttle marker for the optional PR conflict scan. Wall-clock
    # seconds (not ms) of the last scan — compared against
    # ``time.monotonic()`` so a backwards clock jump is benign.
    pr_conflict_scan_last_run: float = 0.0
    codex_totals: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "seconds_running": 0,
        }
    )


from .orchestrator_control import OrchestratorControlMixin  # noqa: E402
from .orchestrator_issue import OrchestratorIssueMixin  # noqa: E402
from .orchestrator_ops import OrchestratorOpsMixin  # noqa: E402
from .orchestrator_rebase import OrchestratorRebaseMixin  # noqa: E402
from .orchestrator_run import OrchestratorRunMixin  # noqa: E402
from .orchestrator_session import OrchestratorSessionMixin  # noqa: E402


class Orchestrator(
    OrchestratorSessionMixin,
    OrchestratorRebaseMixin,
    OrchestratorIssueMixin,
    OrchestratorRunMixin,
    OrchestratorOpsMixin,
    OrchestratorControlMixin,
):
    """Polling engine — GenServer equivalent in Python."""

    def __init__(
        self,
        workflow: WorkflowConfig,
        tracker: TrackerAdapter,
        workspace: WorkspaceManager,
        agent_runner: AgentRunner,
        status_dashboard: StatusDashboard | None = None,
        *,
        stage_runners: dict[str, "AgentRunner"] | None = None,
        workflow_yaml_path: str | None = None,
        asciicast_capture: Any = None,
    ) -> None:
        self.workflow = workflow
        self.tracker = tracker
        self.workspace = workspace
        self.agent_runner = agent_runner
        self.stage_runners = stage_runners or {}
        # F-REC: optional asciicast capture. When set, every per-session
        # :class:`CompositeProgressSink` built by :meth:`_build_session_sink`
        # registers an :class:`AsciicastSink` so the agent's progress
        # events land in the same ``.cast`` file as the other adapters.
        # ``None`` (the default) preserves the existing behaviour — no
        # recording happens, no extra import cost.
        self.asciicast_capture = asciicast_capture
        # Collaboration modes — Phase 2 wires the registry +
        # ``ModeSelector`` + ``Router`` based on the ``modes:`` YAML
        # section. ``ModesConfig`` defaults (no router, only "single"
        # enabled) preserve byte-identical behavior for workflows that
        # don't opt in.
        self._register_collaboration_modes(workflow, agent_runner)
        self._mode_selector = self._build_mode_selector(workflow)
        self._workflow_yaml_path = workflow_yaml_path
        self._workflow_orchestrator = None

        # The StateJournalWriter existed but was never
        # instantiated anywhere, so the visualizer's orchestrator dashboard
        # (reads ``~/.clawcodex/reports/run_*/state_journal.ndjson``) always
        # showed "no runs". One journal per daemon lifetime; writes are
        # fire-and-forget and must never affect orchestration.
        self._viz_journal = None
        try:
            from datetime import datetime, timezone

            from .state_journal import StateJournalWriter

            journal_run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._viz_journal = StateJournalWriter(
                Path.home() / ".clawcodex" / "reports" / journal_run_id,
                journal_run_id,
            )
            self._viz_journal.write_event(
                {
                    "type": "orchestrator_start",
                    "workflow": workflow_yaml_path or "",
                }
            )
        except Exception:
            logger.exception("state journal init failed — dashboard disabled")
            self._viz_journal = None

        # Initialize the declarative workflow engine
        if workflow_yaml_path:
            from .workflow_orchestrator import WorkflowOrchestrator

            self._workflow_orchestrator = WorkflowOrchestrator(
                workflow_config=workflow,
                workflow_yaml_path=workflow_yaml_path,
                agent_runner=agent_runner,
                tracker=tracker,
                status_dashboard=status_dashboard,
                diagnostics_callback=self._update_run_diagnostics,
            )
            logger.info(
                "Workflow engine enabled: %s (%s, %d stages)",
                workflow_yaml_path,
                self._workflow_orchestrator.schema.name,
                len(self._workflow_orchestrator.schema.stages),
            )

        self.status_dashboard = status_dashboard or StatusDashboard()
        self._agent_config = workflow.agent
        # IM-side channel adapter (e.g. FeishuAppChannelAdapter). When
        # set, :meth:`_build_session_sink` attaches a
        # :class:`FeishuActivitySink` so the bot's reactions + placeholder
        # progress card track the agent lifecycle for users on IM. None
        # → activity sink disabled (default; not every deployment has an IM
        # channel even if ``im_event_deliver`` is wired).
        self.im_channel_adapter: Any = None
        self._validate_workspace_strategy()
        self.git_sync = GitSyncService(
            tracker,
            workflow.tracker.branch_prefix,
            workflow.workspace.gitignore_patterns,
            workflow.agent,
            workflow.hooks,
            git_username=workflow.workspace.git_username,
            git_email=workflow.workspace.git_email,
            upstream_clone_url=workflow.workspace.upstream_clone_url,
            fork_clone_url=workflow.workspace.repo_clone_url,
            pr_template=workflow.pr_template,
        )
        self._state = OrchestratorState(
            poll_interval_ms=workflow.polling.interval_ms,
            max_concurrent_agents=workflow.agent.max_concurrent_agents,
        )
        self._semaphore = asyncio.Semaphore(workflow.agent.max_concurrent_agents)
        self._shutdown_event = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        # Root-cause fix: map issue_id → asyncio.Task so the stop
        # command can cancel a specific running issue by task.cancel().
        self._issue_tasks: dict[str, asyncio.Task] = {}
        # Store workflow path for metadata
        self._workflow_path: str | None = getattr(workflow, "source_path", None) or getattr(
            workflow, "_source_path", None
        )
        self._dynamic_tracker_config_mtime_ns: int | None = self._workflow_mtime_ns()
        # Workspace root for control command polling
        workspace_root = Path(workspace.config.root)
        self._workspace_root = workspace_root
        # Persistent issue→commit→PR mapping (persists across restarts)
        registry_path = workspace_root / ".clawcodex_issue_registry.json"
        self._registry = IssueRegistry(registry_path)

        # Write orchestrator metadata for CLI discovery
        self._metadata_started_at = time.time()
        from .workspace_locator import write_orchestrator_metadata

        write_orchestrator_metadata(
            workspace_root=workspace_root,
            workflow_path=self._workflow_path,
            started_at=self._metadata_started_at,
        )

        # Clarification handling (three-channel flow)
        clarification_queue_path = workspace_root / ".clawcodex_clarification_queue.json"
        from .clarification_queue import ClarificationQueue

        self._clarification_queue = ClarificationQueue(clarification_queue_path)

        from .clarification import (
            ClarificationConfig,
            ClarificationResolver,
            _DEFAULT_MAX_QUESTIONS_PER_ISSUE,
            _DEFAULT_SIMULTANEOUS_GRACE_MS,
            _DEFAULT_TIMEOUT_AUTHOR_SECONDS,
            _DEFAULT_TIMEOUT_LOCAL_SECONDS,
        )

        self._clarification_resolver = ClarificationResolver(
            clarification_queue=self._clarification_queue,
            tracker=tracker,
            config=ClarificationConfig(
                enabled=getattr(workflow.agent, "clarification_enabled", True),
                timeout_local_seconds=getattr(
                    workflow.agent, "clarification_timeout_local", _DEFAULT_TIMEOUT_LOCAL_SECONDS
                ),
                timeout_author_seconds=getattr(
                    workflow.agent, "clarification_timeout_author", _DEFAULT_TIMEOUT_AUTHOR_SECONDS
                ),
                max_questions_per_issue=getattr(
                    workflow.agent, "max_questions_per_issue", _DEFAULT_MAX_QUESTIONS_PER_ISSUE
                ),
                operator_priority=getattr(workflow.agent, "clarification_operator_priority", True),
                simultaneous_grace_ms=getattr(
                    workflow.agent, "clarification_simultaneous_grace_ms", _DEFAULT_SIMULTANEOUS_GRACE_MS
                ),
                escalation=getattr(workflow.agent, "clarification_escalation", "skip"),
            ),
        )
        self._clarification_gate = None
        clarifier_config = getattr(workflow, "clarifier", None)
        if clarifier_config is not None and bool(getattr(clarifier_config, "enabled", False)):
            from clawcodex_ext.providers.runtime import build_provider_from_config

            from .issue_clarifier import ClarifierCache, IssueClarifierService
            from .issue_clarifier.gate import IssueClarificationGate

            cache = ClarifierCache(
                workspace_root / ".clawcodex_issue_clarifier_cache.json",
                enabled=bool(getattr(clarifier_config, "cache_enabled", True)),
            )

            def _build_clarifier_provider() -> Any:
                return build_provider_from_config(
                    workflow.agent.provider,
                    getattr(workflow.agent, "model", None),
                )

            service = IssueClarifierService(
                config=clarifier_config,
                cache=cache,
                provider_factory=_build_clarifier_provider,
                model=getattr(workflow.agent, "model", None),
            )
            self._clarification_gate = IssueClarificationGate(
                service=service,
                resolver=self._clarification_resolver,
                registry=self._registry,
                config=clarifier_config,
                tracker=self.tracker,
                workspace_focus_callback=self._compute_workspace_focus_for_clarifier,
            )
            logger.info(
                "Issue clarifier enabled (block=%s, author_first=%s)",
                clarifier_config.block_on_unclear,
                clarifier_config.author_first,
            )
        self._progress_context = ToolContext(workspace_root=workspace_root)
        # P3 IM event bridge: if set (by the daemon wiring a gateway deliver),
        # :meth:`_build_session_sink` attaches an :class:`OrchestratorEventEmitter`
        # so key orchestrator events push to IM. None → IM events disabled.
        self.im_event_deliver: "object | None" = None
        self.im_event_channel: str = ""
        self._im_emitters: dict = {}
        # Do NOT keep a single :class:`ProgressReporter` here.
        # Per-session progress is fanned out via
        # :meth:`_build_session_sink` (a fresh
        # :class:`CompositeProgressSink` rooted in a private
        # :class:`ToolContextProgressSink`) so concurrent issues can no
        # longer share ``_current_task_id`` / ``_phase_count`` state.
        # The shared ``_progress_context`` stays because every
        # per-session :class:`ToolContextProgressSink` writes into the
        # same ``ToolContext.tasks[id].metadata.progress_stages`` dict.

    def _build_session_sink(self, task_id: str) -> Any:
        """Build a fresh :class:`CompositeProgressSink` for one session.

        The returned sink is bound to ``task_id`` and owns a private
        :class:`ToolContextProgressSink` instance. Two sinks built for
        different task ids share the underlying ``ToolContext`` (so
        progress stages land in the right place) but have independent
        phase counters, eliminating the legacy single-instance
        cross-talk.

        Future issues (PR review auto-fix sink, retry label sink)
        can register additional sinks on the returned composite via
        :meth:`CompositeProgressSink.add` without touching
        :class:`AgentRunner` or ``progress_reporter.py``.
        """
        from .progress_sink import (
            CompositeProgressSink,
            ToolContextProgressSink,
        )

        inner = ToolContextProgressSink(
            task_id=task_id,
            context=self._progress_context,
            workflow_phases=self.workflow.agent.phases,
            fallback_to_phase_step=bool(self.workflow.agent.fallback_to_phase_step),
        )
        composite = CompositeProgressSink([inner])
        # P3: attach the IM event emitter when a deliver callback is wired.
        if getattr(self, "im_event_deliver", None) is not None:
            from .channel_sink import ChannelProgressSink
            from .events import OrchestratorEvent, OrchestratorEventEmitter

            channel_sink = ChannelProgressSink(self.im_event_deliver)
            emitter = OrchestratorEventEmitter(
                task_id=task_id,
                sinks=[channel_sink],
            )
            # Stash for explicit emit() at blind-spot call sites.
            self._im_emitters[task_id] = emitter
            composite.add(emitter)
            emitter.emit(
                OrchestratorEvent(
                    event_type="issue.started",
                    issue_id=task_id,
                    level=EventLevel.INFO,
                    message="任务已启动",
                    payload=self._issue_payload_for_task_id(task_id),
                )
            )
        # IM-side activity sink: attach only through the public card-update
        # protocol and its declared capability. Channel-specific caches and
        # loop internals stay behind the adapter boundary.
        im_adapter = getattr(self, "im_channel_adapter", None)
        if isinstance(im_adapter, CardUpdateCapability) and im_adapter.capabilities.has(ChannelCapability.CARD_UPDATE):
            from .feishu_activity_sink import FeishuActivitySink

            phases_total = len(self.workflow.agent.phases) if getattr(self.workflow.agent, "phases", None) else None
            activity_sink = FeishuActivitySink(
                task_id=task_id,
                feishu_adapter=im_adapter,
                clock=time.time,
                status_dashboard=self.status_dashboard,
                phases_total=phases_total,
            )
            composite.add(activity_sink)
        # F-REC: when a capture handle is wired (typically by the
        # ``clawcodex record`` CLI or by ``report_writer.write`` dual-
        # write), attach an :class:`AsciicastSink` so phase / session
        # markers land in the .cast. Defensive try/except mirrors the
        # IM-sink block above — recording failures must never block
        # the live orchestrator.
        capture = getattr(self, "asciicast_capture", None)
        if capture is not None:
            try:
                from .asciicast_sink import AsciicastSink

                phases_total = len(self.workflow.agent.phases) if getattr(self.workflow.agent, "phases", None) else None
                composite.add(
                    AsciicastSink(
                        capture,
                        task_id=task_id,
                        phases_total=phases_total,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "asciicast sink attach failed (task_id=%s): %s",
                    task_id,
                    exc,
                )
        return composite

    def _emit_im_event(
        self,
        issue_id: str,
        event_type: str,
        level: EventLevel,
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one key orchestrator event to IM if the bridge is enabled."""
        issue_id = issue_id or "orchestrator"
        emitters = getattr(self, "_im_emitters", {})
        emitter = emitters.get(issue_id)
        if emitter is None:
            deliver = getattr(self, "im_event_deliver", None)
            if deliver is None:
                return
            from .channel_sink import ChannelProgressSink
            from .events import OrchestratorEventEmitter

            emitter = OrchestratorEventEmitter(issue_id, sinks=[ChannelProgressSink(deliver)])
            emitters[issue_id] = emitter
            self._im_emitters = emitters
        from .events import OrchestratorEvent

        emitter.emit(
            OrchestratorEvent(
                event_type=event_type,
                issue_id=issue_id,
                level=level,
                message=message,
                payload=dict(payload or {}),
            )
        )

    def _issue_payload_for_task_id(self, task_id: str) -> dict[str, Any]:
        """Build a payload for issue.started when only the task_id is known.

        At sink-build time the Issue object is on ``session.issue`` but
        ``_build_session_sink`` receives only the task_id. We look up the
        registry record for branch/identifier, and the tracker for repo.
        """
        payload: dict[str, Any] = {}
        registry = getattr(self, "_registry", None)
        record = registry.get(task_id) if registry and task_id else None
        if record is not None:
            if getattr(record, "issue_identifier", None):
                payload["title"] = record.issue_identifier
            if getattr(record, "branch_name", None):
                payload["branch"] = record.branch_name
        repo = self._repo_label()
        if repo:
            payload["repo"] = repo
        return payload

    def _repo_label(self) -> str:
        """Build a 'owner/repo' label from the tracker, or '' if unavailable."""
        tracker = getattr(self, "tracker", None)
        if tracker is None:
            return ""
        owner = getattr(tracker, "owner", None)
        repo = getattr(tracker, "repo", None)
        if owner and repo:
            return f"{owner}/{repo}"
        return ""

    def _issue_payload(self, issue: Issue, **extra: Any) -> dict[str, Any]:
        """Build a rich payload dict for IM events from an Issue + extras.

        Centralizes the issue title / branch / repo context so every emit
        call site gets consistent enrichment without repeating field
        extraction. ``extra`` kwargs are merged in (e.g. commit=, pr=,
        verification=, attempts=).
        """
        payload: dict[str, Any] = {}
        title = getattr(issue, "title", None)
        if title:
            payload["title"] = title
        branch = getattr(issue, "branch_name", None)
        if branch:
            payload["branch"] = branch
        repo = self._repo_label()
        if repo:
            payload["repo"] = repo
        payload.update({k: v for k, v in extra.items() if v is not None})
        return payload

    def _session_payload(self, session: Any, **extra: Any) -> dict[str, Any]:
        """Build a rich payload from an AgentSession + extras.

        Reads issue title/branch, repo, verification status, PR url, and
        commit sha from the session/registry, then merges ``extra``.
        """
        issue = getattr(session, "issue", None)
        payload: dict[str, Any] = {}
        if issue is not None:
            title = getattr(issue, "title", None)
            if title:
                payload["title"] = title
            branch = getattr(issue, "branch_name", None)
            if branch:
                payload["branch"] = branch
            pr_url = getattr(issue, "pr_url", None)
            if pr_url:
                payload["pr"] = pr_url
        repo = self._repo_label()
        if repo:
            payload["repo"] = repo
        ver = getattr(session, "verification_status", None)
        if ver:
            payload["verification"] = ver
        # Try to get commit sha from the registry record
        issue_id = getattr(issue, "id", None) if issue is not None else None
        registry = getattr(self, "_registry", None)
        if issue_id and registry is not None:
            record = registry.get(issue_id)
            if record is not None:
                commit = getattr(record, "commit_sha", None)
                if commit:
                    payload.setdefault("commit", commit)
        payload.update({k: v for k, v in extra.items() if v is not None})
        return payload

    def _register_collaboration_modes(self, workflow: WorkflowConfig, agent_runner: AgentRunner) -> None:
        """Register the ``ModeRunner`` instances that match ``modes.enabled``.

        ``single`` is always registered (it's the safe fallback). Other
        modes are registered only when listed in ``workflow.modes.enabled``
        so an operator can disable a mode without removing its code.
        """
        # Always register "single" — it's both the default fallback and
        # the run mode for legacy / followup / review_followup paths.
        _modes.register("single", SingleModeRunner(agent_runner))

        enabled = {m.strip().lower() for m in workflow.modes.enabled if m}
        if "pipeline" in enabled:
            stages = tuple(workflow.modes.pipeline_stages)
            max_retries = int(getattr(workflow.modes, "pipeline_max_retries_per_stage", 1))
            stage_models = dict(getattr(workflow.modes, "pipeline_stage_models", None) or {})
            stage_max_turns = dict(getattr(workflow.modes, "pipeline_stage_max_turns", None) or {})
            stage_specs = dict(getattr(workflow.modes, "pipeline_stage_specs", None) or {})
            handoff = str(getattr(workflow.modes, "pipeline_handoff", "prompt"))
            try:
                _modes.register(
                    "pipeline",
                    PipelineModeRunner(
                        agent_runner,
                        stages=stages,
                        max_retries_per_stage=max_retries,
                        stage_models=stage_models,
                        stage_max_turns=stage_max_turns,
                        stage_specs=stage_specs,
                        handoff=handoff,
                    ),
                )
            except ValueError as exc:
                # Bad stage_specs (e.g. kind=pipeline nested). Fall back
                # to a spec-less pipeline so the daemon keeps running.
                logger.warning(
                    "Pipeline registration failed (%s) — registering without stage_specs",
                    exc,
                )
                _modes.register(
                    "pipeline",
                    PipelineModeRunner(
                        agent_runner,
                        stages=stages,
                        max_retries_per_stage=max_retries,
                        stage_models=stage_models,
                        stage_max_turns=stage_max_turns,
                        stage_specs={},
                        handoff=handoff,
                    ),
                )
                stage_specs = {}
            logger.info(
                "Collaboration mode registered: pipeline (stages=%s, "
                "max_retries_per_stage=%d, stage_models=%s, "
                "stage_max_turns=%s, stage_specs=%s, handoff=%s)",
                stages,
                max_retries,
                stage_models or "(none)",
                stage_max_turns or "(none)",
                stage_specs or "(none)",
                handoff,
            )
        if "coordinator" in enabled:
            _modes.register("coordinator", CoordinatorModeRunner(agent_runner))
            logger.info("Collaboration mode registered: coordinator")
        if "swarm" in enabled:
            _modes.register(
                "swarm",
                SwarmModeRunner(
                    agent_runner,
                    max_subtasks=workflow.modes.swarm_max_subtasks,
                    max_parallel=workflow.modes.swarm_max_parallel,
                    max_waves=workflow.modes.swarm_max_waves,
                ),
            )
            logger.info(
                "Collaboration mode registered: swarm (max_subtasks=%d, max_parallel=%d, max_waves=%d)",
                workflow.modes.swarm_max_subtasks,
                workflow.modes.swarm_max_parallel,
                workflow.modes.swarm_max_waves,
            )
        if "debate" in enabled:
            proposers = tuple(getattr(workflow.modes, "debate_proposers", None) or ("proposer_a", "proposer_b"))
            judge_model = getattr(workflow.modes, "debate_judge_model", None)
            isolation = getattr(workflow.modes, "debate_isolation", "reset")
            proposer_models = dict(getattr(workflow.modes, "debate_proposer_models", None) or {})
            parallel = bool(getattr(workflow.modes, "debate_parallel", False))
            judge_mode = str(getattr(workflow.modes, "debate_judge_mode", "pick"))
            try:
                _modes.register(
                    "debate",
                    DebateModeRunner(
                        agent_runner,
                        proposers=proposers,
                        judge_model=judge_model,
                        isolation=isolation,
                        proposer_models=proposer_models,
                        parallel=parallel,
                        judge_mode=judge_mode,
                    ),
                )
            except ValueError as exc:
                # Most likely: parallel=True without isolation=worktree,
                # or an invalid judge_mode. Fall back to safe defaults so
                # the daemon keeps running.
                logger.warning(
                    "Debate registration failed (%s) — registering with "
                    "parallel=False, isolation='%s', judge_mode='pick'",
                    exc,
                    isolation,
                )
                _modes.register(
                    "debate",
                    DebateModeRunner(
                        agent_runner,
                        proposers=proposers,
                        judge_model=judge_model,
                        isolation=isolation,
                        proposer_models=proposer_models,
                        parallel=False,
                        judge_mode="pick",
                    ),
                )
                parallel = False
                judge_mode = "pick"
            logger.info(
                "Collaboration mode registered: debate (proposers=%s, "
                "judge_model=%s, isolation=%s, parallel=%s, "
                "proposer_models=%s, judge_mode=%s)",
                proposers,
                judge_model or "(default)",
                isolation,
                parallel,
                proposer_models or "(none)",
                judge_mode,
            )

    def _build_mode_selector(self, workflow: WorkflowConfig) -> ModeSelector:
        """Construct ``ModeSelector`` with the configured router backend."""
        router: Router | None
        kind = workflow.modes.router_kind
        if kind == "heuristic":
            router = HeuristicRouter()
            logger.info("ModeSelector: router=HeuristicRouter")
        elif kind == "llm":
            router = LLMRouter(
                model=workflow.modes.router_model,
                endpoint=workflow.modes.router_endpoint,
                api_key_env_var=workflow.modes.router_api_key_env,
                timeout_seconds=workflow.modes.router_timeout_seconds,
            )
            logger.info(
                "ModeSelector: router=LLMRouter(model=%s, endpoint=%s, api_key_env=%s, timeout=%.1fs)",
                workflow.modes.router_model,
                workflow.modes.router_endpoint,
                workflow.modes.router_api_key_env,
                workflow.modes.router_timeout_seconds,
            )
        else:
            router = None
            logger.info("ModeSelector: no router configured (kind=%s)", kind)

        default_mode = workflow.modes.default
        try:
            return ModeSelector(
                default_mode=default_mode,
                router=router,
                min_confidence=workflow.modes.router_min_confidence,
            )
        except ValueError as exc:
            # workflow.md misconfiguration — fall back to safe defaults
            # instead of crashing the daemon at startup.
            logger.warning("ModeSelector construction failed (%s); using defaults", exc)
            return ModeSelector()

    def _validate_workspace_strategy(self) -> None:
        if self.workflow.workspace.strategy != "sequential":
            return
        if self.workflow.agent.max_concurrent_agents != 1:
            raise ValueError("workspace.strategy=sequential requires agent.max_concurrent_agents=1")
        over_limit_states = [
            state for state, limit in self.workflow.agent.max_concurrent_agents_by_state.items() if limit > 1
        ]
        if over_limit_states:
            raise ValueError(
                "workspace.strategy=sequential requires all agent.max_concurrent_agents_by_state values to be <= 1"
            )

    def _sync_gitignore_to_workspace(self, workspace: Any) -> None:
        """Write ignore patterns for orchestrator-managed workspace files.

        Always writes to ``.git/info/exclude`` (local-only) rather than
        ``.gitignore`` so that orchestrator patterns are never tracked by
        git and never appear in agent commits.
        """
        workspace_path = Path(workspace.path)
        ignore_path = workspace_path / ".git" / "info" / "exclude"
        if not ignore_path.parent.exists():
            return

        patterns = self.git_sync._gitignore_patterns
        existing: set[str] = set()
        if ignore_path.exists():
            existing = {
                line.strip()
                for line in ignore_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            }

        new_patterns = [p for p in patterns if p not in existing]
        if not new_patterns:
            return

        with ignore_path.open("a", encoding="utf-8") as f:
            if ignore_path.exists() and ignore_path.stat().st_size > 0:
                f.write("\n")
            f.write("# ClawCodeX managed — do not edit manually\n")
            for p in new_patterns:
                f.write(f"{p}\n")
        logger.debug("Updated %s with %d patterns", ignore_path, len(new_patterns))

    async def run(self) -> None:
        """Main polling loop. Runs until cancelled."""
        logger.info(
            "Orchestrator starting: interval=%sms max_concurrent=%s",
            self._state.poll_interval_ms,
            self._state.max_concurrent_agents,
        )

        # Best-effort session_start at the top of the polling
        # loop. The session id is the workflow root path's basename
        # plus a stable hash so the per-day aggregator can group all
        # orchestrator daemons across the day. Failures are swallowed.
        orch_start = time.monotonic()
        orch_session_id = self._derive_orchestrator_session_id()
        try:
            from telemetry import record_session_start

            record_session_start(
                session_id=orch_session_id,
                entrypoint="orchestrator",
                client_type="cli",
                is_non_interactive=True,
            )
        except Exception:  # nosec B110
            pass

        # Clean up terminal workspaces on startup
        await self.workspace.run_terminal_workspace_cleanup()
        await self._recover_stale_running_records()

        # Start metadata heartbeat for CLI discovery
        heartbeat_task = asyncio.create_task(self._metadata_heartbeat_loop())
        self._tasks.add(heartbeat_task)

        try:
            while not self._shutdown_event.is_set():
                await self._poll_and_dispatch()
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._state.poll_interval_ms / 1000.0,
                    )
                except asyncio.TimeoutError:
                    pass

            logger.info("Orchestrator shutting down")
            await self._cancel_all_tasks()
            exit_status = 0
        except Exception as exc:
            # Best-effort error event with stable fingerprint.
            # Failures are swallowed.
            try:
                from telemetry import record_error

                record_error(session_id=orch_session_id, exc=exc)
            except Exception:  # nosec B110
                pass
            exit_status = 1
            raise
        finally:
            # Best-effort session_end + command_run.
            try:
                from telemetry import (
                    record_command_run,
                    record_session_end,
                )

                duration_s = time.monotonic() - orch_start
                record_session_end(
                    session_id=orch_session_id,
                    duration_s=duration_s,
                    exit_status=exit_status,
                )
                record_command_run(
                    session_id=orch_session_id,
                    command_name="orchestrator",
                    mode="daemon",
                    success=(exit_status == 0),
                    duration_s=duration_s,
                    exit_status=exit_status,
                )
            except Exception:  # nosec B110
                pass
