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

# pylint: disable=relative-beyond-top-level

"""StageRunner adapter.

Bridges DeclarativeWorkflowEngine and AgentRunner,
adapting stage execution into synthetic-issue work units consumable by AgentRunner.

Design decision DD-5: Option A (synthetic-issue adapter) preferred,
keeping all of AgentRunner's robustness mechanisms (stagnation detection, progress reporting, verification pipeline, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .checkpoint import ArtifactResolver
from .validators import ContractValidator
from .workflow_state import StageNode, WorkflowState

if TYPE_CHECKING:
    from ..agent_runner import AgentRunner, AgentSession
    from ..config.schema import AgentConfig, SandboxConfig, WorkflowConfig

logger = logging.getLogger(__name__)


@dataclass
class StageRunResult:
    """StageRunner execution result."""

    stage_id: int
    success: bool
    outputs: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None
    message: str = ""


@dataclass
class GateRunResult:
    """GATE stage execution result."""

    stage_id: int
    approved: bool
    reason: str = ""
    cost_usd: float = 0.0


@dataclass
class DecisionRunResult:
    """DECISION stage execution result."""

    stage_id: int
    outcome: str  # proceed | pivot | refine | rollback
    next_stage: int | None = None
    cost_usd: float = 0.0


class StageRunner:
    """Stage execution adapter.

    Invokes AgentRunner through a synthetic issue, reusing its full lifecycle management:
    - Stagnation detection
    - Progress reporting (ProgressSink)
    - Verification pipeline
    - Clarification queue
    - Cost tracking
    """

    MAX_RETRIES = 2

    def __init__(
        self,
        agent_runner: "AgentRunner",
        workflow_config: "WorkflowConfig",
        agent_config: "AgentConfig | None" = None,
        sandbox_config: "SandboxConfig | None" = None,
        workspace_dir: str = "",
        run_dir: str = "",
        tracker: Any = None,
        status_dashboard: Any = None,
        clarification_resolver: Any = None,
        progress_reporter: Any = None,
        llm_client: Any = None,
        diagnostics_callback: Any = None,
    ) -> None:
        self._agent_runner = agent_runner
        self._workflow_config = workflow_config
        self._agent_config = agent_config
        self._sandbox_config = sandbox_config
        self._workspace_dir = workspace_dir
        self._run_dir = run_dir
        self._tracker = tracker
        self._status_dashboard = status_dashboard
        self._clarification_resolver = clarification_resolver
        self._progress_reporter = progress_reporter
        self._llm_client = llm_client
        self._diagnostics_callback = diagnostics_callback
        self._bundle_path: Path | None = None
        self._validator = ContractValidator(
            workspace_dir=self._workspace_dir,
            llm_client=self._llm_client,
        )

    def set_bundle_path(self, bundle_path: Path | str | None) -> None:
        self._bundle_path = Path(bundle_path).resolve() if bundle_path else None

    # -- Public interface --─────────────────────────────────────────────────

    async def run(self, stage_node: StageNode, state: WorkflowState) -> StageRunResult:
        """Execute an agent stage.

        Builds a synthetic issue, runs it via AgentRunner, with automatic retries.
        """
        last_error: str | None = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return await self._execute_agent_stage(stage_node, state)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Stage %s attempt %d/%d failed: %s",
                    stage_node.id,
                    attempt + 1,
                    self.MAX_RETRIES + 1,
                    exc,
                )
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(2**attempt)

        return StageRunResult(
            stage_id=stage_node.id,
            success=False,
            error=last_error,
        )

    async def run_gate(self, stage_node: StageNode, state: WorkflowState) -> GateRunResult:
        """Execute a GATE stage."""
        mode = stage_node.gate_mode
        max_retries = stage_node.max_retries

        for attempt in range(max_retries + 1):
            if mode == "auto":
                result = await self._run_auto_gate(stage_node, state)
            elif mode == "threshold":
                result = await self._run_threshold_gate(stage_node, state)
            else:
                return GateRunResult(
                    stage_id=stage_node.id,
                    approved=False,
                    reason=f"GATE stage {stage_node.id} requires manual approval",
                )

            if result.approved or attempt >= max_retries:
                return result

            logger.warning(
                "GATE stage %s attempt %d/%d rejected, retrying...",
                stage_node.id,
                attempt + 1,
                max_retries + 1,
            )
            await asyncio.sleep(2**attempt)

        return GateRunResult(
            stage_id=stage_node.id,
            approved=False,
            reason=f"GATE stage {stage_node.id} failed after {max_retries + 1} attempts",
        )

    async def run_decision(self, stage_node: StageNode, state: WorkflowState) -> DecisionRunResult:
        """Execute a DECISION stage."""
        max_retries = stage_node.max_retries

        for attempt in range(max_retries + 1):
            try:
                session = await self._run_synthetic_issue(
                    prompt=self._build_decision_prompt(stage_node, state),
                    stage_node=stage_node,
                )
                output_text = session.output_text if session else ""
                outcome = self._parse_decision_outcome(output_text, stage_node)
                next_stage = self._resolve_next_stage(outcome, stage_node)

                if outcome != "proceed" or attempt >= max_retries:
                    return DecisionRunResult(
                        stage_id=stage_node.id,
                        outcome=outcome,
                        next_stage=next_stage,
                    )
                # Retrying with "proceed" outcome: back off before the next attempt
                await asyncio.sleep(2**attempt)
            except Exception as exc:
                logger.warning(
                    "Decision stage %s attempt %d/%d failed: %s",
                    stage_node.id,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)
                else:
                    return DecisionRunResult(stage_id=stage_node.id, outcome="failed")

        return DecisionRunResult(stage_id=stage_node.id, outcome="failed")

    # -- Agent stage execution (DD-5: synthetic-issue adapter) --────────────────

    async def _execute_agent_stage(
        self,
        stage_node: StageNode,
        state: WorkflowState,
    ) -> StageRunResult:
        """Execute an agent stage via a synthetic issue + AgentRunner."""
        # Record total cost before the stage to compute the stage delta
        cost_before = self._get_total_cost_usd()

        prompt = self._build_stage_prompt(stage_node, state)
        session = await self._run_synthetic_issue(prompt=prompt, stage_node=stage_node)

        if session is None:
            return StageRunResult(
                stage_id=stage_node.id,
                success=False,
                error="AgentRunner returned no session",
            )

        output_text = session.output_text if hasattr(session, "output_text") else ""
        status = session.status if hasattr(session, "status") else "unknown"

        # Stage cost = total cost delta
        cost_delta = self._get_total_cost_usd() - cost_before

        return StageRunResult(
            stage_id=stage_node.id,
            success=status == "completed",
            outputs=[output_text] if output_text else [],
            message=output_text,
            cost_usd=max(cost_delta, 0.0),
            error=None if status == "completed" else f"Session status: {status}",
        )

    async def _run_synthetic_issue(
        self,
        prompt: str,
        stage_node: StageNode,
    ) -> "AgentSession | None":
        """Build a synthetic issue and invoke AgentRunner (DD-5)."""
        from ..issue import Issue
        from ..workspace import Workspace
        from ..agent_runner import AgentSession

        # Build synthetic issue
        synthetic_issue = Issue(
            id=f"stage-{stage_node.id:02d}",
            identifier=f"stage-{stage_node.id:02d}",
            title=f"[{stage_node.phase}] {stage_node.name}",
            description=prompt,
            labels=["workflow-stage", f"workflow-{stage_node.phase}"],
        )

        # Build Workspace (shared directory, DD-6)
        workspace_path = Path(self._workspace_dir) if self._workspace_dir else Path(".")
        workspace = Workspace(
            path=workspace_path,
            issue_identifier=f"stage-{stage_node.id:02d}",
            issue_id=f"stage-{stage_node.id:02d}",
        )

        # Build AgentSession
        session = AgentSession(
            issue=synthetic_issue,
            workspace=workspace,
            run_kind=f"workflow-stage-{stage_node.phase}",
            run_id=f"stage-{stage_node.id:02d}",
        )

        # Invoke AgentRunner
        # Note: no tracker is passed because a synthetic issue (stage-03) is not a real tracker issue;
        # tracker.fetch_issue_states_by_ids returns 400 for invalid IDs.
        # agent_runner already handles multi-round continuation with tracker=None.
        try:
            await self._agent_runner.run(
                session=session,
                workflow=self._workflow_config,
                tracker=None,
                status_dashboard=self._status_dashboard,
                clarification_resolver=self._clarification_resolver,
                progress_reporter=self._progress_reporter,
                diagnostics_callback=self._diagnostics_callback,
            )
        except Exception as exc:
            logger.exception("AgentRunner.run failed for stage %s", stage_node.id)
            session.status = "failed"
            session.output_text = str(exc)

        return session

    # -- GATE handling --────────────────────────────────────────────────

    async def _run_auto_gate(self, stage_node: StageNode, state: WorkflowState) -> GateRunResult:
        """Auto GATE: judged from validator results."""
        if not stage_node.validators:
            return GateRunResult(stage_id=stage_node.id, approved=True, reason="no validators, auto-approved")

        results = await self._validator.validate_all(stage_node.validators)
        all_passed = all(r.passed for r in results)

        return GateRunResult(
            stage_id=stage_node.id,
            approved=all_passed,
            reason="All validators passed"
            if all_passed
            else f"Validators failed: {[r.message for r in results if not r.passed]}",
        )

    async def _run_threshold_gate(self, stage_node: StageNode, state: WorkflowState) -> GateRunResult:
        """Threshold GATE: judged via synthetic issue + LLM scoring."""
        try:
            prompt = (
                f"Evaluate the following work output and assign a score from 0.0 to 1.0.\n"
                f"Respond with ONLY: score: <number>\n\n"
                f"Stage: {stage_node.name}\n"
                f"Prompt: {stage_node.prompt}\n"
            )
            session = await self._run_synthetic_issue(prompt=prompt, stage_node=stage_node)
            output_text = session.output_text if session else ""
            score = self._extract_score(output_text)
            approved = score >= stage_node.gate_threshold

            return GateRunResult(
                stage_id=stage_node.id,
                approved=approved,
                reason=f"Score {score:.2f} >= threshold {stage_node.gate_threshold}"
                if approved
                else f"Score {score:.2f} < threshold {stage_node.gate_threshold}",
            )
        except Exception as exc:
            return GateRunResult(stage_id=stage_node.id, approved=False, reason=f"Threshold gate error: {exc}")

    # -- DECISION handling --────────────────────────────────────────────

    def _build_decision_prompt(self, stage_node: StageNode, state: WorkflowState) -> str:
        """Build the decision-stage prompt."""
        outcomes = list(stage_node.decision_outcomes.keys())
        return (
            f"Based on the completed stages, decide the next action.\n"
            f"Available outcomes: {', '.join(outcomes)}\n"
            f"Respond with ONE word: the chosen outcome.\n\n"
            f"Stage: {stage_node.name}\n"
            f"Context: {stage_node.prompt}\n"
        )

    def _parse_decision_outcome(self, output_text: str, stage_node: StageNode) -> str:
        """Parse a decision result from LLM output."""
        import re

        text_lower = output_text.strip().lower()
        outcomes = list(stage_node.decision_outcomes.keys())
        for outcome in outcomes:
            if re.search(rf"\b{re.escape(outcome.lower())}\b", text_lower):
                return outcome
        return "failed"

    def _resolve_next_stage(self, outcome: str, stage_node: StageNode) -> int | None:
        """Resolve the next stage from the decision outcome."""
        decision_spec = stage_node.decision_outcomes.get(outcome, {})
        return decision_spec.get("next")

    # -- Prompt building --───────────────────────────────────────────────

    def _build_stage_prompt(self, stage_node: StageNode, state: WorkflowState) -> str:
        """Build a stage prompt.

        Structure:
        1. WORKFLOW.md template (via PromptBuilder.render; includes project context, coding conventions, etc.)
        2. Current stage instruction (stage_node.prompt)
        3. Previous stage outputs (for downstream stages)
        4. Output verification requirements
        """
        parts = []

        # 1. Base WORKFLOW.md prompt (project context, coding conventions, implementation approach, etc.)
        base_prompt = self._render_base_prompt(state)
        if base_prompt:
            parts.append(base_prompt)

        # 2. Current stage instruction
        parts.append(f"\n## Current Stage: {stage_node.name}")
        parts.append(f"Phase: {stage_node.phase}")
        stage_agent = (stage_node.agent_config or {}).get("agent")
        if stage_agent:
            parts.append(
                f"\n## Assigned Stage Agent\n"
                f"Execute this stage as sub-agent `@{stage_agent}` via the Agent tool. "
                f"That agent owns the stage skill, tools, and bridge dispatch for this step."
            )
        parts.append(stage_node.prompt or f"Execute stage: {stage_node.name}")

        # Git constraints: commit allowed on the issue branch; push and branch operations forbidden
        # git_sync pushes and creates the PR once the workflow completes
        parts.append(
            "\n## ⚠️ Git Constraints\n"
            "- You may use `git add` and `git commit` on the current branch.\n"
            "- Do NOT run `git push` -- the orchestrator handles push and PR creation.\n"
            "- Do NOT run `git checkout`, `git switch`, or create new branches.\n"
            "- Do NOT create pull requests.\n"
            "- Use Write / Edit tools to modify source files."
        )

        # 3. Previous stage outputs
        if state.completed_stages:
            parts.append("\n## Completed Stages")
            for sid in state.completed_stages:
                sresult = state.get_stage_result(sid)
                if sresult:
                    parts.append(f"- Stage {sid}: {sresult.status.value}")
                    if sresult.outputs:
                        parts.append(f"  Output: {sresult.outputs[0][:50000]}")

        # 4. Output verification requirements
        if stage_node.validators:
            parts.append("\n## Output Requirements")
            for v in stage_node.validators:
                parts.append(f"- {v.get('type', 'unknown')}: {v}")

        prompt = "\n".join(parts)

        # Resolve cross-stage artifact references
        prompt = ArtifactResolver.resolve(
            prompt,
            state=state,
            workspace_dir=str(self._workspace_dir) if self._workspace_dir else "",
        )

        return prompt

    def _render_base_prompt(self, state: WorkflowState) -> str:
        """Obtain the base WORKFLOW.md prompt via PromptBuilder.render.

        Uses the raw issue object (stored in state.issue_context['_issue'])
        to render the WORKFLOW.md template, preserving project context, coding conventions, etc.
        """
        issue = state.issue_context.get("_issue") if state.issue_context else None
        if issue is None:
            # Fallback: without an issue object, build a simple prompt from issue_context
            if state.issue_context:
                parts = ["## Issue Context"]
                parts.append(f"Title: {state.issue_context.get('title', 'N/A')}")
                desc = state.issue_context.get("description", "")
                if desc:
                    parts.append(f"Description: {desc}")
                return "\n".join(parts)
            return ""

        try:
            from ..prompt_builder import PromptBuilder

            return PromptBuilder.render(issue=issue)
        except Exception as exc:
            logger.warning("PromptBuilder.render failed, using fallback: %s", exc)
            # Fallback: use the issue's title + description directly
            title = getattr(issue, "title", "") or state.issue_context.get("title", "")
            desc = getattr(issue, "description", "") or state.issue_context.get("description", "")
            return f"## Issue: {title}\n\n{desc}"

    # -- Utility methods --─────────────────────────────────────────────────

    @staticmethod
    def _get_total_cost_usd() -> float:
        """Get the current accumulated cost from core bootstrap state."""
        try:
            from src.bootstrap.state import get_total_cost_usd

            return get_total_cost_usd()
        except Exception as exc:
            logger.error("Failed to read accumulated cost from bootstrap state: %s", exc)
            raise

    @staticmethod
    def _extract_score(text: str) -> float:
        """Extract a score (0.0-1.0) from LLM output."""
        import re

        match = re.search(r"(?:score)[:\s]*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(score, 1.0))
            except ValueError:
                pass
        return 0.0
