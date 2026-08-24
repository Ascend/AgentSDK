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
"""Debate mode: independent proposers followed by one judge.

Unlike a pipeline, proposer prompts deliberately omit other proposals.
The judge receives all proposals, selects or synthesizes a result, and
then performs the implementation. Session state is reset between stages
so each ``AgentRunner.run`` invocation has an independent transcript.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from extensions.orchestrator.agent_runner import AgentRunner, AgentSession
    from extensions.orchestrator.config.schema import WorkflowConfig

logger = logging.getLogger(__name__)


_DEFAULT_PROPOSERS: tuple[str, ...] = ("proposer_a", "proposer_b")
_JUDGE_STAGE: str = "judge"
_PROPOSER_OUTPUT_TAIL_CHARS: int = 3000
_VALID_ISOLATIONS: frozenset[str] = frozenset({"reset", "worktree", "none"})
_SAFE_NAME_RE: re.Pattern[str] = re.compile(r"[^a-zA-Z0-9_-]+")

IsolationMode = Literal["reset", "worktree", "none"]
JudgeMode = Literal["pick", "synthesize"]


# Proposer "lenses" — different angles each proposer is forced to argue
# from. Without this, every proposer runs with an identical prompt and a
# zero-temperature deepseek model returns nearly-identical proposals,
# making the "debate" theatre rather than substance. Lenses make sure
# the judge actually sees contrasting takes.
#
# Auto-assigned by proposer index. The Nth proposer in
# ``modes.debate.proposers`` gets ``_PROPOSER_LENSES[N % len]``.
# Cycling is safe — even with 5+ proposers each still gets *some*
# differentiating angle.
_PROPOSER_LENSES: tuple[tuple[str, str], ...] = (
    (
        "simplicity-first",
        "Prefer the simplest possible approach. Argue for the smallest "
        "diff, the fewest moving parts, the least new abstractions — "
        "even at some cost to flexibility or performance.",
    ),
    (
        "robustness-first",
        "Prefer the most robust approach. Argue for explicit error "
        "handling, defensive validation, clear failure modes, and "
        "easier debugging — even at some cost to terseness.",
    ),
    (
        "user-first",
        "Prefer the approach with the cleanest external API. Argue for "
        "discoverability, documentation, backward compatibility, and "
        "API ergonomics — even at some internal-implementation cost.",
    ),
    (
        "performance-first",
        "Prefer the most performant approach. Argue for fewer allocations, "
        "less work per call path, and lower memory pressure — even at "
        "some readability cost.",
    ),
)


_PROPOSER_PROMPT_TEMPLATE: str = (
    "You are **{name}**, one of {n} independent proposers in a Debate workflow.\n\n"
    "**Your assigned lens: {lens_name}**\n"
    "{lens_instruction}\n\n"
    "Your scope:\n"
    "1. Read the issue and relevant files.\n"
    "2. Design ONE coherent approach to solving the issue, viewed through your lens.\n"
    "3. Write a short proposal (5–10 bullets) describing your approach: "
    "files you'd change, key decisions, trade-offs. Make your lens visible "
    "in the proposal — the judge should be able to tell from your text "
    "that you argued from the **{lens_name}** angle.\n\n"
    "**Hard constraints:**\n"
    "- Do NOT modify any source code in this stage. Reads + written proposal only.\n"
    "- Do NOT read the team mailboxes — propose independently so the judge sees a genuinely different take.\n"
    "- Stay in your lens — even when another angle would also work, advocate for yours.\n\n"
    "End your final response with `[{tag}]`."
)


_JUDGE_PROMPT_PICK: str = (
    "You are the **JUDGE** in a Debate workflow.\n\n"
    "{proposals}\n\n"
    "Your scope (**PICK mode** — choose ONE proposer, no hybridization):\n"
    "1. Compare the proposals above on correctness, simplicity, and fit.\n"
    "2. Pick ONE proposer to win — name them explicitly and give a 2–4 sentence justification.\n"
    "3. Implement the winning proposal verbatim: edit the files, run tests if relevant.\n"
    "4. Output a brief summary of what you implemented.\n\n"
    "End your final response with `[JUDGE DONE]`."
)


_JUDGE_PROMPT_SYNTHESIZE: str = (
    "You are the **JUDGE** in a Debate workflow.\n\n"
    "{proposals}\n\n"
    "Your scope (**SYNTHESIZE mode** — combine best ideas from ALL proposers):\n"
    "1. Read every proposal above. Identify the concrete claims / choices "
    "each proposer made (2–5 bullets per proposal).\n"
    "2. For each claim, decide: KEEP (this proposer got it right), REJECT "
    "(their reasoning is weaker), or MERGE (take an idea from A and an idea "
    "from B that combine cleanly).\n"
    "3. Write your synthesized approach as a numbered plan of concrete "
    "steps — each step should CITE which proposer(s) contributed it, e.g. "
    '"1. Use lazy init (proposer_a) but wrap it in a threading.Lock '
    '(proposer_b) for concurrent access."\n'
    "4. Implement the synthesized plan: edit the files, run tests if relevant.\n"
    "5. Output a brief summary of what you implemented + which proposer "
    "contributed each material decision.\n\n"
    "End your final response with `[JUDGE DONE]`."
)


# Legacy alias — some existing tests reference _JUDGE_PROMPT_TEMPLATE.
# Points at the pick-mode template since that's the historical default.
_JUDGE_PROMPT_TEMPLATE: str = _JUDGE_PROMPT_PICK


@dataclass
class _StageResult:
    stage: str
    status: str
    output: str


class DebateModeRunner:
    """Run N proposer rounds + 1 judge round on the same session/workspace."""

    def __init__(
        self,
        agent_runner: "AgentRunner",
        *,
        proposers: tuple[str, ...] = _DEFAULT_PROPOSERS,
        judge_model: str | None = None,
        isolation: IsolationMode = "reset",
        proposer_models: dict[str, str] | None = None,
        parallel: bool = False,
        judge_mode: JudgeMode = "pick",
    ) -> None:
        if not proposers:
            raise ValueError("DebateModeRunner requires at least one proposer stage")
        if isolation not in _VALID_ISOLATIONS:
            raise ValueError(f"isolation must be one of {sorted(_VALID_ISOLATIONS)}, got {isolation!r}")
        if parallel and isolation != "worktree":
            raise ValueError(
                "parallel=True requires isolation='worktree' (each parallel branch needs its own physical workspace)"
            )
        if judge_mode not in {"pick", "synthesize"}:
            raise ValueError(f"judge_mode must be 'pick' or 'synthesize', got {judge_mode!r}")
        self._agent_runner = agent_runner
        self._proposers: tuple[str, ...] = tuple(proposers)
        self._judge_model = judge_model.strip() if judge_model else None
        self._isolation = isolation
        self._proposer_models: dict[str, str] = {
            k: v.strip() for k, v in (proposer_models or {}).items() if v and v.strip()
        }
        # Warn about keys that don't correspond to any proposer — silent
        # no-op is the classic "why isn't my config working" trap.
        unknown_proposers = set(self._proposer_models) - set(self._proposers)
        if unknown_proposers:
            logger.warning(
                "DebateModeRunner: proposer_models has unknown keys %s "
                "(known proposers: %s) — these overrides will be IGNORED",
                sorted(unknown_proposers),
                list(self._proposers),
            )
        self._parallel = parallel
        self._judge_mode = judge_mode

    @property
    def proposers(self) -> tuple[str, ...]:
        return self._proposers

    @property
    def judge_model(self) -> str | None:
        return self._judge_model

    @property
    def isolation(self) -> IsolationMode:
        return self._isolation

    @property
    def proposer_models(self) -> dict[str, str]:
        return dict(self._proposer_models)

    @property
    def parallel(self) -> bool:
        return self._parallel

    @property
    def judge_mode(self) -> JudgeMode:
        return self._judge_mode

    async def run(
        self,
        session: "AgentSession",
        workflow: "WorkflowConfig",
        **hooks: Any,
    ) -> list[_StageResult]:
        if self._isolation == "reset":
            self._require_clean_workspace(session)
        # Snapshot the workspace HEAD before any proposer runs so we can
        # reset between proposers + before judge. Without this, a
        # proposer that ignores its "no code edits" prompt would
        # contaminate the next proposer's view of the codebase.
        baseline_ref = self._snapshot_workspace_head(session)
        workspace_value = getattr(session.workspace, "path", None)
        original_workspace_path = Path(workspace_value) if workspace_value is not None else None

        logger.info(
            "Debate issue=%s isolation=%s judge_model=%s parallel=%s proposer_models=%s",
            session.issue.id,
            self._isolation,
            self._judge_model or "(default)",
            self._parallel,
            self._proposer_models or "(none)",
        )

        # ---- Proposer rounds ----
        if self._parallel:
            proposer_results = await self._run_proposers_parallel(session, workflow, baseline_ref, **hooks)
        else:
            proposer_results = await self._run_proposers_sequential(
                session, workflow, baseline_ref, original_workspace_path, **hooks
            )

        # If ANY proposer terminally failed, skip judge — there's nothing
        # to compare against.
        if any(not self._stage_succeeded(r.status) for r in proposer_results):
            session.status = "failed"
            session.session_end_reason = "debate_proposer_failed"
            session.session_end_summary = "At least one debate proposer failed before judging."
            logger.warning(
                "Debate issue=%s aborting before judge (at least one proposer hit terminal failure)",
                session.issue.id,
            )
            return proposer_results

        # ---- Judge round (sees ALL proposers' outputs) ----
        # Judge runs in the ORIGINAL workspace (not a worktree) so its
        # commit lands on the real branch the orchestrator will sync.
        # Reset isolation still resets the original dir to baseline
        # before judge — judge starts clean and implements the winner.
        self._reset_session_for_next_stage(session)
        if self._isolation == "reset":
            self._reset_workspace_to(session, baseline_ref)
        session.prompt_override = self._build_judge_prompt(proposer_results, session)
        session.run_kind = f"debate:{_JUDGE_STAGE}"
        logger.info(
            "Debate issue=%s judge starting (over %d proposers, model=%s)",
            session.issue.id,
            len(proposer_results),
            self._judge_model or "(workflow default)",
        )
        judge_runner, judge_workflow = self._isolated_runtime(workflow, self._judge_model)
        await judge_runner.run(session, judge_workflow, **hooks)
        proposer_results.append(
            _StageResult(
                stage=_JUDGE_STAGE,
                status=session.status,
                output=(session.output_text or "")[-_PROPOSER_OUTPUT_TAIL_CHARS:],
            )
        )
        logger.info(
            "Debate issue=%s judge finished (status=%s)",
            session.issue.id,
            session.status,
        )
        return proposer_results

    async def _run_proposers_sequential(
        self,
        session: "AgentSession",
        workflow: "WorkflowConfig",
        baseline_ref: str | None,
        original_workspace_path: Path | None,
        **hooks: Any,
    ) -> list[_StageResult]:
        """One-at-a-time proposers, honoring per-proposer model overrides."""
        results: list[_StageResult] = []
        for index, name in enumerate(self._proposers):
            self._reset_session_for_next_stage(session)
            lens = self._lens_for_index(index)
            worktree_path = self._apply_isolation_before_stage(session, baseline_ref, stage_label=name)
            session.prompt_override = self._build_proposer_prompt(name, session, lens=lens)
            session.run_kind = f"debate:{name}"
            proposer_model = self._proposer_models.get(name)
            logger.info(
                "Debate issue=%s proposer=%s lens=%s starting (run_kind=%s, workspace=%s, model=%s)",
                session.issue.id,
                name,
                lens[0],
                session.run_kind,
                getattr(session.workspace, "path", None),
                proposer_model or "(workflow default)",
            )
            try:
                proposer_runner, proposer_workflow = self._isolated_runtime(
                    workflow,
                    proposer_model,
                )
                await proposer_runner.run(session, proposer_workflow, **hooks)
            finally:
                self._restore_workspace_after_stage(session, original_workspace_path, worktree_path)
            tail = (session.output_text or "")[-_PROPOSER_OUTPUT_TAIL_CHARS:]
            results.append(_StageResult(stage=name, status=session.status, output=tail))
            logger.info(
                "Debate issue=%s proposer=%s finished (status=%s)",
                session.issue.id,
                name,
                session.status,
            )
            if not self._stage_succeeded(session.status):
                # Sequential mode aborts immediately so we don't waste tokens.
                logger.warning(
                    "Debate issue=%s sequential abort: proposer=%s status=%s",
                    session.issue.id,
                    name,
                    session.status,
                )
                break
        return results

    async def _run_proposers_parallel(
        self,
        session: "AgentSession",
        workflow: "WorkflowConfig",
        baseline_ref: str | None,
        **hooks: Any,
    ) -> list[_StageResult]:
        """True parallel proposers via ``asyncio.gather``.

        Per proposer:
        * deep-copy the session (each parallel branch needs its own
          turn_count / status / output_text / run_id);
        * create a fresh git worktree (already required by the
          parallel/worktree precondition in __init__);
        * point the per-branch session.workspace.path at that worktree;
        * gather all branches concurrently.

        Each branch receives a shallow runner copy plus a copied agent
        configuration, so heterogeneous models never mutate shared state.
        """
        workspace_value = getattr(session.workspace, "path", None)
        original_workspace_path = Path(workspace_value) if workspace_value is not None else None
        branches: list[tuple[str, Any, Any]] = []  # (name, branch_session, worktree_path)

        # Set up one branch per proposer (workspace + worktree).
        for index, name in enumerate(self._proposers):
            branch_session = self._fork_session_for_branch(session, name)
            self._reset_session_for_next_stage(branch_session)
            lens = self._lens_for_index(index)
            worktree_path = self._create_worktree_and_swap(branch_session, baseline_ref, stage_label=name)
            branch_session.prompt_override = self._build_proposer_prompt(name, branch_session, lens=lens)
            branch_session.run_kind = f"debate:{name}"
            branches.append((name, branch_session, worktree_path))

        logger.info(
            "Debate issue=%s starting %d proposers in PARALLEL",
            session.issue.id,
            len(branches),
        )

        async def _run_one(name: str, branch_session: Any) -> _StageResult:
            branch_runner, branch_workflow = self._isolated_runtime(
                workflow,
                self._proposer_models.get(name),
            )
            await branch_runner.run(branch_session, branch_workflow, **hooks)
            tail = (branch_session.output_text or "")[-_PROPOSER_OUTPUT_TAIL_CHARS:]
            logger.info(
                "Debate issue=%s proposer=%s finished (status=%s)",
                session.issue.id,
                name,
                branch_session.status,
            )
            return _StageResult(stage=name, status=branch_session.status, output=tail)

        # return_exceptions=True keeps parallel semantics ROBUST — one
        # proposer's failure doesn't cancel the other + doesn't propagate
        # out as an unhandled exception. This matches the sequential
        # mode's per-proposer failure handling (which records a
        # _StageResult with status='failed' and aborts before judge).
        results_or_excs: list[Any] = []
        try:
            results_or_excs = await asyncio.gather(
                *(_run_one(name, bs) for name, bs, _ in branches),
                return_exceptions=True,
            )
        finally:
            # Tear down all worktrees regardless of outcome. Note we
            # pass the ORIGINAL session.workspace.path — do NOT stringify
            # None (str(None) == "None" would masquerade as a real path).
            for _name, _bs, wt_path in branches:
                if wt_path is not None:
                    self._remove_worktree(session, wt_path, original_workspace_path)

        # Convert per-branch exceptions to _StageResult(status='failed')
        # so the caller sees a uniform list-of-results and can decide
        # whether to skip judge (via the ANY-failed check upstream).
        results: list[_StageResult] = []
        for (name, _bs, _wt), item in zip(branches, results_or_excs):
            if isinstance(item, BaseException):
                logger.warning(
                    "Debate issue=%s proposer=%s raised %s: %s",
                    session.issue.id,
                    name,
                    type(item).__name__,
                    str(item)[:200],
                )
                results.append(
                    _StageResult(
                        stage=name,
                        status="failed",
                        output=f"{type(item).__name__}: {item}"[:_PROPOSER_OUTPUT_TAIL_CHARS],
                    )
                )
            else:
                results.append(item)
        return results

    @staticmethod
    def _fork_session_for_branch(session: "AgentSession", branch_name: str) -> Any:
        """Create a per-branch session for parallel execution.

        We can't share the original ``session`` across coroutines because
        ``AgentRunner.run`` mutates many fields concurrently. ``copy.copy``
        is too shallow — it shares the lists / dicts / asyncio primitives
        between branches and causes silent transcript / event corruption.

        This implementation explicitly:
        * shallow-copies the session (gets fresh refs for scalars);
        * shallow-copies the workspace (so workspace.path swap is per-branch);
        * resets per-run scalars (turn_count / status / output_text / run_id);
        * **gives each branch its own copies of mutable containers** so
          two coroutines writing transcripts don't trample each other;
        * **resets async primitives to None** so AgentRunner.run creates
          fresh per-branch ``event_queue`` / ``pause_resume_event`` /
          ``state_cache`` / ``_transcript_storage`` on entry.
        """
        try:
            branch = copy.copy(session)
        except Exception:
            branch = SimpleNamespace(**vars(session))
        # Per-branch workspace (path swap doesn't bleed into sibling).
        if hasattr(session, "workspace"):
            branch.workspace = copy.copy(session.workspace)
        # Per-run scalars.
        branch.turn_count = 0
        branch.status = "running"
        branch.output_text = ""
        branch.run_id = None
        branch.session_end_reason = None
        branch.session_end_summary = ""
        branch.consecutive_429_count = 0
        branch.rate_limit_pending_turn = None
        # CRITICAL: replace shared mutable containers so two coroutines
        # writing transcripts in parallel don't corrupt each other's state.
        if hasattr(branch, "_transcript_tool_uses"):
            branch._transcript_tool_uses = []
        if hasattr(branch, "_transcript_pending_results"):
            branch._transcript_pending_results = {}
        if hasattr(branch, "_transcript_result_order"):
            branch._transcript_result_order = []
        if hasattr(branch, "_transcript_asst_text"):
            branch._transcript_asst_text = ""
        # Async primitives — let AgentRunner re-create per branch.
        if hasattr(branch, "event_queue"):
            branch.event_queue = None
        if hasattr(branch, "pause_resume_event"):
            branch.pause_resume_event = None
        if hasattr(branch, "state_cache"):
            branch.state_cache = None
        if hasattr(branch, "_transcript_storage"):
            branch._transcript_storage = None
        return branch

    def _isolated_runtime(self, workflow: Any, model: str | None) -> tuple[Any, Any]:
        """Copy model-bearing objects so one issue cannot reroute another."""
        run_agent = copy.copy(self._agent_runner)
        run_workflow = copy.copy(workflow)
        base_agent = getattr(self._agent_runner, "agent_config", None)
        if base_agent is None:
            base_agent = getattr(workflow, "agent", None)
        run_config = copy.copy(base_agent) if base_agent is not None else None

        if run_config is not None and model:
            run_config.model = model
        if run_config is not None and hasattr(run_workflow, "agent"):
            run_workflow.agent = run_config
        if run_config is not None and hasattr(run_agent, "agent_config"):
            run_agent.agent_config = run_config
        return run_agent, run_workflow

    def _apply_isolation_before_stage(
        self,
        session: "AgentSession",
        baseline_ref: str | None,
        *,
        stage_label: str,
    ) -> Path | None:
        """Apply the configured isolation strategy.

        Returns the worktree path if one was created (so the caller can
        restore + remove it after the stage), else ``None``.
        """
        if self._isolation == "reset":
            self._reset_workspace_to(session, baseline_ref)
            return None
        if self._isolation == "worktree":
            return self._create_worktree_and_swap(session, baseline_ref, stage_label=stage_label)
        # "none" — no-op, full contamination tolerated.
        return None

    def _restore_workspace_after_stage(
        self,
        session: "AgentSession",
        original_path: Path | None,
        worktree_path: Path | None,
    ) -> None:
        """Restore session.workspace.path + tear down worktree if any."""
        if worktree_path is None:
            return
        try:
            session.workspace.path = original_path
        except Exception:  # pragma: no cover — defensive  # nosec B110
            pass
        self._remove_worktree(session, worktree_path, original_path)

    # ------------------------------------------------------------------
    # Worktree isolation helpers
    # ------------------------------------------------------------------

    def _create_worktree_and_swap(
        self,
        session: "AgentSession",
        baseline_ref: str | None,
        *,
        stage_label: str,
    ) -> Path | None:
        """Create a fresh git worktree at baseline_ref, swap session in.

        Falls back to reset-style isolation if worktree creation fails
        (e.g. non-git workspace, disk full) so the stage still runs.
        """
        original_path = getattr(session.workspace, "path", None)
        if original_path is None or baseline_ref is None:
            # Can't create a worktree without a git workspace; degrade
            # to reset semantics (which itself is a no-op when baseline
            # is None — matches "none" isolation).
            if baseline_ref is not None:
                self._reset_workspace_to(session, baseline_ref)
            return None

        safe_stage = _SAFE_NAME_RE.sub("_", stage_label)[:40] or "stage"
        issue_id = getattr(session.issue, "id", "issue") or "issue"
        safe_issue = _SAFE_NAME_RE.sub("_", issue_id)[:40]
        worktree_path = Path(original_path).parent / f".debate-worktree-{safe_issue}-{safe_stage}"
        # If a leftover worktree from a previous run exists, remove it.
        if worktree_path.exists():
            self._remove_worktree(session, worktree_path, original_path)
        try:
            r = subprocess.run(  # nosec B607
                [
                    "git",
                    "worktree",
                    "add",
                    "--detach",
                    str(worktree_path),
                    baseline_ref,
                ],
                cwd=str(original_path),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if r.returncode != 0:
                logger.warning(
                    "Debate issue=%s git worktree add failed (exit=%d, stderr=%s) — degrading to reset isolation",
                    session.issue.id,
                    r.returncode,
                    r.stderr.strip()[:200],
                )
                self._reset_workspace_to(session, baseline_ref)
                return None
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "Debate issue=%s git worktree add raised %s — degrading to reset isolation",
                session.issue.id,
                exc,
            )
            self._reset_workspace_to(session, baseline_ref)
            return None

        try:
            # Preserve the type of workspace.path: if it was a Path
            # originally, swap with a Path (downstream code does
            # path / "x" which only works for Path, not str).
            if isinstance(original_path, Path):
                session.workspace.path = worktree_path
            else:
                session.workspace.path = str(worktree_path)
        except Exception:  # pragma: no cover — defensive
            logger.exception(
                "Debate issue=%s could not swap session workspace path",
                session.issue.id,
            )
            self._remove_worktree(session, worktree_path, original_path)
            return None

        logger.info(
            "Debate issue=%s created worktree %s for stage=%s",
            session.issue.id,
            worktree_path,
            stage_label,
        )
        return worktree_path

    def _remove_worktree(
        self,
        session: "AgentSession",
        worktree_path: Path,
        original_path: Path | None,
    ) -> None:
        """Best-effort teardown of a per-stage worktree."""
        if original_path is None:
            return
        try:
            subprocess.run(  # nosec B607
                [
                    "git",
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree_path),
                ],
                cwd=str(original_path),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception:  # pragma: no cover — defensive  # nosec B110
            pass
        # Fallback hard-delete if git worktree remove left anything behind.
        if worktree_path.exists():
            try:
                shutil.rmtree(worktree_path, ignore_errors=True)
            except Exception:  # pragma: no cover — defensive  # nosec B110
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _require_clean_workspace(session: "AgentSession") -> None:
        """Refuse destructive reset isolation when work already exists."""
        workspace_path = getattr(session.workspace, "path", None)
        if workspace_path is None:
            return
        probe = subprocess.run(  # nosec B607
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode != 0:
            return
        result = subprocess.run(  # nosec B607
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "Debate issue=%s could not verify workspace cleanliness; reset isolation will be disabled",
                session.issue.id,
            )
            raise RuntimeError("cannot verify workspace is clean for reset isolation")
        dirty_entries = [line for line in result.stdout.splitlines() if line.strip()]
        if dirty_entries:
            preview = ", ".join(dirty_entries[:5])
            raise RuntimeError(
                f"reset isolation requires a clean workspace; commit or stash existing changes first ({preview})"
            )

    @classmethod
    def _lens_for_index(cls, index: int) -> tuple[str, str]:
        """Pick the lens for the Nth proposer. Cycles if N > len(lenses)."""
        return _PROPOSER_LENSES[index % len(_PROPOSER_LENSES)]

    @staticmethod
    def _snapshot_workspace_head(session: "AgentSession") -> str | None:
        """Record the current HEAD commit sha so we can reset later.

        Returns ``None`` (and skips later resets) if the workspace
        isn't a git repo — defensive so the runner stays useful in
        non-git workspaces (tests, local-tracker mode, etc.).
        """
        workspace_path = getattr(session.workspace, "path", None)
        if workspace_path is None:
            return None
        try:
            r = subprocess.run(  # nosec B607
                ["git", "rev-parse", "HEAD"],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if r.returncode != 0:
                logger.info(
                    "Debate issue=%s workspace is not a git repo "
                    "(git rev-parse HEAD exit=%d) — proceeding without reset",
                    session.issue.id,
                    r.returncode,
                )
                return None
            sha = r.stdout.strip()
            logger.info(
                "Debate issue=%s workspace baseline = %s",
                session.issue.id,
                sha[:12],
            )
            return sha
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "Debate issue=%s snapshot_workspace_head raised %s",
                session.issue.id,
                exc,
            )
            return None

    @staticmethod
    def _reset_workspace_to(session: "AgentSession", baseline_ref: str | None) -> None:
        """Hard-reset workspace to the baseline HEAD + clean untracked.

        No-op when ``baseline_ref`` is None (non-git workspace).
        Failures are logged but never raised — the run continues even
        if reset fails. Worst case is reduced isolation, not a crash.
        """
        if baseline_ref is None:
            return
        workspace_path = getattr(session.workspace, "path", None)
        if workspace_path is None:
            return
        try:
            reset_result = subprocess.run(  # nosec B607
                ["git", "reset", "--hard", baseline_ref],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if reset_result.returncode != 0:
                logger.warning(
                    "Debate issue=%s git reset failed; refusing to clean workspace: %s",
                    session.issue.id,
                    reset_result.stderr.strip()[:300],
                )
                return
            clean_result = subprocess.run(  # nosec B607
                ["git", "clean", "-fd"],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if clean_result.returncode != 0:
                logger.warning(
                    "Debate issue=%s git clean failed: %s",
                    session.issue.id,
                    clean_result.stderr.strip()[:300],
                )
                return
            logger.info(
                "Debate issue=%s workspace reset to baseline %s",
                session.issue.id,
                baseline_ref[:12],
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "Debate issue=%s reset_workspace_to(%s) raised %s",
                session.issue.id,
                baseline_ref[:12] if baseline_ref else baseline_ref,
                exc,
            )

    @staticmethod
    def _reset_session_for_next_stage(session: "AgentSession") -> None:
        session.turn_count = 0
        session.status = "running"
        session.output_text = ""
        session.session_end_reason = None
        session.session_end_summary = ""
        session.run_id = None  # force a fresh transcript per stage
        session.consecutive_429_count = 0
        session.rate_limit_pending_turn = None

    @staticmethod
    def _stage_succeeded(status: str) -> bool:
        # Same tolerance window as PipelineModeRunner — accept benign
        # non-terminal exits so a proposer that hit max_turns mid-design
        # still gets its partial output to the judge. ``read_only_loop``
        # is intentional for proposer stages (they are forbidden to
        # edit code, so finishing read-only is the success criterion).
        return status in {
            "completed",
            "max_turns_exceeded",
            "running",
            "read_only_loop",
        }

    def _build_proposer_prompt(
        self,
        name: str,
        session: "AgentSession",
        *,
        lens: tuple[str, str] | None = None,
    ) -> str:
        lens_name, lens_instruction = lens or _PROPOSER_LENSES[0]
        body = _PROPOSER_PROMPT_TEMPLATE.format(
            name=name,
            n=len(self._proposers),
            tag=f"{name.upper()} DONE",
            lens_name=lens_name,
            lens_instruction=lens_instruction,
        )
        return f"{body}\n\n{self._format_issue_block(session)}"

    def _build_judge_prompt(self, proposer_results: list[_StageResult], session: "AgentSession") -> str:
        chunks: list[str] = []
        for r in proposer_results:
            chunks.append(f"## Proposal from {r.stage} (final status: {r.status})\n\n{r.output}".strip())
        proposals_block = "\n\n".join(chunks) if chunks else "(no proposals were produced)"
        template = _JUDGE_PROMPT_SYNTHESIZE if self._judge_mode == "synthesize" else _JUDGE_PROMPT_PICK
        body = template.format(proposals=proposals_block)
        return f"{body}\n\n{self._format_issue_block(session)}"

    @staticmethod
    def _format_issue_block(session: "AgentSession") -> str:
        issue = session.issue
        title = getattr(issue, "title", "") or ""
        body = getattr(issue, "description", "") or ""
        return f"## Issue\nTitle: {title}\n\n{body}".rstrip()


__all__ = ["DebateModeRunner"]
