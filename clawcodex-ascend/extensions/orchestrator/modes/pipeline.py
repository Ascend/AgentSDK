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

"""Pipeline mode — run an issue across N sequential agent stages.

A pipeline stage is one full ``AgentRunner.run`` invocation on the SAME
``AgentSession`` / workspace, scoped to a single named responsibility
(e.g. analyzer, implementer, tester). Between stages we:

* Reset the per-run scalars on the session (``turn_count``, ``status``,
  ``output_text``, ``session_end_reason``, ``session_end_summary``) so
  the next stage gets its full ``max_turns`` budget and isn't seen as
  "already completed".
* Force a fresh ``run_id`` so each stage's transcript ends up in its
  own ``~/.clawcodex/tool-events/<run_id>/events.ndjson`` directory —
  this is what gives ops people separate audit trails per stage.
* Reset the 429 backoff counters so a rate-limit episode in stage 1
  doesn't doom stage 2 to skip its first turn.

The previous stage's tail-truncated output is injected into the next
stage's ``prompt_override`` as handoff context. We keep the tail (last
N chars) instead of a head/middle slice because LLM outputs typically
end with the actionable conclusion / summary; older context that the
model would still need is recoverable from the workspace files.

This implementation runs stages SERIALLY on the same workspace. For
truly parallel pipeline stages — useful when stages produce independent
side outputs — see ``CoordinatorModeRunner`` in Phase 3.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent_runner import AgentRunner, AgentSession
    from ..config.schema import WorkflowConfig

logger = logging.getLogger(__name__)


# Auto-managed team file for mailbox handoff. Created at pipeline start
# (when handoff=mailbox), torn down by orchestrator workspace cleanup.
_PIPELINE_TEAM_FILE: str = ".clawcodex/team.json"
_PIPELINE_TEAM_NAME: str = "pipeline-team"
_PIPELINE_MAILBOXES_DIR: str = ".clawcodex/mailboxes"


# Handoff via a real file in the workspace beats string-tail tricks:
#   1. Survives stage restarts and inspection between stages.
#   2. Next stage can ``Read`` it like any other file — no special prompt
#      slot management.
#   3. Auditable: the plan is in git history if the stage commits it.
# The previous prompt's tail is still inlined as a fallback for
# robustness (in case the stage refused to write the plan file), but
# the file is the load-bearing channel.
_PIPELINE_PLAN_FILE: str = ".clawcodex/pipeline-plan.md"

# Tail size used only when the plan file is missing (degraded mode).
_PRIOR_OUTPUT_TAIL_CHARS: int = 2000


# Stage prompt templates. Kept inline (not in a config file) because
# they're load-bearing for correctness; changing them is a behavioral
# change that should land in a code review, not a YAML edit.
_STAGE_PROMPTS: dict[str, str] = {
    "analyzer": (
        "You are the **ANALYZER** stage of a Pipeline workflow.\n\n"
        "Your scope:\n"
        "1. Read the relevant files to understand the issue.\n"
        "2. Identify which files need changes and what the changes should look like.\n"
        "3. **Write** your plan to the file `" + _PIPELINE_PLAN_FILE + "` "
        "(create the `.clawcodex/` directory if it doesn't exist) using "
        "the structured template below. The IMPLEMENTER stage will Read "
        "this file as its main handoff input.\n\n"
        "Plan template (paste verbatim, then fill in the fields):\n"
        "```markdown\n"
        "# Pipeline Plan\n\n"
        "## Files to change\n"
        "- `path/to/file1.py` — what to change here\n"
        "- `path/to/file2.py` — what to change here\n\n"
        "## Approach\n"
        "1. Step 1...\n"
        "2. Step 2...\n\n"
        "## Validation\n"
        "How TESTER should verify (which tests / commands).\n"
        "```\n\n"
        "Do **NOT** modify any other code in this stage. Reads + write the plan file only.\n\n"
        "End your final response with `[ANALYZER DONE]`."
    ),
    "implementer": (
        "You are the **IMPLEMENTER** stage of a Pipeline workflow.\n\n"
        "ANALYZER's plan lives at `" + _PIPELINE_PLAN_FILE + "` in the workspace. "
        "**Read it first** — it tells you which files to change and how.\n\n"
        "Reference (last stage's chat output, in case the plan file is missing):\n\n{prior}\n\n"
        "Your scope:\n"
        "1. Read `" + _PIPELINE_PLAN_FILE + "` to get the plan.\n"
        "2. Apply the changes the plan calls for.\n"
        "3. Keep edits minimal — don't refactor adjacent code unless the plan asked for it.\n"
        "4. Output a brief summary of what you changed.\n\n"
        "End your final response with `[IMPLEMENTER DONE]`."
    ),
    "tester": (
        "You are the **TESTER** stage of a Pipeline workflow.\n\n"
        "ANALYZER's plan lives at `" + _PIPELINE_PLAN_FILE + "`. The **## Validation** "
        "section tells you how to verify the implementation.\n\n"
        "Reference (last stage's chat output):\n\n{prior}\n\n"
        "Your scope:\n"
        "1. Read the plan file's Validation section.\n"
        "2. Run the validation commands / tests.\n"
        "3. If tests fail, fix obvious issues — for non-trivial code changes, leave a clear failure summary and stop.\n"
        "4. Output a brief verification summary.\n\n"
        "End your final response with `[TESTER DONE]`."
    ),
}


_DEFAULT_STAGES: tuple[str, ...] = ("analyzer", "implementer", "tester")


@dataclass
class _StageResult:
    """One stage's outcome — kept in memory across the pipeline run."""

    stage: str
    status: str
    output: str  # tail-truncated for handoff


class PipelineModeRunner:
    """Run an issue across multiple sequential AgentRunner stages.

    Multi-agent depth knobs:
    * ``stage_models``  — per-stage LLM. Heterogeneous brains, not just
      role labels (e.g. analyzer=<strong-model>, implementer=<fast-model>).
    * ``handoff``       — ``"prompt"`` (legacy: inject prior in next
      stage's prompt) or ``"mailbox"`` (each stage SendMessage to next;
      next Reads its inbox first — same protocol as Coordinator).
    """

    _ALLOWED_NESTED_KINDS: frozenset[str] = frozenset({"agent", "debate", "coordinator"})

    def __init__(
        self,
        agent_runner: "AgentRunner",
        *,
        stages: tuple[str, ...] = _DEFAULT_STAGES,
        max_retries_per_stage: int = 1,
        stage_models: dict[str, str] | None = None,
        stage_max_turns: dict[str, int] | None = None,
        stage_specs: dict[str, dict[str, Any]] | None = None,
        handoff: str = "prompt",
    ) -> None:
        if not stages:
            raise ValueError("PipelineModeRunner requires at least one stage")
        if max_retries_per_stage < 0:
            raise ValueError("max_retries_per_stage must be >= 0")
        if handoff not in {"prompt", "mailbox"}:
            raise ValueError(f"handoff must be 'prompt' or 'mailbox', got {handoff!r}")
        self._agent_runner = agent_runner
        self._stages: tuple[str, ...] = tuple(stages)
        self._max_retries_per_stage = max_retries_per_stage
        # Filter empty-string models (e.g. ``{"analyzer": ""}``) — an
        # empty model id would cause AgentRunner to send a broken request.
        # The schema parser already filters these, but direct constructor
        # calls (tests, shell scripts) can bypass the parser.
        self._stage_models: dict[str, str] = {k: v for k, v in (stage_models or {}).items() if v and v.strip()}
        # Warn about keys that don't correspond to any stage — silent
        # no-op is the classic "why isn't my config working" trap.
        unknown_stages = set(self._stage_models) - set(self._stages)
        if unknown_stages:
            logger.warning(
                "PipelineModeRunner: stage_models has unknown keys %s "
                "(known stages: %s) — these overrides will be IGNORED",
                sorted(unknown_stages),
                list(self._stages),
            )
        # Same shape as stage_models for per-stage max_turns overrides.
        self._stage_max_turns: dict[str, int] = {
            k: v for k, v in (stage_max_turns or {}).items() if isinstance(v, int) and v > 0
        }
        unknown_mt = set(self._stage_max_turns) - set(self._stages)
        if unknown_mt:
            logger.warning(
                "PipelineModeRunner: stage_max_turns has unknown keys %s "
                "(known stages: %s) — these overrides will be IGNORED",
                sorted(unknown_mt),
                list(self._stages),
            )
        # Nested-mode stage specs. Each entry: {kind, config}.
        # Validation at construction time so bad configs fail fast
        # instead of exploding mid-run.
        self._stage_specs: dict[str, dict[str, Any]] = {}
        for stage_name, spec in (stage_specs or {}).items():
            kind = str(spec.get("kind", "agent")).strip().lower()
            if kind not in self._ALLOWED_NESTED_KINDS:
                raise ValueError(
                    f"PipelineModeRunner: stage_specs[{stage_name!r}].kind="
                    f"{kind!r} not in {sorted(self._ALLOWED_NESTED_KINDS)}. "
                    "Nested pipeline is explicitly forbidden to avoid "
                    "the infinite-recursion trap."
                )
            cfg = dict(spec.get("config") or {})
            # kind=agent + non-empty config is a silent-no-op trap —
            # the operator wired something that would never fire.
            if kind == "agent" and cfg:
                logger.warning(
                    "PipelineModeRunner: stage_specs[%r] has kind=agent "
                    "but non-empty config=%s — config will be IGNORED. "
                    "Change kind to 'debate' or 'coordinator' to use it.",
                    stage_name,
                    cfg,
                )
            self._stage_specs[stage_name] = {"kind": kind, "config": cfg}
        unknown_specs = set(self._stage_specs) - set(self._stages)
        if unknown_specs:
            logger.warning(
                "PipelineModeRunner: stage_specs has unknown keys %s (known stages: %s) — these will be IGNORED",
                sorted(unknown_specs),
                list(self._stages),
            )
        # Eagerly instantiate + cache sub-runners so bad configs
        # (e.g. debate parallel=True + isolation=reset) fail at daemon
        # startup, not partway through the first pipeline issue's
        # implementer stage. Only cache for stages the runner will
        # actually execute (stage in stages + kind != agent).
        self._nested_runner_cache: dict[str, Any] = {}
        for stage_name, spec in self._stage_specs.items():
            if stage_name not in self._stages:
                continue  # already warned above
            if spec["kind"] == "agent":
                continue  # nothing to instantiate
            try:
                self._nested_runner_cache[stage_name] = self._build_nested_runner(spec)
            except Exception as exc:
                raise ValueError(
                    f"PipelineModeRunner: stage_specs[{stage_name!r}] "
                    f"({spec['kind']!r}) failed to construct with "
                    f"config={spec['config']}: {type(exc).__name__}: {exc}"
                ) from exc
        self._handoff = handoff

    @property
    def stages(self) -> tuple[str, ...]:
        return self._stages

    @property
    def max_retries_per_stage(self) -> int:
        return self._max_retries_per_stage

    @property
    def stage_models(self) -> dict[str, str]:
        return dict(self._stage_models)

    @property
    def stage_max_turns(self) -> dict[str, int]:
        return dict(self._stage_max_turns)

    @property
    def stage_specs(self) -> dict[str, dict[str, Any]]:
        # Deep-copy to prevent external mutation of internal state.
        return {k: {**v, "config": dict(v["config"])} for k, v in self._stage_specs.items()}

    @property
    def handoff(self) -> str:
        return self._handoff

    async def run(
        self,
        session: "AgentSession",
        workflow: "WorkflowConfig",
        **hooks: Any,
    ) -> list[_StageResult]:
        # If handoff is via mailbox, bootstrap the team.json so the
        # SendMessage tool inside each stage can write to the right
        # mailbox. The file lives in workspace/.clawcodex/team.json
        # and is read by headless.py at agent startup. Each stage's
        # CLAUDE_CODE_AGENT_NAME env is set per-stage in _run_stage.
        if self._handoff == "mailbox":
            self._ensure_team_file(session)

        prior: list[_StageResult] = []
        for stage in self._stages:
            result = await self._run_stage_with_retry(stage, prior, session, workflow, **hooks)
            prior.append(result)

            # A stage that exhausted its retries with a terminal failure
            # aborts the pipeline — pouring more tokens into a stage
            # whose predecessor produced no usable output is waste.
            if not self._stage_succeeded(result.status):
                logger.warning(
                    "Pipeline issue=%s aborting after stage=%s (terminal status=%s, exhausted %d retries)",
                    session.issue.id,
                    stage,
                    result.status,
                    self._max_retries_per_stage,
                )
                break

        # The outer orchestrator looks at session.status to decide
        # whether to git_sync. The final stage's status flows through
        # unchanged, which is the behavior we want.
        return prior

    async def _run_stage_with_retry(
        self,
        stage: str,
        prior: list[_StageResult],
        session: "AgentSession",
        workflow: "WorkflowConfig",
        **hooks: Any,
    ) -> _StageResult:
        """Run one stage; on terminal failure, retry up to N times.

        Each retry adds a brief "previous attempt failed: <reason>"
        note to the prompt so the model has context for what went
        wrong and can try a different approach.
        """
        # Mailbox handoff: re-ensure team.json before EVERY stage attempt.
        # Live e2e on h144 caught this: git operations between stages
        # (checkout / reset) can restore an older team.json committed
        # to the repo, silently pointing SendMessage at the wrong team.
        # Idempotent re-write eliminates that failure mode.
        if self._handoff == "mailbox":
            self._ensure_team_file(session)

        last_attempt_note = ""
        for attempt in range(self._max_retries_per_stage + 1):
            self._reset_session_for_next_stage(session)
            session.prompt_override = self._build_stage_prompt(stage, prior, session, retry_note=last_attempt_note)
            session.run_kind = f"pipeline:{stage}" if attempt == 0 else f"pipeline:{stage}:retry{attempt}"
            stage_model = self._stage_models.get(stage)
            nested_spec = self._stage_specs.get(stage)
            stage_kind = nested_spec["kind"] if nested_spec else "agent"
            # handoff mode only applies to plain-agent stages; nested
            # sub-runners manage their own handoff (or none at all).
            # Log accordingly so operators aren't misled.
            handoff_label = self._handoff if stage_kind == "agent" else f"(delegated to {stage_kind})"
            logger.info(
                "Pipeline issue=%s stage=%s starting (run_kind=%s, attempt=%d/%d, kind=%s, model=%s, handoff=%s)",
                session.issue.id,
                stage,
                session.run_kind,
                attempt + 1,
                self._max_retries_per_stage + 1,
                stage_kind,
                stage_model or "(workflow default)",
                handoff_label,
            )
            with self._stage_overrides(workflow, stage, stage_model):
                if stage_kind == "agent":
                    await self._agent_runner.run(session, workflow, **hooks)
                else:
                    # Nested mode dispatch — instantiate a sub-ModeRunner
                    # per stage per attempt (cheap; no shared state
                    # concerns because the sub-runner uses `session`
                    # directly and any workspace-path swap it does is
                    # restored in its own finally).
                    sub_runner = self._make_nested_runner(
                        stage,
                        nested_spec,  # type: ignore[arg-type]
                    )
                    await sub_runner.run(session, workflow, **hooks)

            tail = (session.output_text or "")[-_PRIOR_OUTPUT_TAIL_CHARS:]
            self._log_plan_file_status(session, stage)
            logger.info(
                "Pipeline issue=%s stage=%s finished (status=%s, attempt=%d)",
                session.issue.id,
                stage,
                session.status,
                attempt + 1,
            )

            if self._stage_succeeded(session.status):
                return _StageResult(stage=stage, status=session.status, output=tail)

            # Terminal failure — record reason for the next retry prompt.
            last_attempt_note = (
                f"Attempt {attempt + 1} of this stage exited with "
                f"status={session.status!r}. Try a different angle: "
                f"if you got stuck reading, switch to writing the plan/edit; "
                f"if you got stuck editing, narrow the scope."
            )
            if attempt < self._max_retries_per_stage:
                logger.warning(
                    "Pipeline issue=%s stage=%s attempt=%d failed (status=%s) — retrying",
                    session.issue.id,
                    stage,
                    attempt + 1,
                    session.status,
                )

        # All retries exhausted — surface the final failure.
        return _StageResult(
            stage=stage,
            status=session.status,
            output=(session.output_text or "")[-_PRIOR_OUTPUT_TAIL_CHARS:],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _log_plan_file_status(session: "AgentSession", stage: str) -> None:
        """After a stage finishes, log whether the plan file is on disk.

        Helps operators tell whether the structured handoff actually
        worked. The next stage can fall back to the chat-output tail
        if the plan file is missing, but the file is preferred.
        """
        try:
            workspace_path = getattr(session.workspace, "path", None)
            if workspace_path is None:
                return
            plan_path = Path(workspace_path) / _PIPELINE_PLAN_FILE
            if plan_path.exists() and plan_path.stat().st_size > 0:
                logger.info(
                    "Pipeline issue=%s stage=%s wrote plan file (%d bytes)",
                    session.issue.id,
                    stage,
                    plan_path.stat().st_size,
                )
            elif stage == "analyzer":
                logger.warning(
                    "Pipeline issue=%s analyzer did NOT write %s — next "
                    "stage will fall back to chat-output tail (degraded)",
                    session.issue.id,
                    _PIPELINE_PLAN_FILE,
                )
        except Exception:  # pragma: no cover — defensive
            pass  # nosec B110 — best-effort analyzer output check

    # ------------------------------------------------------------------
    # Per-stage model + CLAUDE_CODE_AGENT_NAME swap (try/finally)
    # ------------------------------------------------------------------

    class _StageOverride:
        """Context manager — swaps workflow.agent.model, agent_runner.max_turns,
        and ``CLAUDE_CODE_AGENT_NAME`` env for one stage, restores on exit.

        The 3 swaps are independently opt-in:

        * ``stage_model`` — only when non-empty **AND different from the
          current value** (avoids noisy "overriding X → X" no-op log lines).
        * ``stage_max_turns`` — only when configured and > 0. Mutates
          ``agent_runner.max_turns`` directly because AgentRunner captures
          max_turns at __init__ from agent_config (not per-run).
        * ``CLAUDE_CODE_AGENT_NAME`` — only when ``set_agent_name=True``
          (mailbox handoff needs it; prompt-only handoff shouldn't touch
          the shared env because other issues on the same daemon might
          be using it).

        Each swap tracks its own ``_touched_*`` flag so ``__exit__`` only
        restores what ``__enter__`` actually changed.
        """

        def __init__(
            self,
            workflow: Any,
            agent_runner: Any,
            stage_name: str,
            *,
            stage_model: str | None = None,
            stage_max_turns: int | None = None,
            set_agent_name: bool = False,
        ) -> None:
            self._workflow = workflow
            self._agent_runner = agent_runner
            self._stage_name = stage_name
            self._stage_model = stage_model
            self._stage_max_turns = stage_max_turns
            self._set_agent_name = set_agent_name
            self._original_model: Any = None
            self._original_max_turns: int | None = None
            self._original_agent_name: str | None = None
            self._touched_model = False
            self._touched_max_turns = False
            self._touched_agent_name = False

        def __enter__(self) -> "PipelineModeRunner._StageOverride":
            # Identity for SendMessage — only set when mailbox handoff
            # is active so prompt-only pipelines don't leak env and
            # concurrent-issue daemons don't get cross-issue pollution.
            if self._set_agent_name:
                self._original_agent_name = os.environ.get("CLAUDE_CODE_AGENT_NAME")
                os.environ["CLAUDE_CODE_AGENT_NAME"] = self._stage_name
                self._touched_agent_name = True
            # Model swap — only when configured AND actually different.
            # This eliminates the noisy "overriding X → X" no-op log
            # for stages whose configured model happens to equal the
            # workflow default.
            if self._stage_model:
                agent_cfg = getattr(self._workflow, "agent", None)
                if agent_cfg is not None and hasattr(agent_cfg, "model"):
                    current_model = agent_cfg.model
                    if current_model != self._stage_model:
                        self._original_model = current_model
                        agent_cfg.model = self._stage_model
                        self._touched_model = True
                        logger.info(
                            "Pipeline stage=%s: temporarily overriding workflow.agent.model %s → %s",
                            self._stage_name,
                            self._original_model,
                            self._stage_model,
                        )
            # max_turns swap — mutate agent_runner directly (AgentRunner
            # captures max_turns at __init__ from agent_config, not
            # per-run, so we can't just set agent_config.max_turns).
            if self._stage_max_turns is not None and self._stage_max_turns > 0:
                current_mt = getattr(self._agent_runner, "max_turns", None)
                if current_mt != self._stage_max_turns:
                    self._original_max_turns = current_mt
                    self._agent_runner.max_turns = self._stage_max_turns
                    self._touched_max_turns = True
                    logger.info(
                        "Pipeline stage=%s: temporarily overriding agent_runner.max_turns %s → %s",
                        self._stage_name,
                        self._original_max_turns,
                        self._stage_max_turns,
                    )
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self._touched_model:
                agent_cfg = getattr(self._workflow, "agent", None)
                if agent_cfg is not None:
                    agent_cfg.model = self._original_model
                    logger.info(
                        "Pipeline stage=%s: restored workflow.agent.model to %s",
                        self._stage_name,
                        self._original_model,
                    )
            if self._touched_max_turns:
                self._agent_runner.max_turns = self._original_max_turns
                logger.info(
                    "Pipeline stage=%s: restored agent_runner.max_turns to %s",
                    self._stage_name,
                    self._original_max_turns,
                )
            if self._touched_agent_name:
                if self._original_agent_name is None:
                    os.environ.pop("CLAUDE_CODE_AGENT_NAME", None)
                else:
                    os.environ["CLAUDE_CODE_AGENT_NAME"] = self._original_agent_name

    def _stage_overrides(
        self,
        workflow: Any,
        stage: str,
        stage_model: str | None,
    ) -> "PipelineModeRunner._StageOverride":
        return PipelineModeRunner._StageOverride(
            workflow,
            self._agent_runner,
            stage,
            stage_model=stage_model,
            stage_max_turns=self._stage_max_turns.get(stage),
            set_agent_name=(self._handoff == "mailbox"),
        )

    # ------------------------------------------------------------------
    # Mailbox handoff bootstrap (team.json auto-create)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Nested-mode dispatch (hierarchical agents)
    # ------------------------------------------------------------------

    def _make_nested_runner(self, stage: str, spec: dict[str, Any]) -> Any:
        """Return the cached sub-ModeRunner for ``stage``.

        Sub-runners are constructed once at Pipeline __init__ time
        (see ``_nested_runner_cache``) so config errors surface at
        daemon startup, not partway through an issue run. Retries of
        the same stage re-use the same sub-runner instance — this is
        safe because ``ModeRunner.run`` is stateless per invocation.
        """
        cached = self._nested_runner_cache.get(stage)
        if cached is not None:
            return cached
        # Defensive path: the cache was never populated for this stage
        # (shouldn't happen given __init__ validation). Fall back to
        # a fresh build so we don't crash outright.
        logger.warning(
            "PipelineModeRunner: nested runner for stage=%r not in cache — "
            "constructing on demand (indicates a bug in cache init)",
            stage,
        )
        return self._build_nested_runner(spec)

    def _build_nested_runner(self, spec: dict[str, Any]) -> Any:
        """Concrete constructor for a sub-ModeRunner. Called at init
        time (once per nested stage) + as a defensive fallback.

        Local imports here avoid a circular init between the modes
        package and the individual mode modules.
        """
        kind = spec["kind"]
        config = spec["config"]
        if kind == "debate":
            from .debate import DebateModeRunner

            return DebateModeRunner(self._agent_runner, **config)
        if kind == "coordinator":
            from .coordinator import CoordinatorModeRunner

            if config:
                logger.warning(
                    "Pipeline nested coordinator stage received config=%s "
                    "but CoordinatorModeRunner takes no per-stage params "
                    "— config IGNORED.",
                    config,
                )
            return CoordinatorModeRunner(self._agent_runner)
        raise ValueError(f"unhandled nested stage kind={kind!r}")

    def _ensure_team_file(self, session: "AgentSession") -> None:
        """Auto-write ``.clawcodex/team.json`` listing all pipeline stages
        as team members so SendMessage can find their mailboxes.
        """
        workspace_path = getattr(session.workspace, "path", None)
        if workspace_path is None:
            return
        team_path = Path(workspace_path) / _PIPELINE_TEAM_FILE
        team_data = {
            "team_name": _PIPELINE_TEAM_NAME,
            "lead_agent_id": self._stages[0],
            "members": [{"agent_id": s, "name": s, "role": s} for s in self._stages],
        }
        try:
            team_path.parent.mkdir(parents=True, exist_ok=True)
            team_path.write_text(json.dumps(team_data, indent=2), encoding="utf-8")
            logger.info(
                "Pipeline issue=%s wrote %s for mailbox handoff (%d members: %s)",
                session.issue.id,
                _PIPELINE_TEAM_FILE,
                len(self._stages),
                self._stages,
            )
        except Exception:
            logger.exception(
                "Pipeline issue=%s failed to write %s — mailbox handoff will degrade to prompt-only",
                session.issue.id,
                _PIPELINE_TEAM_FILE,
            )

    @staticmethod
    def _reset_session_for_next_stage(session: "AgentSession") -> None:
        session.turn_count = 0
        session.status = "running"
        session.output_text = ""
        session.session_end_reason = None
        session.session_end_summary = ""
        # Force a brand-new run_id for each stage so transcripts /
        # tool-events end up in separate per-stage directories.
        session.run_id = None
        # Reset 429 bookkeeping so a prior stage's rate-limit episode
        # doesn't leak into the next stage's first turn.
        session.consecutive_429_count = 0
        session.rate_limit_pending_turn = None

    @staticmethod
    def _stage_succeeded(status: str) -> bool:
        # ``completed`` is the happy path. We also tolerate:
        # - ``max_turns_exceeded``: stage made progress before hitting cap
        # - ``running``: still in progress (rare; defensive)
        # - ``read_only_loop``: BY DESIGN for analyzer-style stages that
        #   only read + plan. Without this, an analyzer that finishes
        #   correctly (just produces a text plan, no edits) gets
        #   misclassified as a failure and aborts the whole pipeline.
        # Anything else (``stagnation``, ``loop_detected``, ``failed``,
        # ``cancelled``, ``paused``) remains a hard stop.
        return status in {
            "completed",
            "max_turns_exceeded",
            "running",
            "read_only_loop",
        }

    def _build_stage_prompt(
        self,
        stage: str,
        prior: list[_StageResult],
        session: "AgentSession",
        *,
        retry_note: str = "",
    ) -> str:
        template = _STAGE_PROMPTS.get(
            stage,
            (
                f"You are the **{stage.upper()}** stage of a Pipeline workflow.\n\n"
                "Prior stages' summaries:\n\n{prior}\n\n"
                "Complete this stage and end your final response with "
                f"`[{stage.upper()} DONE]`."
            ),
        )
        prior_block = self._format_prior(prior)
        body = template.format(prior=prior_block)
        sections = [body]
        if self._handoff == "mailbox":
            sections.append(self._mailbox_handoff_instructions(stage))
        if retry_note:
            sections.append(f"## Retry context\n\n{retry_note}")
        sections.append(self._format_issue_block(session))
        return "\n\n".join(sections)

    def _mailbox_handoff_instructions(self, stage: str) -> str:
        """Instructions appended when handoff=mailbox. Adds SendMessage /
        mailbox-Read steps so each stage's identity participates in the
        team protocol — same as Coordinator + Plan B+C work.
        """
        idx = self._stages.index(stage)
        is_first = idx == 0
        is_last = idx == len(self._stages) - 1
        next_stage = self._stages[idx + 1] if not is_last else None
        prev_stage = self._stages[idx - 1] if not is_first else None
        inbox_path = f"{_PIPELINE_MAILBOXES_DIR}/{_PIPELINE_TEAM_NAME}/{stage}.jsonl"

        lines = [f"## Mailbox handoff (you are agent **{stage}**)"]
        if not is_first:
            lines.append(
                f"- Your inbox: `{inbox_path}`. **Read it first** — the "
                f"previous stage ({prev_stage}) posted a structured "
                "handoff note there."
            )
        else:
            lines.append("- You are the first stage; your inbox is empty.")
        if not is_last:
            lines.append(
                f'- When done, call `SendMessage(to="{next_stage}", '
                f'message="<your handoff summary>")` so the next stage '
                "knows you're done and can read your context."
            )
        else:
            lines.append("- You are the last stage; no SendMessage needed.")
        return "\n".join(lines)

    @staticmethod
    def _format_prior(prior: list[_StageResult]) -> str:
        if not prior:
            return "(no prior stages — you are first)"
        chunks: list[str] = []
        for s in prior:
            chunks.append(f"### Stage: {s.stage} (final status: {s.status})\n{s.output}".strip())
        return "\n\n".join(chunks)

    @staticmethod
    def _format_issue_block(session: "AgentSession") -> str:
        issue = session.issue
        title = getattr(issue, "title", "") or ""
        body = getattr(issue, "description", "") or ""
        return f"## Issue\nTitle: {title}\n\n{body}".rstrip()


__all__ = ["PipelineModeRunner"]
