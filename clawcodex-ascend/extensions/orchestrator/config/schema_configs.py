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

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackerConfig:
    kind: str = "linear"
    endpoint: str = "https://api.linear.app/graphql"
    api_key: str | None = None
    project_slug: str | None = None
    owner: str | None = None
    repo: str | None = None
    clone_url: str | None = None
    assignee: str | None = None
    branch_prefix: str | None = None
    issues_path: str | None = None
    active_states: list[str] = field(default_factory=lambda: ["Todo", "In Progress"])
    terminal_states: list[str] = field(
        default_factory=lambda: [
            "Closed",
            "Cancelled",
            "Canceled",
            "Duplicate",
            "Done",
        ]
    )
    # Issues carrying any of these labels (case-insensitive) are
    # excluded from the candidate queue at fetch time. Use for
    # web-only workflow labels (e.g. "completed", "wontfix") that the
    # tracker's `state` field does not reflect as terminal. Empty
    # list = no exclusion.
    skip_labels: list[str] = field(default_factory=list)
    # Issues must carry at least ONE of these labels (OR semantics,
    # case-insensitive) to enter the candidate queue. Use to scope
    # the orchestrator to a particular class of work (e.g. only
    # `priority/high` or `priority/urgent`). Empty list = no
    # requirement. Evaluated before `skip_labels`.
    require_any_labels: list[str] = field(default_factory=list)
    # A candidate title must start with configured prefixes. ``any`` is
    # OR/union semantics and ``all`` is AND/intersection semantics. Empty
    # prefixes disable this filter.
    title_prefixes: list[str] = field(default_factory=list)
    title_prefix_match: str = "any"


@dataclass
class PollingConfig:
    interval_ms: int = 30_000


def _default_tmp_workspace() -> str:
    return os.path.join(os.environ.get("TMPDIR", "/tmp"), "symphony_workspaces")  # nosec B108


@dataclass
class WorkspaceConfig:
    root: str = field(default_factory=_default_tmp_workspace)
    hooks: dict[str, Any] = field(default_factory=dict)
    repo_clone_url: str | None = None
    # Fork workflow: upstream repo URL (PR target). Falls back to single-repo
    # mode when absent or equal to repo_clone_url.
    upstream_clone_url: str | None = None
    clone_depth: int | None = 1
    checkout_issue_branch: bool = True
    git_username: str | None = None
    git_email: str | None = None
    git_token: str | None = None
    gitignore_patterns: list[str] = field(default_factory=list)
    strategy: str = "isolated"
    base_branch: str | None = None
    integration_branch: str | None = None
    require_clean_start: bool = True
    require_clean_between_issues: bool = True
    preserve_on_terminal: bool = True
    # Conditional preservation: keep workspace for specific end-states so
    # users can inspect artifacts, re-run verification, or debug failures.
    preserve_on_failure: bool = True
    preserve_on_abandoned: bool = True
    preserve_on_timeout: bool = True
    sequential_lock: bool = True
    # Python interpreter resolution cascade (level 2):
    # workspace-scoped Python interpreter. When ``python_executable``
    # is set, it overrides the ``agent.python_executable`` default.
    # When empty, the resolver will try ``python_auto_detect`` to
    # locate the interpreter from project-level signals
    # (``.python-version``, ``pyvenv.cfg``, ``environment.yml``,
    # ``.venv/pyvenv.cfg``). When detection is disabled or fails,
    # the resolver falls back to ``agent.python_executable`` and
    # finally to "no constraint" (the agent uses PATH ``python3``).
    python_executable: str = ""
    python_auto_detect: bool = True
    # Ordered list of relative paths to probe for python interpreter
    # hints. The first match wins. Default probes cover pyenv, venv,
    # uv/poetry virtualenvs, pipenv, and conda env files.
    python_detect_files: list[str] = field(
        default_factory=lambda: [
            ".python-version",
            "pyvenv.cfg",
            ".venv/pyvenv.cfg",
            "Pipfile",
            "environment.yml",
        ]
    )


@dataclass
class WorkerConfig:
    ssh_hosts: list[str] = field(default_factory=list)
    max_concurrent_agents_per_host: int | None = None


@dataclass
class VerificationConfig:
    timeout_ms: int = 600_000
    # Regression guard (defect R1): when ``agent.test_command`` is empty,
    # verification used to pass vacuously — an agent could break hundreds
    # of existing tests and still ship a "completed" MR. With the guard
    # enabled, git-sync falls back to an auto-detected test run (pytest
    # today) and compares failures against the pre-change baseline; only
    # net-new failures block the push. Repos with no detectable test
    # suite record ``verification_status=skipped_no_tests`` instead of
    # pretending to have passed.
    regression_guard: bool = True
    # Explicit fallback command (overrides auto-detection). Runs from the
    # workspace root; non-zero exit = failing tests.
    fallback_test_command: str = ""


@dataclass
class ReproFirstConfig:
    """Repro-first gate: reproduce the bug before the fix stage may run.

    When enabled, each new issue first gets a reproduction-only agent
    pass that must produce an executable check (non-zero exit while the
    bug exists). Issues whose described behavior cannot be demonstrated
    are failed with a "cannot reproduce" report back on the tracker
    instead of an unverifiable fix MR.
    """

    enabled: bool = False
    # Wall-clock budget for the reproduction agent pass.
    timeout_ms: int = 900_000
    # Budget for executing the reproduction command itself.
    command_timeout_ms: int = 300_000
    # When non-empty, only issues carrying at least one of these labels
    # go through the gate (e.g. ["bug"]); empty means every issue.
    labels: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    max_concurrent_agents: int = 10
    max_turns: int = 600
    max_retry_backoff_ms: int = 300_000
    max_retry_attempts: int = 5
    # Base delay (ms) for retries triggered by max_turns being exhausted.
    # Shared retry budget; capped at max_retry_backoff_ms via exponential backoff.
    max_turns_retry_delay_ms: int = 30_000
    max_concurrent_agents_by_state: dict[str, int] = field(default_factory=dict)
    # NEW: ClawCodex-specific fields
    provider: str = "anthropic"
    permission_mode: str = "dontAsk"
    # Per-tool decision audit log level. "none" disables the NDJSON
    # audit trail; "minimal" records only denied decisions; "full" records
    # every tool call. Defaults to "minimal" to save disk.
    audit_log: str = "minimal"
    test_command: str = ""
    build_command: str = ""
    lint_command: str = ""
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    repro_first: ReproFirstConfig = field(default_factory=ReproFirstConfig)
    # Rate limit on operator-driven retries. When an
    # issue's `IssueRecord.retry_count` reaches this value, the
    # orchestrator refuses to honor further `agent:retry` labels /
    # `/agent retry` comment commands, even with a force flag from
    # the CLI (which is logged as a high-priority audit entry).
    max_retries_per_issue: int = 3
    # Allow `agent:retry` / `agent:follow-up` /
    # `/agent retry` to be triggered by any GitHub-style user, not
    # just the issue author. By default we enforce the strict
    # "author or maintainer only" rule. Setting this to True
    # disables the role check (e.g. for trusted-team scenarios).
    allow_anyone_to_retry: bool = False
    # 429-aware in-turn backoff. When the upstream provider returns
    # HTTP 429 (rate limit) inside a single QueryRunner turn, the
    # AgentRunner sleeps for an exponentially-growing delay and
    # re-issues the same prompt instead of failing immediately. After
    # ``rate_limit_max_retries`` consecutive 429s the circuit breaker
    # opens (``status="rate_limit_circuit_open"``) and the run is
    # handed back to the orchestrator's inter-run retry queue.
    #
    # Model name override. When set, overrides the provider's default
    # model (e.g. ``gpt-4o`` for OpenAI, ``claude-sonnet-4-20250514``
    # for Anthropic).  Leave ``None`` to use the provider's built-in
    # default (which may be a placeholder like ``gpt-5.4`` that does
    # not exist on the real API — see stagnation root-cause analysis).
    model: str | None = None
    # Multi-model stage overrides: keyed by run_kind (e.g. "review_followup",
    # "agent_followup"), each value is a dict with optional "provider" and/or
    # "model" keys. The orchestrator builds per-stage AgentRunners on top of
    # the main agent config; missing keys inherit from the parent.
    stage_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # the inter-run retry queue between separate AgentRunner.run()
    # invocations; these fields govern backoff WITHIN a single run.
    rate_limit_base_delay_ms: int = 30_000
    rate_limit_max_backoff_ms: int = 600_000
    rate_limit_exponential_factor: float = 2.0
    rate_limit_max_retries: int = 40
    # Minimum interval (ms) between successive provider API requests within
    # a single agent run. When non-zero, the agent sleeps for the remaining
    # time before issuing each new request. Default 1000ms (1s delay) to avoid
    # rate limits on providers with tight per-minute quotas (e.g. MiniMax
    # personal plan). Set to 0 for unlimited request rate.
    delay_between_requests_ms: int = 2000
    run_timeout_ms: int = 1_800_000
    # Stream-stall watchdog: abort a run once the headless
    # session shows no activity (no tool events, no stdout growth) for
    # this long, instead of waiting out the whole run_timeout_ms budget.
    # Default 300s: measured healthy runs pause up to 240s (long LLM
    # turns not streamed to stdout); genuine hangs sat 949s/1140s.
    # 0 disables. See QueryConfig.stall_timeout_s.
    stall_timeout_ms: int = 300_000
    # Early-diagnosis tier: emit a stall_suspected diagnostic (debug
    # event + WARNING log) after this much silence — guarantees a clear
    # diagnosis within ~30s of a hang without false-kill risk. 0 disables.
    stall_warn_ms: int = 30_000
    # File-path whitelist gate (glob patterns). When non-empty, only files
    # matching at least one pattern may enter the commit.  The gate runs
    # AFTER ``git add -A`` and unstages any file that doesn't match.
    # An empty list disables the gate (default).
    allowed_changed_files: list[str] = field(default_factory=list)
    # Human review gating. When True, the orchestrator marks each
    # completed issue as PENDING_REVIEW instead of COMPLETED after sync,
    # requiring a human to run `orchestrator issue review --id <id> --approve`
    # before the issue transitions to COMPLETED.
    # Works with all tracker kinds (local, GitHub, Gitee, GitCode, Linear).
    review_required: bool = False
    auto_approve: bool = False
    # Multi-agent collaboration mode (MVP). When True, the agent
    # launched for each issue runs in "coordinator mode" — it gets the
    # restricted coordinator tool set (Agent / SendMessage / TaskStop +
    # lightweight Read / WebSearch / WebFetch) and is expected to
    # spawn worker sub-agents via the Agent tool, coordinating their
    # work via SendMessage (mailbox JSONL). All multi-agent
    # infrastructure already exists in clawcodex_ext/coordinator/ and
    # clawcodex_ext/tool_system/tools/{agent,send_message}.py; this
    # flag merely flips the env var (CLAUDE_CODE_COORDINATOR_MODE)
    # that activates them before AgentRunner.run().
    coordinator_mode: bool = False
    # Root-cause fix: stagnation / loop guards. After
    # ``max_no_op_turns`` consecutive turns where the LLM made zero
    # tool calls and produced empty output, the runner emits
    # session_end_reason="stagnation" and breaks the outer while
    # loop. Loop detection: if the same tool-call signature appears
    # ``loop_detection_threshold`` times within the last
    # ``loop_detection_window`` turns, emit
    # session_end_reason="loop_detected".
    max_no_op_turns: int = 3
    loop_detection_window: int = 5
    loop_detection_threshold: int = 3
    # Skip the tracker poll in ``_should_continue`` when the
    # issue state has been identical across ``N`` consecutive polls.
    # Set to 0 to disable the cache and always poll (identical to
    # the pre-cache behaviour). The cache lives on the ``AgentSession``
    # instance — concurrent sessions never share state.
    perf_should_continue_skip_turns: int = 3
    # ProgressSink protocol refactor. ``phases`` is the ordered
    # list of named workflow phases the orchestrator drives a session
    # through. When the LLM completes a phase, :class:`ToolContextProgressSink`
    # uses ``(n / total) * 100`` to compute an honest progress
    # percentage; when ``phases`` is empty, the sink reports
    # ``progress=None`` (the dashboard shows "Phase N (progress unknown)")
    # instead of the misleading 25/50/75/100 sequence.
    # ``fallback_to_phase_step`` keeps the old ``phase_count * 25``
    # behavior for soft migration periods; new workflows should leave
    # it False and rely on ``phases`` (or explicit LLM ``ProgressReport``
    # calls) for percentage.
    phases: list[str] = field(default_factory=list)
    fallback_to_phase_step: bool = False
    # Root-cause fix: per-turn tool call cap. When the LLM
    # produces more than this many tool calls in a single turn,
    # the agent runner stops processing tool events and waits for
    # SessionComplete to force a turn boundary. This prevents
    # infinite tool-call loops (no SessionComplete emitted) while
    # still allowing complex multi-step operations.
    max_tools_per_turn: int = 50
    # Path of the Python interpreter the agent should use when
    # running shell commands inside the workspace. Empty string
    # (the default) means "do not inject a path instruction; let
    # the LLM rely on PATH." When set, an absolute path here is
    # injected into both the turn-0 issue prompt and the
    # continuation guidance so the agent does not waste turns
    # hunting for the right interpreter. Replace the
    # previously-hardcoded `/root/Conda/bin/python3` in
    # ``PromptBuilder.build_continuation_prompt``.
    python_executable: str = ""
    # Environment variables injected into every Bash subprocess
    # spawned by the agent and every verification/hook subprocess
    # spawned by the orchestrator. Values override inherited daemon
    # env, so ``PATH`` can be extended without breaking the host.
    env: dict[str, str] = field(default_factory=dict)
    # Three-channel clarification flow tuning. These map 1:1 onto
    # ``ClarificationConfig`` fields consumed in orchestrator.py
    # (``getattr(workflow.agent, ...)``). Defaults mirror the module-level
    # ``_DEFAULT_*`` constants in extensions/orchestrator/clarification.py
    # — keep them in sync when retuning.
    clarification_enabled: bool = True
    clarification_timeout_local: float = 30 * 60  # 30 minutes for local channels
    clarification_timeout_author: float = 72 * 3600  # 72 hours for author channel
    max_questions_per_issue: int = 3
    clarification_operator_priority: bool = True  # operator answers beat author
    clarification_simultaneous_grace_ms: float = 5000  # 5 seconds for "tied" answers
    # What to do when all three channels time out: "skip" | "mark_failed" | "notify"
    clarification_escalation: str = "skip"


@dataclass
class SandboxConfig:
    command: str = ""
    approval_policy: str | dict[str, Any] = field(
        default_factory=lambda: {
            "reject": {
                "sandbox_approval": True,
                "rules": True,
                "mcp_elicitations": True,
            }
        }
    )
    thread_sandbox: str = "workspace-write"
    turn_sandbox_policy: dict[str, Any] | None = None
    turn_timeout_ms: int = 3_600_000
    read_timeout_ms: int = 5_000
    stall_timeout_ms: int = 300_000


@dataclass
class HooksConfig:
    after_create: str | None = None
    before_run: str | None = None
    after_run: str | None = None
    before_remove: str | None = None
    pre_commit: str | None = None
    pre_push: str | None = None
    post_sync: str | None = None
    timeout_ms: int = 60_000


@dataclass
class ReviewFeedbackConfig:
    enabled: bool = False
    mode: str = "manual"
    poll_interval_ms: int = 60_000
    max_feedback_items_per_run: int = 20
    include_ci_failures: bool = True
    reply_to_comments: bool = True
    ignore_authors: list[str] = field(default_factory=list)
    ignored_comment_commands: list[str] = field(default_factory=list)
    ignored_feedback_sources: list[str] = field(default_factory=list)
    ignored_body_patterns: list[str] = field(default_factory=list)
    bot_login: str | None = None
    max_log_chars_per_check: int = 12_000
    max_followup_attempts_per_pr: int = 5
    pending_feedback_timeout_seconds: int = 600


@dataclass
class ObservabilityConfig:
    dashboard_enabled: bool = True
    refresh_ms: int = 1_000
    render_interval_ms: int = 16


@dataclass
class ServerConfig:
    port: int | None = None
    host: str = "127.0.0.1"


@dataclass
class ModesConfig:
    """Multi-agent collaboration-mode configuration.

    Wired by ``orchestrator.Orchestrator`` to instantiate ``ModeSelector``
    plus a ``Router`` backend and register the requested ``ModeRunner``
    implementations. Reading this section in workflow.md is opt-in:
    omitting the section yields ``ModesConfig()`` defaults, which mean
    "Phase-1 behavior — only ``single`` mode is registered and routing
    is disabled".

    YAML shape::

        modes:
          enabled: [single, pipeline]       # which modes to register
          default: single                   # fallback when router fails
          router:
            kind: heuristic                 # heuristic | llm | none
            model: <router-model>           # only used when kind=llm
            min_confidence: 0.5             # router picks below this fall back
          pipeline:
            stages: [analyzer, implementer, tester]

    Unknown keys are ignored — the loader tolerates new keys added in
    later phases so an older daemon can still read a forward-versioned
    workflow.md without crashing.
    """

    enabled: list[str] = field(default_factory=lambda: ["single"])
    default: str = "single"
    router_kind: str = "none"  # "none" | "heuristic" | "llm"
    router_model: str = "deepseek-v4-flash"  # only consulted when router_kind=="llm"
    router_endpoint: str = "https://api.deepseek.com/chat/completions"
    router_api_key_env: str = "DEEPSEEK_API_KEY"
    router_timeout_seconds: float = 15.0
    router_min_confidence: float = 0.5
    pipeline_stages: list[str] = field(default_factory=lambda: ["analyzer", "implementer", "tester"])
    # Stage retry: how many times PipelineModeRunner will re-attempt a
    # stage that exited with a terminal-failure status before aborting
    # the whole pipeline. 0 = no retries (legacy behavior).
    pipeline_max_retries_per_stage: int = 1
    # Per-stage model overrides — heterogeneous LLM agents within one
    # pipeline. Each stage name maps to a model id; absent = workflow
    # default. Sequential execution → no concurrent env-var races, so
    # we just try/finally swap workflow.agent.model per stage.
    # Makes Pipeline a *real* multi-agent system (different "agents"
    # via different LLM brains, not just role labels).
    pipeline_stage_models: dict[str, str] = field(default_factory=dict)
    # Per-stage max_turns override — workflow.agent.max_turns is a
    # single value applied everywhere; realistic Pipelines need
    # different budgets per stage (analyzer reads a lot, implementer
    # edits fast, tester runs commands). Absent stage = workflow default.
    pipeline_stage_max_turns: dict[str, int] = field(default_factory=dict)
    # Nested mode dispatch — a Pipeline stage can itself run under a
    # different ModeRunner instead of a plain AgentRunner. Absent /
    # empty = agent (legacy). Only "agent", "debate", "coordinator"
    # are allowed; nested pipeline is rejected to avoid the infinite
    # recursion trap.
    #
    # YAML shape:
    #   modes:
    #     pipeline:
    #       stages: [analyzer, implementer, tester]
    #       stage_specs:
    #         implementer:
    #           kind: debate
    #           config:
    #             proposers: [conservative, bold]
    #             judge_mode: synthesize
    #             isolation: worktree
    pipeline_stage_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Handoff strategy between Pipeline stages:
    #   "prompt"  — inject prior output as text in next stage's prompt (legacy)
    #   "mailbox" — each stage SendMessage(to=<next stage>); next stage Reads
    #               its mailbox first. Uses the existing team.json /
    #               SendMessage infra from the Coordinator mode work.
    pipeline_handoff: str = "prompt"
    debate_proposers: list[str] = field(default_factory=lambda: ["proposer_a", "proposer_b"])
    # Optional stronger model for the judge stage. None = use the
    # workflow's default agent.model (same as proposers). Set to e.g.
    # "deepseek-v4" to upgrade just the judging step.
    debate_judge_model: str | None = None
    # Judge behavior:
    #   "pick"       — pick 1 winning proposer verbatim (default; legacy)
    #   "synthesize" — combine best ideas from ALL proposers into a
    #                  hybrid solution, citing which proposer contributed
    #                  each piece. Better fit when both proposals have
    #                  genuine merits and you don't have to pick one.
    debate_judge_mode: str = "pick"
    # Per-proposer model overrides — only honored in sequential mode.
    # In parallel mode (see debate_parallel) all proposers share the
    # workflow default model to avoid concurrent env mutations.
    debate_proposer_models: dict[str, str] = field(default_factory=dict)
    # Workspace isolation strategy between proposers (and before judge):
    #   "reset"    — git reset --hard + git clean (default; cheap, single dir)
    #   "worktree" — git worktree add per proposer (real physical isolation)
    #   "none"     — no isolation (proposer A's edits leak to proposer B)
    debate_isolation: str = "reset"
    # Parallel proposers (asyncio.gather). Requires isolation=worktree
    # so each parallel branch has its own physical workspace. When False
    # (default), proposers run sequentially.
    debate_parallel: bool = False
    # Dynamic task decomposition. The seed task graph is persisted in
    # the issue workspace and executed through the existing coordinator mode.
    swarm_max_subtasks: int = 8
    swarm_max_parallel: int = 3
    swarm_max_waves: int = 6


# ---------------------------------------------------------------------------
# Top-level WorkflowConfig
# ---------------------------------------------------------------------------


@dataclass
class RulesConfig:
    """Configuration for learned rule extraction from PR review feedback."""

    enabled: bool = False
    path: str = ""
    max_rules: int = 20
    min_confidence: str = "low"


@dataclass
class PrConflictScanConfig:
    """Configuration for the optional PR conflict scan daemon job.

    When ``enabled=False`` (the default) the daemon does not poll the
    remote PR mergeable state at all — operators must trigger rebase
    via CLI / label / comment. Setting ``enabled=True`` turns on a
    background scan that, for each open PR with a workspace + branch,
    asks the tracker for the mergeable state and invokes
    ``rebase_for_pr`` when ``has_conflicts`` is True.

    Why this is opt-in: GitCode does not reliably expose ``mergeable``
    (JS-rendered page), so the scan is a no-op there. Operators on
    GitHub / Gitee can opt-in for proactive conflict detection; on
    GitCode the other three triggers remain the canonical path.
    """

    enabled: bool = False
    poll_interval_ms: int = 300_000  # 5 minutes
    max_rebase_attempts_per_issue: int = 3
    max_prs_per_scan: int = 25
    use_force_push: bool = False  # corresponds to CLI --force
    bot_login: str | None = None
    scan_states: tuple[str, ...] = ("open",)


@dataclass
class ClarifierConfig:
    """Pre-dispatch issue clarity analysis.

    This config is deliberately separate from ``ClarificationConfig`` in
    ``clarification.py``. The clarifier decides *whether* a question is
    needed; the existing resolver owns delivery, replies, and escalation.
    """

    enabled: bool = False
    block_on_unclear: bool = True
    author_first: bool = True
    max_questions: int = 3
    max_rounds: int = 2
    min_confidence: float = 0.7
    max_input_tokens: int = 6000
    max_output_tokens: int = 800
    fail_open: bool = True
    cache_enabled: bool = True
    max_analyses_per_poll: int = 4
    # Follow-up workspace focus enrichment
    workspace_focus_enabled: bool = False
    # Ops enhancement 2: optional dedicated remote-wait label; empty string = no push
    remote_label: str = ""


@dataclass
class PrTemplateConfig:
    """Optional, workflow-defined pull request title and body templates.

    Templates are rendered by :class:`GitSyncService` with a deliberately
    small, data-only set of ``{{ variable }}`` placeholders.  An empty body
    preserves the built-in PR body for backwards compatibility.
    """

    title: str = ""
    body: str = ""
