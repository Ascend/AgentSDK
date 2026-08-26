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

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..tracker import (
    default_active_states_for_kind,
    default_terminal_states_for_kind,
    normalize_tracker_kind,
    tracker_kind_info,
)

from .schema_configs import (
    AgentConfig,
    _default_tmp_workspace,
    ClarifierConfig,
    HooksConfig,
    ModesConfig,
    ObservabilityConfig,
    PollingConfig,
    PrConflictScanConfig,
    PrTemplateConfig,
    ReproFirstConfig,
    ReviewFeedbackConfig,
    RulesConfig,
    SandboxConfig,
    ServerConfig,
    TrackerConfig,
    VerificationConfig,
    WorkerConfig,
    WorkspaceConfig,
)

logger = logging.getLogger(__name__)


def _resolve_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("$"):
        env_name = value[1:]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", env_name):
            env_value = os.environ.get(env_name)
            if env_value is None or env_value == "":
                return None
            return env_value
    return value


def _normalize_secret_value(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _expand_path(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    resolved = _resolve_env_value(value)
    if resolved is None or resolved == "":
        return fallback
    return os.path.expanduser(resolved)


def _normalize_keys(value: Any, *, _inside_env: bool = False) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k) if _inside_env else str(k).lower()
            # Env var names are case-sensitive; preserve them under any
            # ``env`` key while continuing to normalize all other keys.
            next_inside_env = _inside_env or (not _inside_env and key == "env")
            result[key] = _normalize_keys(v, _inside_env=next_inside_env)
        return result
    if isinstance(value, list):
        return [_normalize_keys(v, _inside_env=_inside_env) for v in value]
    return value


def _drop_nil_values(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            cleaned = _drop_nil_values(v)
            if cleaned is not None:
                result[k] = cleaned
        return result
    if isinstance(value, list):
        return [_drop_nil_values(v) for v in value]
    return value


def _normalize_state_limits(limits: dict[str, Any] | None) -> dict[str, int]:
    if not limits:
        return {}
    result: dict[str, int] = {}
    for state_name, limit in limits.items():
        key = str(state_name).strip().lower()
        if key and isinstance(limit, int) and limit > 0:
            result[key] = limit
    return result


def _normalize_string_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default)


def _normalize_workspace_strategy(value: Any) -> str:
    strategy = str(value or "isolated").strip().lower()
    if strategy not in {"isolated", "shared", "sequential"}:
        raise ValueError("workspace.strategy must be one of: isolated, shared, sequential")
    return strategy


def _parse_repro_first_config(raw: Any) -> "ReproFirstConfig":
    """Build a ``ReproFirstConfig`` from the ``agent.repro_first`` YAML
    section. Tolerant of a missing/malformed section (all defaults,
    gate disabled).
    """
    if not isinstance(raw, dict):
        return ReproFirstConfig()

    def _int(key: str, default: int) -> int:
        try:
            value = int(raw.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return ReproFirstConfig(
        enabled=bool(raw.get("enabled", False)),
        timeout_ms=_int("timeout_ms", 900_000),
        command_timeout_ms=_int("command_timeout_ms", 300_000),
        labels=_normalize_string_list(raw.get("labels"), default=[]),
    )


def _parse_modes_config(raw: dict[str, Any]) -> "ModesConfig":
    """Build a ``ModesConfig`` from the parsed ``modes`` YAML section.

    Tolerant of:
    * missing section (``raw == {}``) → all defaults
    * unknown router kinds → coerced to ``"none"``
    * malformed ``min_confidence`` → coerced to default ``0.5``
    """
    router_raw = raw.get("router") or {}
    pipeline_raw = raw.get("pipeline") or {}
    debate_raw = raw.get("debate") or {}
    swarm_raw = raw.get("swarm") or {}

    router_kind = str(router_raw.get("kind", "none")).strip().lower()
    if router_kind not in {"none", "heuristic", "llm"}:
        logger.warning(
            "modes.router.kind=%r is unknown — falling back to 'none'",
            router_kind,
        )
        router_kind = "none"

    try:
        min_conf = float(router_raw.get("min_confidence", 0.5))
    except (TypeError, ValueError):
        min_conf = 0.5
    min_conf = max(0.0, min(1.0, min_conf))

    try:
        router_timeout = float(router_raw.get("timeout_seconds", 15.0))
    except (TypeError, ValueError):
        router_timeout = 15.0
    router_timeout = max(1.0, router_timeout)

    pipeline_handoff = str(pipeline_raw.get("handoff", "prompt")).strip().lower()
    if pipeline_handoff not in {"prompt", "mailbox"}:
        logger.warning(
            "modes.pipeline.handoff=%r is unknown — falling back to 'prompt'",
            pipeline_handoff,
        )
        pipeline_handoff = "prompt"

    return ModesConfig(
        enabled=_normalize_string_list(raw.get("enabled"), default=["single"]),
        default=str(raw.get("default", "single")).strip().lower() or "single",
        router_kind=router_kind,
        router_model=(str(router_raw.get("model", "deepseek-v4-flash")).strip() or "deepseek-v4-flash"),
        router_endpoint=(
            str(router_raw.get("endpoint", "https://api.deepseek.com/chat/completions")).strip()
            or "https://api.deepseek.com/chat/completions"
        ),
        router_api_key_env=(str(router_raw.get("api_key_env", "DEEPSEEK_API_KEY")).strip() or "DEEPSEEK_API_KEY"),
        router_timeout_seconds=router_timeout,
        router_min_confidence=min_conf,
        pipeline_stages=_normalize_string_list(
            pipeline_raw.get("stages"),
            default=["analyzer", "implementer", "tester"],
        ),
        pipeline_max_retries_per_stage=max(0, int(pipeline_raw.get("max_retries_per_stage", 1) or 0)),
        pipeline_stage_models=_normalize_model_map(pipeline_raw.get("stage_models")),
        pipeline_stage_max_turns=_normalize_int_map(pipeline_raw.get("stage_max_turns"), min_value=1),
        pipeline_stage_specs=_normalize_stage_specs(pipeline_raw.get("stage_specs")),
        pipeline_handoff=pipeline_handoff,
        debate_proposers=_normalize_string_list(
            debate_raw.get("proposers"),
            default=["proposer_a", "proposer_b"],
        ),
        debate_judge_model=(str(debate_raw["judge_model"]).strip() if debate_raw.get("judge_model") else None),
        debate_proposer_models=_normalize_model_map(debate_raw.get("proposer_models")),
        debate_isolation=_normalize_debate_isolation(debate_raw.get("isolation", "reset")),
        debate_parallel=bool(debate_raw.get("parallel", False)),
        debate_judge_mode=_normalize_debate_judge_mode(debate_raw.get("judge_mode", "pick")),
        swarm_max_subtasks=max(1, int(swarm_raw.get("max_subtasks", 8))),
        swarm_max_parallel=max(1, int(swarm_raw.get("max_parallel", 3))),
        swarm_max_waves=max(1, int(swarm_raw.get("max_waves", 6))),
    )


def _normalize_int_map(value: Any, *, min_value: int = 0) -> dict[str, int]:
    """Same shape as ``_normalize_model_map`` but for int values.

    Silently drops entries whose value can't be coerced to int or is
    below ``min_value``. Useful for per-stage numeric overrides like
    ``max_turns`` where a 0 or negative value is nonsense.
    """
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in value.items():
        key = str(k).strip()
        if not key:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv < min_value:
            continue
        out[key] = iv
    return out


def _normalize_stage_specs(value: Any) -> dict[str, dict[str, Any]]:
    """Normalize Pipeline stage_specs YAML into a clean dict.

    Silently drops:
    * non-dict entries
    * entries without a ``kind`` key
    * entries whose kind isn't in the allowed set

    (Silent drop rather than raise because config-loader shouldn't
    crash the daemon on operator typos; PipelineModeRunner will still
    log an unknown-key warning if the referenced stage doesn't exist.)
    """
    if not isinstance(value, dict):
        return {}
    allowed_kinds = {"agent", "debate", "coordinator"}
    out: dict[str, dict[str, Any]] = {}
    for stage_name, spec in value.items():
        if not isinstance(spec, dict):
            logger.warning(
                "modes.pipeline.stage_specs[%r] is not a dict — ignored",
                stage_name,
            )
            continue
        kind = str(spec.get("kind", "agent")).strip().lower()
        if kind not in allowed_kinds:
            logger.warning(
                "modes.pipeline.stage_specs[%r].kind=%r not in %s — ignored",
                stage_name,
                kind,
                sorted(allowed_kinds),
            )
            continue
        config = spec.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        out[str(stage_name).strip()] = {"kind": kind, "config": dict(config)}
    return out


def _normalize_debate_judge_mode(value: Any) -> str:
    candidate = str(value or "pick").strip().lower()
    if candidate not in {"pick", "synthesize"}:
        logger.warning(
            "modes.debate.judge_mode=%r is unknown — falling back to 'pick'",
            candidate,
        )
        return "pick"
    return candidate


def _normalize_model_map(value: Any) -> dict[str, str]:
    """Normalize a YAML map of role-name → model-id into a clean dict.

    Tolerant of:
    * None / missing key → empty dict
    * non-string keys/values → coerced via str() + stripped
    * empty-string values → dropped (signals "use default")
    """
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        key = str(k).strip()
        val = str(v).strip() if v is not None else ""
        if key and val:
            out[key] = val
    return out


def _normalize_debate_isolation(value: Any) -> str:
    candidate = str(value or "reset").strip().lower()
    if candidate not in {"reset", "worktree", "none"}:
        logger.warning(
            "modes.debate.isolation=%r is unknown — falling back to 'reset'",
            candidate,
        )
        return "reset"
    return candidate


def _resolve_orchestrator_permission_mode(
    raw_value: Any,
    *,
    is_orchestrator: bool,
) -> str:
    """Resolve permission_mode with headless auto-override.

    When a workflow.md is being loaded for the orchestrator (detected by the
    presence of a ``tracker`` section), a ``dontAsk`` value — whether explicit
    or default — is auto-promoted to ``bypassPermissions``. This ensures
    fully unattended execution, since ``dontAsk`` may still trigger
    ``ApprovalPolicy`` checks that can block tool calls in headless mode.

    Explicit non-default values are preserved so users can opt back into a
    more restrictive mode if needed.
    """
    raw = str(raw_value).strip() if raw_value else "dontAsk"
    canonical_modes = {
        "acceptedits": "acceptEdits",
        "bypasspermissions": "bypassPermissions",
        "default": "default",
        "dontask": "dontAsk",
        "plan": "plan",
    }
    normalized = canonical_modes.get(raw.lower(), raw)
    if is_orchestrator and normalized == "dontAsk":
        return "bypassPermissions"
    return normalized


_VALID_AUDIT_LOG_LEVELS = {"none", "minimal", "full"}


def _resolve_audit_log(raw_value: Any) -> str:
    """Canonicalize audit_log level."""
    raw = str(raw_value).strip().lower() if raw_value else "minimal"
    if raw in _VALID_AUDIT_LOG_LEVELS:
        return raw
    logger.warning(
        "audit_log=%r is not one of %s; falling back to 'minimal'",
        raw_value,
        sorted(_VALID_AUDIT_LOG_LEVELS),
    )
    return "minimal"


def permission_mode_to_triple(
    permission_mode: str,
    *,
    interactive: bool | None = None,
    default_decision: str | None = None,
    audit_log: str | None = None,
) -> dict[str, Any]:
    """Translate legacy permission_mode enum into three orthogonal fields.

    Explicit overrides take precedence; missing values are inferred from the
    legacy mode. The current wiring only sets ``audit_log``; ``interactive`` and
    ``default_decision`` are reserved for future extensions.
    """
    mode = str(permission_mode).strip() if permission_mode else "default"
    mapping: dict[str, dict[str, Any]] = {
        "default": {"interactive": True, "default_decision": "ask", "audit_log": "minimal"},
        "plan": {"interactive": True, "default_decision": "ask", "audit_log": "minimal"},
        "acceptEdits": {"interactive": True, "default_decision": "allow", "audit_log": "minimal"},
        "bypassPermissions": {
            "interactive": False,
            "default_decision": "allow",
            "audit_log": "minimal",
        },
        "dontAsk": {"interactive": False, "default_decision": "deny", "audit_log": "minimal"},
        "auto": {"interactive": False, "default_decision": "allow", "audit_log": "minimal"},
        "bubble": {"interactive": True, "default_decision": "ask", "audit_log": "minimal"},
    }
    defaults = mapping.get(mode, mapping["default"])
    result = {
        "interactive": interactive if interactive is not None else defaults["interactive"],
        "default_decision": default_decision if default_decision is not None else defaults["default_decision"],
        "audit_log": audit_log if audit_log is not None else defaults["audit_log"],
    }
    if result["default_decision"] not in {"allow", "deny", "ask"}:
        result["default_decision"] = defaults["default_decision"]
    return result


def _normalize_title_prefix_match(value: Any) -> str:
    mode = str(value or "any").strip().lower()
    if mode not in {"any", "all"}:
        logger.warning("tracker.title_prefix_match=%r is invalid; using 'any'", value)
        return "any"
    return mode


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class WorkflowConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    polling: PollingConfig = field(default_factory=PollingConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    review_feedback: ReviewFeedbackConfig = field(default_factory=ReviewFeedbackConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)
    pr_template: PrTemplateConfig = field(default_factory=PrTemplateConfig)
    pr_conflict_scan: "PrConflictScanConfig" = field(default_factory=PrConflictScanConfig)
    clarifier: "ClarifierConfig" = field(default_factory=ClarifierConfig)
    source_path: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WorkflowConfig":
        """Build from a raw dict (already parsed YAML front matter)."""
        raw = _normalize_keys(_drop_nil_values(raw))

        tracker_raw = raw.get("tracker", {})
        polling_raw = raw.get("polling", {})
        workspace_raw = raw.get("workspace", {})
        worker_raw = raw.get("worker", {})
        agent_raw = raw.get("agent", {})
        codex_raw = raw.get("sandbox") or raw.get("codex") or {}
        hooks_raw = raw.get("hooks", {})
        review_feedback_raw = raw.get("review_feedback", {})
        rules_raw = raw.get("rules", {})
        modes_raw = raw.get("modes", {}) or {}
        observability_raw = raw.get("observability", {})
        server_raw = raw.get("server", {})
        pr_conflict_scan_raw = raw.get("pr_conflict_scan", {})
        clarifier_raw = raw.get("clarifier", {}) or {}
        pr_template_raw = raw.get("pr_template", {}) or {}
        if not isinstance(pr_template_raw, dict):
            logger.warning("pr_template must be a mapping; ignoring it")
            pr_template_raw = {}

        tracker_kind = normalize_tracker_kind(tracker_raw.get("kind", "linear"))
        tracker_info = tracker_kind_info(tracker_kind)
        tracker_active_states = _normalize_string_list(
            tracker_raw.get("active_states"),
            default_active_states_for_kind(tracker_kind),
        )
        tracker_terminal_states = _normalize_string_list(
            tracker_raw.get("terminal_states"),
            default_terminal_states_for_kind(tracker_kind),
        )
        tracker_skip_labels = _normalize_string_list(
            tracker_raw.get("skip_labels"),
            [],
        )
        tracker_require_any_labels = _normalize_string_list(
            tracker_raw.get("require_any_labels"),
            [],
        )
        tracker_title_prefixes = _normalize_string_list(tracker_raw.get("title_prefixes"), [])
        tracker_title_prefix_match = _normalize_title_prefix_match(
            tracker_raw.get(
                "title_prefix_match",
                tracker_raw.get("title_prefix_match_mode", tracker_raw.get("title_prefix_mode")),
            )
        )

        tracker = TrackerConfig(
            kind=tracker_kind,
            endpoint=_resolve_env_value(tracker_raw.get("endpoint")) or tracker_info.default_endpoint,
            api_key=_normalize_secret_value(_resolve_env_value(tracker_raw.get("api_key")))
            or _resolve_first_env(tracker_info.api_key_env_vars),
            project_slug=tracker_raw.get("project_slug"),
            owner=_resolve_env_value(tracker_raw.get("owner")) or _resolve_first_env(tracker_info.owner_env_vars),
            repo=_resolve_env_value(tracker_raw.get("repo")) or _resolve_first_env(tracker_info.repo_env_vars),
            clone_url=_resolve_env_value(tracker_raw.get("clone_url")),
            assignee=_resolve_env_value(tracker_raw.get("assignee"))
            or _resolve_first_env(tracker_info.assignee_env_vars),
            branch_prefix=_resolve_env_value(tracker_raw.get("branch_prefix")),
            issues_path=_normalize_secret_value(_expand_path(tracker_raw.get("issues_path"), "")),
            active_states=tracker_active_states,
            terminal_states=tracker_terminal_states,
            skip_labels=tracker_skip_labels,
            require_any_labels=tracker_require_any_labels,
            title_prefixes=tracker_title_prefixes,
            title_prefix_match=tracker_title_prefix_match,
        )

        workspace_root = _expand_path(workspace_raw.get("root"), _default_tmp_workspace())
        workspace_strategy = _normalize_workspace_strategy(workspace_raw.get("strategy"))
        workspace = WorkspaceConfig(
            root=workspace_root,
            hooks=workspace_raw.get("hooks", {}),
            repo_clone_url=_resolve_env_value(workspace_raw.get("repo_clone_url")),
            upstream_clone_url=_resolve_env_value(workspace_raw.get("upstream_clone_url")),
            clone_depth=workspace_raw.get("clone_depth", 1),
            checkout_issue_branch=workspace_raw.get("checkout_issue_branch", True),
            git_username=_resolve_env_value(workspace_raw.get("git_username")),
            git_email=_resolve_env_value(workspace_raw.get("git_email")),
            git_token=_normalize_secret_value(_resolve_env_value(workspace_raw.get("git_token"))),
            gitignore_patterns=workspace_raw.get(
                "gitignore_patterns",
                [
                    ".orchestrator_control",
                    ".operator_hints.md",
                    ".reports",
                    "*.pyc",
                    "__pycache__",
                    "*.egg-info",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".ruff_cache",
                    "*.log",
                ],
            ),
            strategy=workspace_strategy,
            base_branch=_resolve_env_value(workspace_raw.get("base_branch")),
            integration_branch=_resolve_env_value(workspace_raw.get("integration_branch")),
            require_clean_start=bool(workspace_raw.get("require_clean_start", True)),
            require_clean_between_issues=bool(workspace_raw.get("require_clean_between_issues", True)),
            preserve_on_terminal=bool(workspace_raw.get("preserve_on_terminal", True)),
            preserve_on_failure=bool(workspace_raw.get("preserve_on_failure", True)),
            preserve_on_abandoned=bool(workspace_raw.get("preserve_on_abandoned", True)),
            preserve_on_timeout=bool(workspace_raw.get("preserve_on_timeout", True)),
            sequential_lock=bool(workspace_raw.get("sequential_lock", True)),
        )

        verification_raw = agent_raw.get("verification", {})
        # Multi-model stage overrides: parse agent.stages YAML dict.
        stages_raw = agent_raw.get("stages", {}) or {}
        stage_overrides: dict[str, dict[str, Any]] = {}
        for stage_name, stage_cfg in stages_raw.items():
            if not isinstance(stage_cfg, dict):
                continue
            override: dict[str, Any] = {}
            provider = _resolve_env_value(stage_cfg.get("provider"))
            model = _resolve_env_value(stage_cfg.get("model"))
            if provider:
                override["provider"] = provider
            if model:
                override["model"] = model
            if override:
                stage_overrides[stage_name] = override
        agent = AgentConfig(
            max_concurrent_agents=agent_raw.get("max_concurrent_agents", 10),
            max_turns=agent_raw.get("max_turns", 600),
            max_retry_backoff_ms=agent_raw.get("max_retry_backoff_ms", 300_000),
            max_retry_attempts=agent_raw.get("max_retry_attempts", 5),
            max_turns_retry_delay_ms=agent_raw.get("max_turns_retry_delay_ms", 30_000),
            max_concurrent_agents_by_state=_normalize_state_limits(agent_raw.get("max_concurrent_agents_by_state")),
            provider=agent_raw.get("provider", "anthropic"),
            permission_mode=_resolve_orchestrator_permission_mode(
                agent_raw.get("permission_mode"),
                is_orchestrator=bool(tracker_raw),
            ),
            # Orthogonal audit_log level, independent of permission_mode.
            audit_log=_resolve_audit_log(agent_raw.get("audit_log")),
            test_command=_resolve_env_value(agent_raw.get("test_command")) or "",
            build_command=_resolve_env_value(agent_raw.get("build_command")) or "",
            lint_command=_resolve_env_value(agent_raw.get("lint_command")) or "",
            verification=VerificationConfig(
                timeout_ms=verification_raw.get("timeout_ms", 600_000),
                regression_guard=bool(verification_raw.get("regression_guard", True)),
                fallback_test_command=(_resolve_env_value(verification_raw.get("fallback_test_command")) or ""),
            ),
            repro_first=_parse_repro_first_config(agent_raw.get("repro_first") or {}),
            # Retry rate limit + role check settings
            max_retries_per_issue=agent_raw.get("max_retries_per_issue", 3),
            allow_anyone_to_retry=bool(agent_raw.get("allow_anyone_to_retry", False)),
            # 429-aware in-turn backoff (see AgentConfig docstring above)
            rate_limit_base_delay_ms=agent_raw.get("rate_limit_base_delay_ms", 30_000),
            rate_limit_max_backoff_ms=agent_raw.get("rate_limit_max_backoff_ms", 600_000),
            rate_limit_exponential_factor=float(agent_raw.get("rate_limit_exponential_factor", 2.0)),
            rate_limit_max_retries=agent_raw.get("rate_limit_max_retries", 40),
            delay_between_requests_ms=agent_raw.get("delay_between_requests_ms", 2000),
            run_timeout_ms=agent_raw.get("run_timeout_ms", 1_800_000),
            stall_timeout_ms=agent_raw.get("stall_timeout_ms", 300_000),
            stall_warn_ms=agent_raw.get("stall_warn_ms", 30_000),
            # File-path whitelist gate (see AgentConfig docstring).
            allowed_changed_files=_normalize_string_list(agent_raw.get("allowed_changed_files"), default=[]),
            # Review gate — when True, sync ends at PENDING_REVIEW
            # instead of COMPLETED, requiring human approve CLI command.
            review_required=bool(agent_raw.get("review_required", False)),
            auto_approve=bool(agent_raw.get("auto_approve", False)),
            # MVP multi-agent: coordinator mode toggle (from workflow.md)
            coordinator_mode=bool(agent_raw.get("coordinator_mode", False)),
            # Named workflow phases drive honest progress
            # percentages in ToolContextProgressSink. ``phases`` is
            # parsed as a list (the YAML ``- a`` / ``- b`` syntax)
            # and defaults to empty. ``fallback_to_phase_step``
            # reverts to the legacy ``phase_count * 25`` step function
            # without crashing the loader. ``fallback_to_phase_step``
            # defaults to ``False`` so new workflows see ``None``
            # instead of misleading 25/50/75/100.
            phases=_normalize_string_list(agent_raw.get("phases"), default=[]),
            fallback_to_phase_step=bool(agent_raw.get("fallback_to_phase_step", False)),
            # Root-cause fix: stagnation / loop guard knobs.
            # These were defined in AgentConfig (schema.py) and set in
            # workflow.md, but ``from_dict`` never forwarded them to the
            # dataclass constructor, so the schema defaults (3/5/3) were
            # always used regardless of the YAML config.
            max_no_op_turns=int(agent_raw.get("max_no_op_turns", 3)),
            loop_detection_window=int(agent_raw.get("loop_detection_window", 5)),
            loop_detection_threshold=int(agent_raw.get("loop_detection_threshold", 3)),
            # Per-turn tool cap: schema default was 50 but ``from_dict`` did not
            # forward the YAML value, so workflow.md edits were ignored.
            max_tools_per_turn=int(agent_raw.get("max_tools_per_turn", 50)),
            # Root-cause fix: model name override.
            model=_resolve_env_value(agent_raw.get("model")) or None,
            # Multi-model stage overrides (parsed above).
            stage_overrides=stage_overrides,
            # Per-run env vars merged into Bash/hook subprocess env.
            env={str(k): str(v) for k, v in (agent_raw.get("env") or {}).items() if v is not None},
            # Three-channel clarification flow tuning. Keys mirror the
            # ``getattr(workflow.agent, ...)`` reads in orchestrator.py;
            # defaults mirror the ``_DEFAULT_*`` constants in clarification.py.
            clarification_enabled=bool(agent_raw.get("clarification_enabled", True)),
            clarification_timeout_local=float(agent_raw.get("clarification_timeout_local", 30 * 60)),
            clarification_timeout_author=float(agent_raw.get("clarification_timeout_author", 72 * 3600)),
            max_questions_per_issue=int(agent_raw.get("max_questions_per_issue", 3)),
            clarification_operator_priority=bool(agent_raw.get("clarification_operator_priority", True)),
            clarification_simultaneous_grace_ms=float(agent_raw.get("clarification_simultaneous_grace_ms", 5000)),
            clarification_escalation=str(agent_raw.get("clarification_escalation", "skip")),
        )
        if workspace.strategy == "sequential":
            if agent.max_concurrent_agents != 1:
                raise ValueError("workspace.strategy=sequential requires agent.max_concurrent_agents=1")
            over_limit_states = [state for state, limit in agent.max_concurrent_agents_by_state.items() if limit > 1]
            if over_limit_states:
                raise ValueError(
                    "workspace.strategy=sequential requires all agent.max_concurrent_agents_by_state values to be <= 1"
                )

        sandbox = SandboxConfig(
            command=codex_raw.get("command", ""),
            approval_policy=codex_raw.get("approval_policy", SandboxConfig().approval_policy),
            thread_sandbox=codex_raw.get("thread_sandbox", "workspace-write"),
            turn_sandbox_policy=codex_raw.get("turn_sandbox_policy"),
            turn_timeout_ms=codex_raw.get("turn_timeout_ms", 3_600_000),
            read_timeout_ms=codex_raw.get("read_timeout_ms", 5_000),
            stall_timeout_ms=codex_raw.get("stall_timeout_ms", 300_000),
        )

        hooks = HooksConfig(
            after_create=_resolve_env_value(hooks_raw.get("after_create")),
            before_run=_resolve_env_value(hooks_raw.get("before_run")),
            after_run=_resolve_env_value(hooks_raw.get("after_run")),
            before_remove=_resolve_env_value(hooks_raw.get("before_remove")),
            pre_commit=_resolve_env_value(hooks_raw.get("pre_commit")),
            pre_push=_resolve_env_value(hooks_raw.get("pre_push")),
            post_sync=_resolve_env_value(hooks_raw.get("post_sync")),
            timeout_ms=hooks_raw.get("timeout_ms", 60_000),
        )

        return cls(
            tracker=tracker,
            polling=PollingConfig(interval_ms=polling_raw.get("interval_ms", 30_000)),
            workspace=workspace,
            worker=WorkerConfig(
                ssh_hosts=worker_raw.get("ssh_hosts", []),
                max_concurrent_agents_per_host=worker_raw.get("max_concurrent_agents_per_host"),
            ),
            agent=agent,
            sandbox=sandbox,
            hooks=hooks,
            rules=RulesConfig(
                enabled=bool(rules_raw.get("enabled", False)),
                path=str(rules_raw.get("path", "")).strip(),
                max_rules=int(rules_raw.get("max_rules", 20)),
                min_confidence=str(rules_raw.get("min_confidence", "low")).strip().lower(),
            ),
            review_feedback=ReviewFeedbackConfig(
                enabled=bool(review_feedback_raw.get("enabled", False)),
                mode=str(review_feedback_raw.get("mode", "manual")).strip().lower() or "manual",
                poll_interval_ms=review_feedback_raw.get("poll_interval_ms", 60_000),
                max_feedback_items_per_run=review_feedback_raw.get("max_feedback_items_per_run", 20),
                include_ci_failures=bool(review_feedback_raw.get("include_ci_failures", True)),
                reply_to_comments=bool(review_feedback_raw.get("reply_to_comments", True)),
                ignore_authors=_normalize_string_list(review_feedback_raw.get("ignore_authors"), []),
                ignored_comment_commands=_normalize_string_list(
                    review_feedback_raw.get("ignored_comment_commands"), []
                ),
                ignored_feedback_sources=_normalize_string_list(
                    review_feedback_raw.get("ignored_feedback_sources"), []
                ),
                ignored_body_patterns=_normalize_string_list(review_feedback_raw.get("ignored_body_patterns"), []),
                bot_login=_resolve_env_value(review_feedback_raw.get("bot_login")),
                max_log_chars_per_check=review_feedback_raw.get("max_log_chars_per_check", 12_000),
                max_followup_attempts_per_pr=review_feedback_raw.get("max_followup_attempts_per_pr", 5),
                pending_feedback_timeout_seconds=review_feedback_raw.get("pending_feedback_timeout_seconds", 600),
            ),
            observability=ObservabilityConfig(
                dashboard_enabled=observability_raw.get("dashboard_enabled", True),
                refresh_ms=observability_raw.get("refresh_ms", 1_000),
                render_interval_ms=observability_raw.get("render_interval_ms", 16),
            ),
            server=ServerConfig(
                port=server_raw.get("port"),
                host=server_raw.get("host", "127.0.0.1"),
            ),
            modes=_parse_modes_config(modes_raw),
            pr_template=PrTemplateConfig(
                title=str(pr_template_raw.get("title", "") or "").strip(),
                body=str(pr_template_raw.get("body", "") or ""),
            ),
            pr_conflict_scan=PrConflictScanConfig(
                enabled=bool(pr_conflict_scan_raw.get("enabled", False)),
                poll_interval_ms=pr_conflict_scan_raw.get("poll_interval_ms", 300_000),
                max_rebase_attempts_per_issue=pr_conflict_scan_raw.get("max_rebase_attempts_per_issue", 3),
                max_prs_per_scan=pr_conflict_scan_raw.get("max_prs_per_scan", 25),
                use_force_push=bool(pr_conflict_scan_raw.get("use_force_push", False)),
                bot_login=_resolve_env_value(pr_conflict_scan_raw.get("bot_login")),
                scan_states=tuple(_normalize_string_list(pr_conflict_scan_raw.get("scan_states"), ["open"])),
            ),
            clarifier=ClarifierConfig(
                enabled=bool(clarifier_raw.get("enabled", False)),
                block_on_unclear=bool(clarifier_raw.get("block_on_unclear", True)),
                author_first=bool(clarifier_raw.get("author_first", True)),
                max_questions=max(1, int(clarifier_raw.get("max_questions", 3))),
                max_rounds=max(1, int(clarifier_raw.get("max_rounds", 2))),
                min_confidence=max(
                    0.0,
                    min(1.0, float(clarifier_raw.get("min_confidence", 0.7))),
                ),
                max_input_tokens=max(256, int(clarifier_raw.get("max_input_tokens", 6000))),
                max_output_tokens=max(128, int(clarifier_raw.get("max_output_tokens", 800))),
                fail_open=bool(clarifier_raw.get("fail_open", True)),
                cache_enabled=bool(clarifier_raw.get("cache_enabled", True)),
                max_analyses_per_poll=max(
                    1,
                    int(clarifier_raw.get("max_analyses_per_poll", 4)),
                ),
            ),
        )

    def resolve_turn_sandbox_policy(self, workspace_path: str | None = None) -> dict[str, Any]:
        if self.sandbox.turn_sandbox_policy:
            return self.sandbox.turn_sandbox_policy
        root = workspace_path or self.workspace.root
        return {
            "type": "workspaceWrite",
            "writableRoots": [root],
            "readOnlyAccess": {"type": "fullAccess"},
            "networkAccess": False,
            "excludeTmpdirEnvVar": False,
            "excludeSlashTmp": False,
        }


def _resolve_first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _normalize_secret_value(os.environ.get(name))
        if value:
            return value
    return None
