#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Settings schema types matching TypeScript settings/types.ts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- PermissionsConfig replaces the legacy `list[PermissionRule]` ---


@dataclass
class PermissionRule:
    """Legacy flat permission rule retained for API and migration compatibility."""

    tool: str = ""
    allow: bool = True
    glob: str | None = None
    regex: str | None = None
    description: str = ""
    source: str = "user"


_PERMISSIONS_KNOWN_SUBKEYS: frozenset[str] = frozenset(
    {
        "allow",
        "deny",
        "ask",
        "defaultMode",
        "additionalDirectories",
        "allowBypassPermissionsMode",
        "rules",
    }
)

# P108-E — keys covered by ``FreezeSettings``. Unknown
# ``freeze.*`` sub-keys flow into ``FreezeSettings.additional`` (forward-
# compat bag) so the on-disk JSON can carry schema additions without a
# code change.
_FREEZE_KNOWN_SUBKEYS: frozenset[str] = frozenset(
    {
        "agent_loop_timeout_s",
        "turn_timeout_s",
        "tool_timeout_s",
        "permission_timeout_s",
        "threshold_s",
        "dump_dir",
    }
)


@dataclass
class PermissionsConfig:
    """Structured `permissions` block — matches on-disk + TS upstream contract.

    (2026-06-02): replaces the legacy ``list[PermissionRule]`` shape that
    was drifting from the on-disk dict format
    (``src/permissions/updates.py:persist_permission_update``) and the TS
    upstream ``permissions.*`` contract. Field names are Python snake_case;
    ``from_dict`` / ``to_dict`` translate to/from the camelCase on-disk keys
    used by the rest of the system.

    Unknown sub-keys land in :attr:`additional` (forward-compat bag) so
    newer schema additions do not require touching this dataclass.
    """

    allow_bypass_permissions_mode: bool = False
    default_mode: str | None = None
    # Always-initialized 3-behavior skeleton: downstream code can read
    # ``rules["allow"]`` / ``rules["deny"]`` / ``rules["ask"]`` without a
    # KeyError. ``to_dict()`` still drops the empty rules dict to keep
    # the on-disk shape minimal.
    rules: dict[str, list[str]] = field(default_factory=lambda: {"allow": [], "deny": [], "ask": []})
    additional_directories: list[str] = field(default_factory=list)
    additional: dict[str, Any] = field(default_factory=dict)
    legacy_rules: list[PermissionRule] = field(default_factory=list, repr=False)

    @classmethod
    def from_dict(cls, data: Any) -> "PermissionsConfig":
        if isinstance(data, list):
            return cls(
                legacy_rules=[
                    PermissionRule(**item) if isinstance(item, dict) else item
                    for item in data
                    if isinstance(item, (dict, PermissionRule))
                ]
            )
        if not isinstance(data, dict):
            return cls()

        rules: dict[str, list[str]] = {}
        rules_raw = data.get("rules")
        rules_source: dict[str, Any] = rules_raw if isinstance(rules_raw, dict) else {}
        for behavior in ("allow", "deny", "ask"):
            bucket = rules_source.get(behavior)
            if not isinstance(bucket, list):
                bucket = data.get(behavior, [])
            if isinstance(bucket, list):
                rules[behavior] = [str(x) for x in bucket]

        add_dirs = data.get("additionalDirectories")
        if not isinstance(add_dirs, list):
            add_dirs = []
        additional_directories = [str(d) for d in add_dirs]

        allow_bypass = data.get("allowBypassPermissionsMode", False)
        allow_bypass_permissions_mode = bool(allow_bypass)

        default_mode_raw = data.get("defaultMode")
        default_mode: str | None
        if isinstance(default_mode_raw, str) and default_mode_raw:
            default_mode = default_mode_raw
        else:
            default_mode = None

        additional = {k: v for k, v in data.items() if k not in _PERMISSIONS_KNOWN_SUBKEYS}

        return cls(
            allow_bypass_permissions_mode=allow_bypass_permissions_mode,
            default_mode=default_mode,
            rules=rules,
            additional_directories=additional_directories,
            additional=additional,
        )

    def to_dict(self) -> dict[str, Any] | list[dict[str, Any]]:
        if self.legacy_rules:
            from dataclasses import asdict

            return [asdict(rule) for rule in self.legacy_rules]
        d: dict[str, Any] = dict(self.additional)
        d["allowBypassPermissionsMode"] = self.allow_bypass_permissions_mode
        if self.default_mode is not None:
            d["defaultMode"] = self.default_mode
        if self.rules:
            d["rules"] = {k: list(v) for k, v in self.rules.items()}
        if self.additional_directories:
            d["additionalDirectories"] = list(self.additional_directories)
        return d

    def __len__(self) -> int:
        return len(self.legacy_rules)

    def __iter__(self):
        return iter(self.legacy_rules)

    def __getitem__(self, index: int) -> PermissionRule:
        return self.legacy_rules[index]


@dataclass
class ToolSettings:
    """Per-tool configuration."""

    enabled: bool = True
    allowed_commands: list[str] = field(default_factory=list)
    denied_commands: list[str] = field(default_factory=list)
    timeout_seconds: int = 120


@dataclass
class OutputStyleSettings:
    """Output style configuration."""

    style: str = "default"  # "default" | "concise" | "verbose" | "markdown"
    max_width: int = 120
    show_thinking: bool = False


@dataclass
class SpinnerVerbsSettings:
    """Custom spinner-verb configuration.

    Mirrors TS ``settings/types.ts:695`` (``spinnerVerbs``). ``mode``:
    ``"append"`` adds ``verbs`` to the built-in defaults; ``"replace"``
    uses only ``verbs``. See :mod:`src.constants.spinner_verbs`.
    """

    mode: str = "append"  # "append" | "replace"
    verbs: list[str] = field(default_factory=list)


@dataclass
class FreezeSettings:
    """P108-E Configure timeout and freeze-detection budgets."""

    agent_loop_timeout_s: float = 600.0
    turn_timeout_s: float = 300.0
    tool_timeout_s: float = 120.0
    permission_timeout_s: float = 30.0
    # ``threshold_s`` is the wall-clock gap before the watchdog
    # declares the agent loop frozen. It MUST be < turn_timeout_s so
    # the threshold trips first and dumps a stack before the layer-2
    # budget fires.
    threshold_s: float = 60.0
    # Diagnostic dump directory. When unset, defaults to the OS temp
    # dir + ``clawcodex-freeze`` (see ``freeze_config.dump_path``).
    dump_dir: str | None = None


@dataclass
class SandboxSettings:
    """Sandbox configuration — mirrors TS ``SandboxSettingsSchema``
    (entrypoints/sandboxTypes.ts:91).

    The port does NOT implement sandbox ENFORCEMENT (TS wraps the external
    ``@anthropic-ai/sandbox-runtime`` — a deferred sub-chapter). So the port's
    sandbox is permanently "unavailable", which maps onto TS's OWN documented
    fallback (sandboxTypes.ts:96-103): when ``enabled`` is true but the sandbox
    cannot start, ``failIfUnavailable`` decides between a hard startup error
    (managed-settings hard gate) and "a warning is shown and commands run
    unsandboxed". Parsing these fields (rather than dropping the whole
    ``sandbox`` key) is what makes that guard possible — an unparsed key would
    silently vanish, giving a false sense of security.
    """

    enabled: bool = False
    fail_if_unavailable: bool = False  # TS failIfUnavailable (default false)
    auto_allow_bash_if_sandboxed: bool = True
    allow_unsandboxed_commands: bool = True
    excluded_commands: list[str] = field(default_factory=list)
    # TS enabledPlatforms (sandboxTypes.ts:104): restrict sandboxing to
    # specific platforms; on a platform NOT listed, TS treats sandbox as
    # disabled (no gate, no warning). Empty = all platforms.
    enabled_platforms: list[str] = field(default_factory=list)


@dataclass
class CompactSettings:
    """Compaction settings."""

    auto_compact: bool = True
    threshold_tokens: int = 100_000
    max_compact_retries: int = 3


@dataclass
class ModelLimitSettings:
    """User-declared runtime limits for custom/private model IDs."""

    context_window: int | None = None
    max_output_tokens: int | None = None


@dataclass
class HookSettings:
    """Hook configuration."""

    enabled: bool = True
    timeout_ms: int = 30_000
    max_concurrent: int = 5


@dataclass
class McpServerSettings:
    """MCP server configuration."""

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class SettingsSchema:
    """Full settings schema matching TypeScript SettingsSchema."""

    # Model
    model: str = ""
    # Provider key that ``model`` was chosen under (ch03 round-3 G1).
    # Multi-provider port: a persisted model is meaningful only with the
    # provider that served it — TS utils/model/model.ts:109-135 documents
    # the cross-provider staleness failure (a stale settings.model kept
    # firing at the wrong endpoint and 400ing after a provider switch).
    # Same pairing rationale as advisor_model/advisor_provider below.
    # Read-side guard: the persisted model applies ONLY when this matches
    # the session's active provider. Distinct from ``provider`` (the
    # active-provider selection) — reusing that key would make the guard
    # vacuous and let /model mutate provider selection.
    model_provider: str = ""
    small_fast_model: str = ""
    # Per-model limits for gateways/private deployments whose metadata is not
    # in the built-in catalog. JSON accepts both modelLimits/contextWindow and
    # the Python-native snake_case spellings.
    model_limits: dict[str, ModelLimitSettings] = field(default_factory=dict)
    # /goal turn-budget backstop (src/goals). Claude Code ships the goal
    # loop unbounded; the port pauses the goal after this many evaluated
    # turns so a mis-judging evaluator can't spend unbounded tokens —
    # ``/goal resume`` continues with a fresh budget. 0/negative → default.
    goal_max_turns: int = 20
    # Advisor — reviewer tool. Empty string = unset (no /advisor).
    # Persisted analogue of TS appState.advisorModel; the /advisor slash
    # command writes here, and _call_model_sync reads from here at request
    # time. See src/utils/advisor.py.
    advisor_model: str = ""
    # Provider key (matches ~/.clawcodex/config.json's providers map) that
    # serves the advisor call. REQUIRED when advisor_model is set —
    # clawcodex is multi-provider and the same model name (e.g.
    # ``claude-opus-4-7``) can live behind anthropic, openai (litellm),
    # openrouter, bedrock, ... so name-based inference is ambiguous.
    # The /advisor command writes both fields together via the
    # ``<provider>:<model>`` syntax. Empty + advisor_model set = misconfig
    # that surfaces as a clear error at the first advisor call.
    advisor_provider: str = ""
    # Force client-side advisor mode (tool dispatch + separate API call)
    # even when the main provider is first-party Anthropic. Default False
    # lets the server-side beta path engage when applicable. Set via
    # ``/advisor <model> --client``. See decide_advisor_mode() in
    # src/utils/advisor.py for the full activation table.
    advisor_client_mode: bool = False
    # Master on/off switch for the advisor (reviewer tool). Default False —
    # the advisor is OFF unless the user explicitly opts in, even when
    # advisor_model/advisor_provider are configured. Enable via
    # ``"advisor_enabled": true`` in ~/.clawcodex/config.json's "settings"
    # block, or by running ``/advisor <provider>:<model>`` (which flips it on).
    # ``decide_advisor_mode`` returns INACTIVE whenever this is False.
    advisor_enabled: bool = False

    # Auto-mode transcript classifier (ch06 round-4 PR-B). The
    # ``feature('TRANSCRIPT_CLASSIFIER')`` analog: default OFF, so `auto`
    # mode keeps today's zero-extra-cost STATIC heuristic. When True, the
    # static heuristic stays the fast-path pre-filter (safe reads/edits/
    # bash resolve with no LLM call) and only the residual asks fire a
    # per-ask LLM security classification on the session provider. Enable
    # only where a classifier model + prompt caching are affordable.
    auto_mode_classifier_enabled: bool = False
    # Optional classifier model/provider (TS getClassifierModel default =
    # the main-loop model). Empty → the session provider + its model.
    auto_mode_classifier_model: str = ""
    auto_mode_classifier_provider: str = ""
    # Iron gate on classifier ERROR (timeout/parse/abort). False (default)
    # = fail-CLOSED (deny) — TS tengu_iron_gate_closed default true. True =
    # fail-open (return the original ask).
    auto_mode_iron_gate_open: bool = False

    # ch11 round-4 — the LLM memory-relevance recall. Default OFF, faithful
    # to TS (its recall is behind isAutoMemoryEnabled() &&
    # feature('tengu_moth_copse'), off in OSS) and because firing an LLM
    # side-query per user turn on the session provider is a real
    # multi-provider cost. When True, the shared adapter recalls up to 5
    # query-relevant memory files and injects their bodies as a
    # <system-reminder>; the static MEMORY.md index injection is unaffected.
    memory_relevance_prefetch_enabled: bool = False

    # Bounded persistent memory + self-improvement review (hermes-agent
    # port, src/memory/). Defaults are donor-faithful (hermes ships the
    # store AND the post-turn review on by default; the review's request is
    # engineered to hit the parent's warm prompt-cache prefix, every write
    # is surfaced via a notification line, and the store is a bounded
    # ≤2,200-char file — visible, reversible, cheap).
    memory_store_enabled: bool = True  # MEMORY.md snapshot block + Memory tool
    user_profile_enabled: bool = True  # USER.md snapshot block
    memory_char_limit: int = 2200  # MEMORY.md budget (chars, not tokens)
    user_char_limit: int = 1375  # USER.md budget
    # Fire a background self-improvement review every N real user turns
    # (0 disables). Donor: memory.nudge_interval, default 10.
    memory_review_interval: int = 10
    # Review notification verbosity: off | on | verbose (donor:
    # display.memory_notifications). "off" suppresses the transcript line
    # while the review still runs and writes.
    memory_notifications: str = "on"
    # Stage memory writes for user approval instead of committing
    # (src/memory/write_approval.py). Donor default: off.
    memory_write_approval: bool = False

    # Provider
    provider: str = "anthropic"

    # Permission mode — back-compat reading channel.
    # Reads through ``permissions.defaultMode`` first; this top-level
    # field is kept for older binaries that wrote the mode outside the
    # ``permissions`` block. Empty string means "unset" and is skipped
    # by ``validate_settings``. will mark this deprecated.
    permission_mode: str = ""

    # Permission configuration — dict-shaped block.
    # Replaces the legacy ``list[PermissionRule]`` (deleted in Sub-H).
    # Matches on-disk format produced by
    # ``src/permissions/updates.py:persist_permission_update`` and the
    # TS upstream ``permissions.*`` contract.
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)

    # Tool settings
    tools: dict[str, ToolSettings] = field(default_factory=dict)

    # Output
    output_style: OutputStyleSettings = field(default_factory=OutputStyleSettings)

    # Sandbox (parsed but enforcement deferred — see SandboxSettings + C8)
    sandbox: SandboxSettings | None = None

    # Spinner verbs (None = built-in defaults)
    spinner_verbs: SpinnerVerbsSettings | None = None

    # Compact
    compact: CompactSettings = field(default_factory=CompactSettings)

    # P108-E — freeze / layer-1 / layer-2 budgets. Declared as a
    # structured block (mirrors ``compact``) so per-knob reset / env-var
    # resolution stays inside the dataclass — ``from_dict`` reads it back
    # automatically when the on-disk JSON matches the dataclass shape.
    freeze: FreezeSettings = field(default_factory=FreezeSettings)

    # Hooks
    hooks: HookSettings = field(default_factory=HookSettings)

    # MCP servers
    mcp_servers: dict[str, McpServerSettings] = field(default_factory=dict)

    # Max turns
    max_turns: int = 0  # 0 = unlimited

    # Max cost (USD)
    max_cost_usd: float = 0.0  # 0 = unlimited

    # Effort
    effort: str = ""  # "", "low", "medium", "high", "xhigh", "max"

    # Plan mode
    plan_mode: bool = False

    # Non-interactive / SDK mode
    non_interactive: bool = False

    # Custom system prompt
    custom_system_prompt: str = ""

    # Append system prompt
    append_system_prompt: str = ""

    # Allowed tools list (empty = all)
    allowed_tools: list[str] = field(default_factory=list)

    # Denied tools list
    denied_tools: list[str] = field(default_factory=list)

    # Fast mode (use small model)
    fast_mode: bool = False

    # Disable dynamic workflows (also honored via CLAUDE_CODE_DISABLE_WORKFLOWS
    # and the camelCase ``disableWorkflows`` JSON key). See src/workflow/gating.py.
    disable_workflows: bool = False

    # Session retention days
    session_retention_days: int = 30

    # REPL/TUI key for accepting the ghost-text suggestion in the input
    # box. Stored in prompt_toolkit's key notation (``"c-e"``, ``"tab"``,
    # ``"c-j"``, ``"f2"`` …) and translated to Textual's event.key form
    # at the TUI call site. Default ``"c-e"`` matches the upstream
    # ``AutoSuggestFromHistory`` accept key.
    accept_suggestion_key: str = "c-e"

    # Whether to also bind ``Tab`` as a context-aware secondary accept
    # key (Plan 4 opt-out). When true (default), the REPL registers a
    # filtered Tab binding and the TUI intercepts Tab in
    # ``on_key`` only while a ghost-text suggestion is visible,
    # mirroring the upstream ``useTypeahead`` Autocomplete-context
    # behaviour. Set to false to keep Tab exclusively bound to the
    # framework's default (focus_next / completion-menu cycling).
    accept_suggestion_tab_alias: bool = True

    # ── Voice Mode ───────────────────────────────────────────────
    # Mirrors TS ``settings.voiceProvider`` (``utils/settings/types.ts``).
    # ``""`` (default) = unset; persisted values: ``"anthropic"`` | ``"doubao"``.
    # The active STT backend is selected by :func:`clawcodex_ext.services.voice.
    # voice_mode_enabled.get_voice_provider`; the empty string is treated as
    # "anthropic" at read time so a fresh install that hasn't run ``/voice``
    # still has a defined default.
    voice_provider: str = ""
    # Master on/off switch written by ``/voice`` (no-arg toggle). Mirrors TS
    # ``settings.voiceEnabled`` (default false). Distinct from ``voice_provider``
    # — provider records the *chosen backend*, this records whether the user
    # has opted into voice input at all. Decoupled so ``/voice doubao`` can
    # flip both atomically while ``/voice anthropic`` only touches the provider.
    voice_enabled: bool = False

    # ── P64-E TTS (Text-to-Speech) ────────────────────────────────
    tts_provider: str = ""
    tts_enabled: bool = False
    tts_voice: str = ""
    tts_silent_text_output: bool = False

    # ── P65-D Voice Dialogue (full-duplex) ─────────────────────────
    # Decoupled from the voice_* / tts_* knobs so the two voice modes
    # can ship independently: a user can keep their STT setup while
    # ignoring dialogue, and vice versa. Defaults match the plan
    # ("minimax" primary backend, text-modality for the first iteration so
    # the caller routes through its own LLM before TTS — easier to debug
    # than the audio modality in MVP).
    dialogue_provider: str = ""  # "" | "minimax" | "openai-realtime"
    dialogue_enabled: bool = False
    dialogue_voice: str = ""  # backend-specific TTS voice id
    dialogue_modality: str = "text"  # "text" | "audio" — output modality
    dialogue_interim_results: bool = True  # forward to provider

    # Extra raw fields for forward compatibility
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        import dataclasses

        d = dataclasses.asdict(self)
        extra = d.pop("extra", {})
        d["permissions"] = self.permissions.to_dict()
        d.update(extra)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SettingsSchema:
        """Deserialize from dict.

        ``permissions`` is now a structured dict (PermissionsConfig)
        matching the on-disk + TS upstream contract. ``from_dict`` accepts
        any of dict / list / None / missing and degrades safely to an
        empty :class:`PermissionsConfig`; unknown sub-keys land in
        ``PermissionsConfig.additional`` (forward-compat bag).

        ``SettingsSchema.extra`` continues to receive *unknown top-level*
        keys (sub-keys of ``permissions`` are NOT extra — they are owned by
        ``PermissionsConfig``).
        """
        import dataclasses

        data = dict(data)
        if "modelLimits" in data:
            # Defaults are materialized in snake_case before user settings are
            # merged, so both keys can coexist. The explicit camelCase user
            # value must replace that default rather than being relegated to
            # ``extra``.
            data["model_limits"] = data.pop("modelLimits")
        known_fields = {f.name for f in dataclasses.fields(cls)}
        known: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in data.items():
            if k in known_fields:
                known[k] = v
            else:
                extra[k] = v

        # Convert nested objects
        if "permissions" in known:
            known["permissions"] = PermissionsConfig.from_dict(known["permissions"])
        if "output_style" in known and isinstance(known["output_style"], dict):
            known["output_style"] = OutputStyleSettings(**known["output_style"])
        # ``sandbox`` — TS's SandboxSettingsSchema uses camelCase + .passthrough();
        # accept both cases and IGNORE unknown keys (the enforcement-only fields
        # the port doesn't act on) so a valid settings.json never fails to load.
        if "sandbox" in known and isinstance(known["sandbox"], dict):
            _sb = known["sandbox"]
            known["sandbox"] = SandboxSettings(
                enabled=bool(_sb.get("enabled", False)),
                fail_if_unavailable=bool(_sb.get("failIfUnavailable", _sb.get("fail_if_unavailable", False))),
                auto_allow_bash_if_sandboxed=bool(
                    _sb.get("autoAllowBashIfSandboxed", _sb.get("auto_allow_bash_if_sandboxed", True))
                ),
                allow_unsandboxed_commands=bool(
                    _sb.get("allowUnsandboxedCommands", _sb.get("allow_unsandboxed_commands", True))
                ),
                excluded_commands=list(_sb.get("excludedCommands", _sb.get("excluded_commands", [])) or []),
                enabled_platforms=list(_sb.get("enabledPlatforms", _sb.get("enabled_platforms", [])) or []),
            )
        if "spinner_verbs" in known and isinstance(known["spinner_verbs"], dict):
            known["spinner_verbs"] = SpinnerVerbsSettings(**known["spinner_verbs"])
        if "compact" in known and isinstance(known["compact"], dict):
            known["compact"] = CompactSettings(**known["compact"])
        if "freeze" in known and isinstance(known["freeze"], dict):
            known["freeze"] = FreezeSettings(**known["freeze"])
        if "model_limits" in known and isinstance(known["model_limits"], dict):
            converted: dict[str, ModelLimitSettings] = {}
            for name, value in known["model_limits"].items():
                if isinstance(value, ModelLimitSettings):
                    converted[str(name)] = value
                elif isinstance(value, dict):
                    converted[str(name)] = ModelLimitSettings(
                        context_window=value.get("contextWindow", value.get("context_window")),
                        max_output_tokens=value.get("maxOutputTokens", value.get("max_output_tokens")),
                    )
            known["model_limits"] = converted
        if "hooks" in known and isinstance(known["hooks"], dict):
            known["hooks"] = HookSettings(**known["hooks"])
        if "tools" in known and isinstance(known["tools"], dict):
            known["tools"] = {
                name: ToolSettings(**v) if isinstance(v, dict) else v for name, v in known["tools"].items()
            }
        if "mcp_servers" in known and isinstance(known["mcp_servers"], dict):
            known["mcp_servers"] = {
                name: McpServerSettings(**v) if isinstance(v, dict) else v for name, v in known["mcp_servers"].items()
            }

        known["extra"] = extra
        return cls(**known)
