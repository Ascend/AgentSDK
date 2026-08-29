#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Bundled skill for safe, scoped ClawCodex configuration updates."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from clawcodex_ext.configuration import (  # pylint: disable=no-name-in-module
    configuration_json_schema,
    get_configuration_contract,
    get_configuration_snapshot,
)
from clawcodex_ext.hooks.hook_types import ALL_HOOK_EVENTS  # pylint: disable=no-name-in-module

from ..bundled_skills import BundledSkillDefinition, register_bundled_skill


# Bandit B105 is a false positive: this is an internal routing sentinel.
_HOOKS_ONLY_TOKEN = "[hooks-only]"  # nosec B105


def _hook_variant(
    hook_type: str,
    required_field: str,
    type_properties: dict[str, Any],
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "type": {"type": "string", "enum": [hook_type]},
        "matcher": {"type": "string"},
        "timeout": {"type": "integer", "minimum": 1, "description": "Milliseconds"},
        "if": {"type": "string", "description": "Permission-rule condition"},
    }
    properties.update(type_properties)
    return {
        "type": "object",
        "properties": properties,
        "required": [required_field],
        "additionalProperties": False,
    }


def _hook_item_schema() -> dict[str, Any]:
    return {
        "oneOf": [
            _hook_variant(
                "command",
                "command",
                {
                    "command": {"type": "string"},
                    "shell": {"type": "string", "enum": ["bash", "powershell"]},
                },
            ),
            _hook_variant("http", "url", {"url": {"type": "string", "format": "uri"}}),
            _hook_variant("prompt", "promptText", {"promptText": {"type": "string"}}),
            _hook_variant(
                "agent",
                "agentInstructions",
                {"agentInstructions": {"type": "string"}},
            ),
        ]
    }


def _hooks_schema() -> dict[str, Any]:
    event_schema = {"type": "array", "items": {"$ref": "#/$defs/hook"}}
    return {
        "type": "object",
        "properties": {event: event_schema for event in ALL_HOOK_EVENTS},
        "additionalProperties": False,
    }


def _settings_json_schema() -> dict[str, Any]:
    schema = configuration_json_schema("settings")
    properties = schema["properties"]
    properties["hooks"] = _hooks_schema()
    schema["title"] = "ClawCodex settings.json"
    schema["$defs"] = {"hook": _hook_item_schema()}
    return schema


def _app_json_schema() -> dict[str, Any]:
    """Return the stable public app schema without exposing live values."""

    schema = configuration_json_schema("app")
    schema["title"] = "ClawCodex app config"
    return schema


def _snapshot_for_context(context: Any | None):
    cwd = getattr(context, "cwd", None) or getattr(context, "workspace_root", None) or Path.cwd()
    return get_configuration_snapshot(cwd)


def _path_table(context: Any | None, *, hooks_only: bool) -> str:
    snapshot = _snapshot_for_context(context)
    rows = ["| Scope | Domain | Path | Intended use |", "|---|---|---|---|"]
    uses = {
        "user": "Personal defaults for every workspace",
        "project": "Shared project configuration; normally committed",
        "local": "Personal project override; normally gitignored",
    }
    for layer in snapshot.layers:
        if not hooks_only:
            rows.append(f"| {layer.scope.value} | app | {layer.app_path} | {uses[layer.scope.value]} |")
        rows.append(f"| {layer.scope.value} | settings | {layer.settings_path} | {uses[layer.scope.value]} |")
    return "\n".join(rows)


def _hook_reference() -> str:
    events = ", ".join(ALL_HOOK_EVENTS)
    hook_schema = json.dumps(_hook_item_schema(), indent=2, sort_keys=True)
    # Bandit B608 is a false positive: this is documentation prompt text, not SQL.
    return f"""## Hooks: event automation, not memory

Use a hook when the user wants ClawCodex to act automatically before, after,
when, whenever, or every time an event occurs. A remembered preference cannot
execute future behavior.

ClawCodex stores hooks as a direct array for each event. Do not copy Claude
Code's nested matcher-group `hooks` array shape:

```json
{{
  "hooks": {{
    "PostToolUse": [
      {{
        "type": "command",
        "matcher": "Write|Edit",
        "command": "ruff format .",
        "shell": "powershell",
        "timeout": 30000
      }}
    ]
  }}
}}
```

The `timeout` field is milliseconds. Choose `shell: "powershell"` or
`shell: "bash"` explicitly when the command is shell-specific. Build commands
for the current platform ({platform.system()}) and this project's actual package
manager, virtual environment, and scripts; do not assume `jq`, `npx`, or a
POSIX shell exists.

Primary events and matchers:

| Event | Typical matcher/use |
|---|---|
| PermissionRequest | Tool name; before a permission prompt |
| PreToolUse | Tool name; inspect, allow, deny, ask, or update input |
| PostToolUse | Tool name; after a successful tool call |
| PostToolUseFailure | Tool name; after a failed tool call |
| UserPromptSubmit | No matcher normally; process submitted input |
| PreCompact / PostCompact | `manual` or `auto` when relevant |
| SessionStart / SessionEnd | Session lifecycle |
| Stop / StopFailure | Run completion lifecycle |
| Notification | Notification type |

All supported events: {events}

Hook types:

- `command`: requires `command`; optionally set `shell`, `matcher`, `timeout`, `if`.
- `http`: requires `url`; SSRF checks and normal hook trust rules apply.
- `prompt`: requires `promptText`.
- `agent`: requires `agentInstructions`.

Representative tool-hook input contains `tool_name`, `tool_input`, and for
post-use events `tool_response`. Command hooks may emit a strict JSON object
with these ClawCodex fields: `decision` (`allow`, `deny`, or `ask`), `reason`,
`updatedInput`, `additionalContexts`, `preventContinuation`, `stopReason`, and
`updatedMCPToolOutput`. Unknown fields invalidate the output payload.

## Hook JSON Schema

```json
{hook_schema}
```

## Construct and verify a hook

1. Read the exact target settings file and check the same event plus matcher.
   If a related hook exists and intent is unclear, ask whether to keep,
   replace, or add alongside it.
2. Construct the command for this repository and platform. Quote paths and
   parse stdin JSON safely. Keep the raw command observable at first; do not
   hide errors or append unconditional success handling before it passes.
3. When a normal shell execution tool is available and permission is granted,
   pipe a representative JSON payload into the raw command. Check its exit
   code and intended side effect. Shell tools may test a command but must never
   write configuration.
4. Persist only through Config. Add one hook with `append_unique` on
   `hooks.<Event>`. For replacement, `set` the complete event array assembled
   from the exact target file so unrelated hooks survive.
5. Re-read both the effective value with Config and the exact target file with
   Read. Config performs JSON/schema validation and atomic replacement.
6. If the event is safe to trigger now, trigger it once and verify the actual
   effect. Use an observable, temporary sentinel when needed and always clean
   it up. If verification fails, restore the prior value through Config.
7. Report the scope, file, operation, and verification result. State clearly
   when an event cannot fire until a later lifecycle transition.
"""  # nosec B608


def _configuration_reference(context: Any | None) -> str:
    paths = _path_table(context, hooks_only=False)
    contract = get_configuration_contract()
    app_fields = ", ".join(sorted(field.name for field in contract if field.domain == "app"))
    managed_routes = "\n".join(f"- `{field.name}`: {field.managed_by}" for field in contract if field.managed_by)
    app_schema = json.dumps(
        _app_json_schema(),
        indent=2,
        sort_keys=True,
    )
    settings_schema = json.dumps(_settings_json_schema(), indent=2, sort_keys=True)
    call_patterns = "\n".join(
        json.dumps(example, separators=(",", ":"))
        for example in (
            {
                "setting": "model",
                "value": "gpt-test",
                "scope": "project",
                "domain": "settings",
                "operation": "set",
            },
            {
                "setting": "default_provider",
                "value": "openai",
                "scope": "user",
                "domain": "app",
                "operation": "set",
            },
            {
                "setting": "permissions.allow",
                "value": "Bash(npm:*)",
                "scope": "project",
                "domain": "settings",
                "operation": "append_unique",
            },
            {
                "setting": "env.DEBUG",
                "value": "true",
                "scope": "local",
                "domain": "settings",
                "operation": "set",
            },
            {
                "setting": "voice_provider",
                "value": "doubao",
                "scope": "user",
                "domain": "settings",
                "operation": "set",
            },
            {
                "setting": "away_summary.idle_seconds",
                "value": 600,
                "scope": "project",
                "domain": "settings",
                "operation": "set",
            },
            {
                "setting": "hooks.PostToolUse",
                "value": {
                    "type": "command",
                    "matcher": "Write|Edit",
                    "command": "ruff format .",
                    "shell": "powershell",
                    "timeout": 30000,
                },
                "scope": "project",
                "domain": "settings",
                "operation": "append_unique",
            },
        )
    )
    return f"""## Configuration contract

{paths}

Precedence is defaults < user < project < local < CLI. At the same scope,
canonical `settings.json` values override legacy `config.json.settings`.
Never copy a merged snapshot into a user/global file.

- Use the `app` domain for provider, account, API endpoint, credential, and
  process-level ClawCodex configuration. Public app roots: {app_fields}.
- Use the `settings` domain for harness behavior, model selection, environment,
  permissions, hooks, output, compaction, and tool behavior.
- API keys and secrets are user-scope only and Config redacts them.
- Project/local environment, hooks, and permissions require workspace trust.
- Fields with `x-managed-by` are read-only through Config. Route them to the
  dedicated subsystem instead:

{managed_routes}

Permission rules live in `permissions.allow`, `permissions.deny`, and
`permissions.ask`. Use exact tool names (`Read`), scoped forms
(`Bash(git status)`), or prefix forms such as `Bash(npm:*)` and
`Skill(prefix:*)`. Deny wins over allow, followed by ask.

## Functional groups and related fields

- Model selection: update `model` with `model_provider`; `provider` selects the
  active harness provider, while app `default_provider` controls startup.
- Advisor: configure `advisor_model` with `advisor_provider`; use
  `advisor_enabled` as the master switch and `advisor_client_mode` only when a
  separate client-side reviewer call is intended.
- Voice: `voice_*` controls speech input, `tts_*` controls spoken output, and
  `dialogue_*` controls full-duplex conversation. Provider-specific voice IDs
  remain strings, while provider and modality fields use the schema enums.
- Reliability: `compact`, `freeze`, `max_turns`, and `max_cost_usd` control
  context and execution budgets; zero retains each documented unlimited or
  disabled meaning.
- Downstream JSON settings: `away_summary` and `intent_forecast` are normal
  layered settings and may be changed incrementally through Config.

## Config call patterns

Use native JSON values, not JSON-encoded strings:

```json
{call_patterns}
```

A provider definition such as `providers.openai.base_url` belongs to the app
domain; selecting the active `provider` and `model` for the harness belongs to
the settings domain. Credentials remain user-scope app values.

## Public App JSON Schema

This stable schema is generated from the central ClawCodex configuration
contract. It contains public fields, defaults, constraints, scopes, and routing
metadata, but never live values or provider names:

```json
{app_schema}
```

## Full Settings JSON Schema

```json
{settings_schema}
```
"""


def _workflow(request: str) -> str:
    return f"""# Update ClawCodex Configuration

User request: {request}

## Required workflow

1. Translate persistent or event-driven intent into configuration. Requests
   such as "from now on", "whenever", "before", or "after" normally require
   a hook, not memory or a conversational promise.
2. Resolve ambiguity before writing. Use AskUserQuestion only when the answer
   cannot be inferred and changes durable behavior: scope, add versus replace,
   exact value, hook event/type/matcher, or shell command.
3. Choose the narrowest scope (`user`, `project`, or `local`) and the correct
   domain (`app` or `settings`).
4. If the field has `x-managed-by`, stop this Config workflow and direct the
   request to that dedicated command or subsystem.
5. Read the effective setting with Config `get`, then Read the exact target
   file shown below. Missing files are empty documents. Do not assume the
   merged effective value is the target file's content.
6. Make the smallest mutation with Config: prefer `append_unique` for additions,
   `remove` for array/object removal, `unset` for deletion, and `set` only for
   replacement. Pass objects and arrays as native JSON values.
7. Let Config validate and atomically replace the target. Never use Bash,
   PowerShell, Edit, Write, Python, or direct filesystem APIs to persist
   ClawCodex configuration.
8. Re-run Config `get` and Read the exact target file. Verify the requested
   effective value, preserved unknown fields, and correct scope. On failure,
   stop; do not attempt a direct-file fallback.
9. Summarize exactly what changed and where. Never reveal secret values.
"""


def _build_update_config_prompt(args: str, context: Any | None = None) -> str:
    raw_request = (args or "").strip()
    hooks_only = raw_request.startswith(_HOOKS_ONLY_TOKEN)
    if hooks_only:
        raw_request = raw_request[len(_HOOKS_ONLY_TOKEN) :].strip()
    request = raw_request or "Inspect the conversation and determine the requested change."

    workflow = _workflow(request)
    hooks = _hook_reference()
    if hooks_only:
        return (
            f"{workflow}\n\nThis is hooks-only mode. Work only in the settings domain; "
            "`hooks` contains event definitions and `hookRuntime` contains runtime controls.\n\n"
            f"{_path_table(context, hooks_only=True)}\n\n{hooks}"
        )
    return f"{workflow}\n\n{_configuration_reference(context)}\n\n{hooks}"


def register_update_config_skill() -> bool:
    return register_bundled_skill(
        BundledSkillDefinition(
            name="update-config",
            description=(
                "Configure ClawCodex through scoped app/settings files: provider/model/env, "
                "permissions, hooks, output, tools, and other harness behavior. Use when the "
                "user asks ClawCodex to configure itself, enable or persist a feature, change "
                "settings.json/settings.local.json, troubleshoot hooks, allow or deny tools, "
                'or requests future automation such as "from now on", "every time", "when X happens", '
                '"whenever X", or "before/after X". Event automation requires hooks rather '
                "than memory. Route MCP and plugin installation to their dedicated subsystems."
            ),
            when_to_use=(
                "Use for persistent ClawCodex configuration and future, repeated, conditional, "
                "before, or after behavior; including model/provider/env, permission, "
                "settings-file, and hook requests."
            ),
            argument_hint="[configuration request]",
            user_invocable=True,
            allowed_tools=["Config", "Read", "AskUserQuestion"],
            get_prompt_for_command=_build_update_config_prompt,
        )
    )


__all__ = ["register_update_config_skill"]
