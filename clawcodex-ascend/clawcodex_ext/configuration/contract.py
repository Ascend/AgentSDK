#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""Single source of truth for user-facing ClawCodex configuration.

The runtime settings dataclass owns the canonical settings roots.  This module
adds the small amount of metadata that cannot be inferred from Python types:
synthetic settings files, downstream extension sections, stable app fields,
and configuration surfaces that must be managed by a dedicated subsystem.
"""

from __future__ import annotations

import copy
import dataclasses
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints


_SCOPES = ("user", "project", "local")


@dataclass(frozen=True, slots=True)
class ConfigurationFieldSpec:
    """Metadata for one public top-level configuration field."""

    name: str
    domain: str
    schema: Mapping[str, Any]
    scopes: tuple[str, ...] = _SCOPES
    description: str = ""
    secret: bool = False
    managed_by: str | None = None

    def schema_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.schema))
        result.setdefault("description", self.description or _humanize(self.name))
        result["x-scopes"] = list(self.scopes)
        result["x-secret"] = self.secret
        if self.managed_by:
            result["readOnly"] = True
            result["x-managed-by"] = self.managed_by
        return result


_SETTINGS_DESCRIPTIONS: dict[str, str] = {
    "model": "Active harness model; pair with model_provider.",
    "model_provider": "Provider key that owns the persisted model selection.",
    "advisor_model": "Reviewer model; pair with advisor_provider.",
    "advisor_provider": "Provider key used for advisor requests.",
    "advisor_enabled": "Master switch for the advisor reviewer.",
    "advisor_client_mode": "Force advisor execution through a separate client request.",
    "provider": "Active provider for the harness session.",
    "permissions": "Tool allow, deny, ask, and default-mode policy.",
    "hooks": "Event hook arrays keyed by ClawCodex hook event.",
    "hookRuntime": "Hook executor runtime limits and master switch.",
    "env": "Environment variables applied by trusted settings layers.",
    "freeze": "Agent, turn, tool, permission, and watchdog timeout controls.",
    "spinner_verbs": "Custom spinner verb merge mode and values.",
    "disable_workflows": "Disable dynamic workflow discovery and execution.",
    "voice_provider": "Push-to-talk speech-to-text provider.",
    "voice_enabled": "Master switch for push-to-talk voice input.",
    "tts_provider": "Text-to-speech provider.",
    "tts_enabled": "Master switch for text-to-speech output.",
    "tts_voice": "Provider-specific text-to-speech voice identifier.",
    "tts_silent_text_output": "Suppress text display while TTS is active.",
    "dialogue_provider": "Full-duplex dialogue provider.",
    "dialogue_enabled": "Master switch for full-duplex dialogue.",
    "dialogue_voice": "Provider-specific dialogue voice identifier.",
    "dialogue_modality": "Dialogue output modality.",
    "dialogue_interim_results": "Forward interim dialogue recognition results.",
    "away_summary": "Automatic away recap behavior.",
    "intent_forecast": "Idle-session intent forecast behavior.",
    "agentRouting": "Legacy agent routing rules consumed by the API provider bridge.",
    "agentModels": "Legacy per-agent model map consumed by the API provider bridge.",
}


_SETTINGS_OVERRIDES: dict[str, dict[str, Any]] = {
    "effort": {"enum": ["", "low", "medium", "high", "max"]},
    "permission_mode": {"enum": ["", "default", "plan", "acceptEdits", "bypassPermissions", "dontAsk"]},
    "voice_provider": {"enum": ["", "anthropic", "doubao"]},
    "tts_provider": {"enum": ["", "openai", "minimax", "gemini"]},
    "dialogue_provider": {"enum": ["", "minimax", "openai-realtime"]},
    "dialogue_modality": {"enum": ["text", "audio"]},
    "max_turns": {"minimum": 0},
    "max_cost_usd": {"minimum": 0},
    "session_retention_days": {"minimum": 1},
}


_APP_FIELD_DEFINITIONS: tuple[tuple[str, dict[str, Any], tuple[str, ...], str], ...] = (
    (
        "default_provider",
        {"type": "string"},
        _SCOPES,
        "Default provider used when no session override is supplied.",
    ),
    (
        "providers",
        {
            "type": "object",
            "default": {},
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "x-secret": True},
                    "base_url": {"type": "string"},
                    "default_model": {"type": "string"},
                    "models": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
        },
        _SCOPES,
        "Named provider endpoints, credentials, and known models.",
    ),
    (
        "session",
        {
            "type": "object",
            "properties": {
                "auto_save": {"type": "boolean", "default": True},
                "max_history": {"type": "integer", "minimum": 1, "default": 100},
            },
            "additionalProperties": True,
            "default": {"auto_save": True, "max_history": 100},
        },
        ("user",),
        "Session persistence defaults.",
    ),
    (
        "theme",
        {"type": "string", "enum": ["auto", "dark", "light", "claude"], "default": "dark"},
        ("user",),
        "TUI and REPL theme name.",
    ),
    (
        "logoColor",
        {
            "type": "string",
            "enum": ["sunset", "forest", "ocean", "monochrome"],
            "default": "sunset",
        },
        ("user",),
        "Startup logo palette name.",
    ),
    (
        "editorMode",
        {"type": "string", "enum": ["normal", "vim"], "default": "normal"},
        ("user",),
        "Interactive editor keybinding mode.",
    ),
    (
        "copyFullResponse",
        {"type": "boolean", "default": False},
        ("user",),
        "Copy the complete assistant response by default.",
    ),
    (
        "selection_mode",
        {"type": "string", "enum": ["arrow", "number"], "default": "arrow"},
        ("user",),
        "AskUserQuestion selection input mode.",
    ),
    (
        "agent_models",
        {"type": "object", "additionalProperties": {"type": "string"}, "default": {}},
        _SCOPES,
        "Per-agent model selection map.",
    ),
    (
        "companion",
        {"type": "object", "additionalProperties": True, "default": {}},
        ("user",),
        "Companion character configuration.",
    ),
    (
        "companion_muted",
        {"type": "boolean", "default": False},
        ("user",),
        "Mute companion reactions without deleting its configuration.",
    ),
)


_MANAGED_FIELDS: dict[tuple[str, str], str] = {
    ("settings", "mcp_servers"): "clawcodex mcp",
    ("app", "telemetry"): "/telemetry or telemetry.toml",
    ("app", "plugins"): "plugin loader/installer",
    ("app", "daemon"): "clawcodex daemon",
}


_settings_extensions: dict[str, tuple[type[Any] | None, dict[str, Any] | None, str]] = {}
_builtins_registered = False


def _humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().capitalize() + "."


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return copy.deepcopy(value)
    return None


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        return {"type": "array", "items": _annotation_schema(args[0] if args else Any)}
    if origin is dict:
        return {
            "type": "object",
            "additionalProperties": _annotation_schema(args[1] if len(args) > 1 else Any),
        }
    if origin in {Union, types.UnionType}:
        non_none = [arg for arg in args if arg is not types.NoneType]
        if len(non_none) == 1 and len(non_none) != len(args):
            return {"anyOf": [_annotation_schema(non_none[0]), {"type": "null"}]}
        return {"anyOf": [_annotation_schema(arg) for arg in args]}
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _dataclass_schema(annotation)
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    return {}


def _dataclass_schema(cls: type[Any]) -> dict[str, Any]:
    hints = get_type_hints(cls)
    properties: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.name in {"extra", "additional"}:
            continue
        child = _annotation_schema(hints.get(field.name, Any))
        if field.default is not dataclasses.MISSING:
            child["default"] = _json_default(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            child["default"] = _json_default(field.default_factory())
        properties[field.name] = child
    return {"type": "object", "properties": properties, "additionalProperties": True}


def register_settings_extension(
    name: str,
    *,
    schema_type: type[Any] | None = None,
    schema: Mapping[str, Any] | None = None,
    description: str = "",
) -> None:
    """Register a JSON-backed ``settings.<name>`` extension contract."""

    if not name or "." in name:
        raise ValueError("settings extension name must be a non-empty root key")
    if schema_type is None and schema is None:
        raise ValueError("settings extension requires schema_type or schema")
    _settings_extensions[name] = (
        schema_type,
        copy.deepcopy(dict(schema)) if schema is not None else None,
        description,
    )


def _ensure_builtin_extensions() -> None:
    global _builtins_registered
    if _builtins_registered:
        return
    # pylint: disable=no-name-in-module  # away_summary/intent_forecast: pending patch migration
    from clawcodex_ext.away_summary.config import AwaySummaryConfig
    from clawcodex_ext.intent_forecast.config import IntentForecastConfig

    register_settings_extension(
        "away_summary",
        schema_type=AwaySummaryConfig,
        description=_SETTINGS_DESCRIPTIONS["away_summary"],
    )
    register_settings_extension(
        "intent_forecast",
        schema_type=IntentForecastConfig,
        description=_SETTINGS_DESCRIPTIONS["intent_forecast"],
    )
    register_settings_extension(
        "agentRouting",
        schema={"type": "object", "additionalProperties": True},
        description=_SETTINGS_DESCRIPTIONS["agentRouting"],
    )
    register_settings_extension(
        "agentModels",
        schema={"type": "object", "additionalProperties": {"type": "string"}},
        description=_SETTINGS_DESCRIPTIONS["agentModels"],
    )
    _builtins_registered = True


def _permissions_schema() -> dict[str, Any]:
    rules = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "allow": rules,
            "deny": rules,
            "ask": rules,
            "defaultMode": {
                "type": "string",
                "enum": ["default", "plan", "acceptEdits", "bypassPermissions", "dontAsk"],
            },
            "additionalDirectories": {"type": "array", "items": {"type": "string"}},
            "allowBypassPermissionsMode": {"type": "boolean"},
        },
        "additionalProperties": True,
        "default": {
            "allow": [],
            "deny": [],
            "ask": [],
            "allowBypassPermissionsMode": False,
        },
    }


def _settings_field_specs() -> list[ConfigurationFieldSpec]:
    from clawcodex_ext.settings.types import SettingsSchema

    _ensure_builtin_extensions()
    schema = _dataclass_schema(SettingsSchema)
    properties = schema["properties"]
    properties["permissions"] = _permissions_schema()
    properties["env"] = {"type": "object", "additionalProperties": {"type": "string"}}
    properties["env"]["default"] = {}
    properties["hookRuntime"] = {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "default": True},
            "timeout_ms": {"type": "integer", "minimum": 1000, "default": 30000},
            "max_concurrent": {"type": "integer", "minimum": 1, "default": 5},
        },
        "additionalProperties": False,
        "default": {"enabled": True, "timeout_ms": 30000, "max_concurrent": 5},
    }
    # ``hooks`` is replaced with the event schema by update-config, because
    # importing hook definitions here would create a configuration/hook cycle.
    properties["hooks"] = {"type": "object", "additionalProperties": {"type": "array"}}
    properties["hooks"]["default"] = {}
    for name, (schema_type, explicit, description) in _settings_extensions.items():
        if explicit is not None:
            child = copy.deepcopy(explicit)
        elif schema_type is not None:
            child = _dataclass_schema(schema_type)
        else:  # Guard the registry invariant if internal callers bypass registration.
            continue
        if schema_type is not None:
            child.setdefault("default", dataclasses.asdict(schema_type()))
        else:
            child.setdefault("default", {})
        child.setdefault("description", description or _humanize(name))
        properties[name] = child

    nested_overrides: dict[str, dict[str, dict[str, Any]]] = {
        "output_style": {
            "style": {"enum": ["default", "concise", "verbose", "markdown"]},
            "max_width": {"minimum": 40},
        },
        "spinner_verbs": {"mode": {"enum": ["append", "replace"]}},
        "compact": {
            "threshold_tokens": {"minimum": 1000},
            "max_compact_retries": {"minimum": 0},
        },
        "freeze": {
            "agent_loop_timeout_s": {"minimum": 0},
            "turn_timeout_s": {"minimum": 0},
            "tool_timeout_s": {"minimum": 0},
            "permission_timeout_s": {"minimum": 0},
            "threshold_s": {"minimum": 0},
        },
        "away_summary": {
            "idle_seconds": {"minimum": 1},
            "min_turns": {"minimum": 0},
            "max_input_tokens": {"minimum": 256},
            "max_output_tokens": {"minimum": 64},
            "response_language": {
                "enum": [
                    "auto",
                    "Chinese",
                    "English",
                    "zh",
                    "zh-cn",
                    "chinese",
                    "中文",
                    "en",
                    "en-us",
                    "english",
                ]
            },
        },
        "intent_forecast": {
            "idle_seconds": {"minimum": 1},
            "max_sessions": {"minimum": 0},
            "max_transcript_tail_messages": {"minimum": 0},
            "max_input_tokens": {"minimum": 512},
            "max_output_tokens": {"minimum": 64},
            "min_confidence": {"minimum": 0, "maximum": 1},
            "response_language": {
                "enum": [
                    "auto",
                    "Chinese",
                    "English",
                    "zh",
                    "zh-cn",
                    "chinese",
                    "中文",
                    "en",
                    "en-us",
                    "english",
                ]
            },
            "intent_strategy": {
                "enum": [
                    "user",
                    "user_first",
                    "user_priority",
                    "workspace",
                    "workspace_first",
                    "workspace_priority",
                    "project",
                    "project_first",
                    "history",
                    "history_first",
                    "history_priority",
                    "session",
                ]
            },
        },
    }
    for root, children in nested_overrides.items():
        root_schema = properties.get(root, {})
        child_properties = root_schema.get("properties", {})
        for child_name, override in children.items():
            if child_name in child_properties:
                child_properties[child_name].update(copy.deepcopy(override))

    specs: list[ConfigurationFieldSpec] = []
    for name, child in properties.items():
        child.update(copy.deepcopy(_SETTINGS_OVERRIDES.get(name, {})))
        managed_by = _MANAGED_FIELDS.get(("settings", name))
        specs.append(
            ConfigurationFieldSpec(
                name=name,
                domain="settings",
                schema=child,
                description=_SETTINGS_DESCRIPTIONS.get(name, _humanize(name)),
                managed_by=managed_by,
            )
        )
    return specs


def _app_field_specs() -> list[ConfigurationFieldSpec]:
    specs = [
        ConfigurationFieldSpec(
            name=name,
            domain="app",
            schema=copy.deepcopy(schema),
            scopes=scopes,
            description=description,
        )
        for name, schema, scopes, description in _APP_FIELD_DEFINITIONS
    ]
    for name in ("telemetry", "plugins", "daemon"):
        specs.append(
            ConfigurationFieldSpec(
                name=name,
                domain="app",
                schema={"type": "object", "additionalProperties": True},
                description=f"Configuration owned by {_MANAGED_FIELDS[('app', name)]}.",
                managed_by=_MANAGED_FIELDS[("app", name)],
            )
        )
    return specs


def get_configuration_contract() -> tuple[ConfigurationFieldSpec, ...]:
    """Return the immutable public configuration contract."""

    return tuple(_app_field_specs() + _settings_field_specs())


def get_configuration_field(setting: str) -> ConfigurationFieldSpec | None:
    root = setting.split(".", 1)[0] if setting else ""
    return next((field for field in get_configuration_contract() if field.name == root), None)


def infer_configuration_domain(setting: str) -> str:
    """Infer a domain from the canonical contract; unknown roots remain app-compatible."""

    field = get_configuration_field(setting)
    return field.domain if field is not None else "app"


def managed_configuration_route(setting: str) -> str | None:
    field = get_configuration_field(setting)
    return field.managed_by if field is not None else None


def configuration_json_schema(domain: str) -> dict[str, Any]:
    """Build a stable, value-free JSON Schema for one configuration domain."""

    if domain not in {"app", "settings"}:
        raise ValueError(f"unknown configuration domain: {domain!r}")
    properties = {field.name: field.schema_dict() for field in get_configuration_contract() if field.domain == domain}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"ClawCodex {domain} configuration",
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }


def _type_matches(expected: str, value: Any) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _value_errors(schema: Mapping[str, Any], value: Any, path: str) -> list[str]:
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        candidate_errors = [
            _value_errors(candidate, value, path) for candidate in alternatives if isinstance(candidate, Mapping)
        ]
        if any(not errors for errors in candidate_errors):
            return []
        return candidate_errors[0] if candidate_errors else []

    expected = schema.get("type")
    if isinstance(expected, str) and not _type_matches(expected, value):
        return [f"{path} must be {expected}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} must be one of {schema['enum']!r}"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            return [f"{path} must be >= {minimum}"]
        if maximum is not None and value > maximum:
            return [f"{path} must be <= {maximum}"]
    if isinstance(value, Mapping):
        properties = schema.get("properties")
        additional = schema.get("additionalProperties", True)
        if isinstance(properties, Mapping):
            for key, child in value.items():
                child_schema = properties.get(key)
                if isinstance(child_schema, Mapping):
                    errors = _value_errors(child_schema, child, f"{path}.{key}")
                    if errors:
                        return errors
                elif additional is False:
                    return [f"{path}.{key} is not a supported field"]
                elif isinstance(additional, Mapping):
                    errors = _value_errors(additional, child, f"{path}.{key}")
                    if errors:
                        return errors
    if isinstance(value, (list, tuple)) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            errors = _value_errors(schema["items"], item, f"{path}[{index}]")
            if errors:
                return errors
    return []


def validate_configuration_document(domain: str, data: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate known public fields while preserving unknown forward-compatible keys."""

    schema = configuration_json_schema(domain)
    errors: list[str] = []
    properties = schema["properties"]
    for key, value in data.items():
        child = properties.get(key)
        if isinstance(child, Mapping):
            errors.extend(_value_errors(child, value, f"{domain}.{key}"))
            if errors:
                break
    return tuple(errors)


__all__ = [
    "ConfigurationFieldSpec",
    "configuration_json_schema",
    "get_configuration_contract",
    "get_configuration_field",
    "infer_configuration_domain",
    "managed_configuration_route",
    "register_settings_extension",
    "validate_configuration_document",
]
