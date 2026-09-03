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

"""The canonical ClawCodex configuration implementation.

App configuration and Claude-compatible harness settings intentionally remain
separate on disk.  This module owns their discovery, precedence, safe mutation,
cache invalidation, and projection onto a live ``ToolContext``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

logger = logging.getLogger(__name__)


class ConfigScope(str, Enum):
    USER = "user"
    PROJECT = "project"
    LOCAL = "local"


class ConfigDomain(str, Enum):
    APP = "app"
    SETTINGS = "settings"


class ConfigOperation(str, Enum):
    GET = "get"
    SET = "set"
    UNSET = "unset"
    APPEND_UNIQUE = "append_unique"
    REMOVE = "remove"


class ConfigurationError(ValueError):
    """A structured, user-correctable configuration request failure."""


_MISSING = object()
_CONFIG_SOURCES = frozenset({"userSettings", "projectSettings", "localSettings"})
_SENSITIVE_SETTINGS_ROOTS = frozenset({"env", "hooks", "permissions"})
_SECRET_PARTS = frozenset({"api_key", "apikey", "api-key", "access_token", "token", "secret", "password"})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return copy.deepcopy(value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = _thaw(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _thaw(value)
    return merged


@dataclass(frozen=True, slots=True)
class ConfigurationLayer:
    scope: ConfigScope
    app_path: Path
    settings_path: Path
    app: Mapping[str, Any]
    legacy_settings: Mapping[str, Any]
    settings: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    cwd: Path
    workspace: Path
    app: Mapping[str, Any]
    settings: Mapping[str, Any]
    typed_settings: Mapping[str, Any]
    layers: tuple[ConfigurationLayer, ...]
    diagnostics: tuple[str, ...] = ()

    def app_dict(self) -> dict[str, Any]:
        return _thaw(self.app)

    def settings_dict(self) -> dict[str, Any]:
        return _thaw(self.settings)

    def typed_settings_dict(self) -> dict[str, Any]:
        return _thaw(self.typed_settings)

    def layer(self, scope: ConfigScope | str) -> ConfigurationLayer:
        wanted = ConfigScope(scope)
        for layer in self.layers:
            if layer.scope is wanted:
                return layer
        raise ConfigurationError(f"configuration scope is unavailable: {wanted.value}")


@dataclass(frozen=True, slots=True)
class ConfigMutationRequest:
    setting: str
    value: Any = field(default=_MISSING, repr=False)
    scope: ConfigScope | str = ConfigScope.USER
    domain: ConfigDomain | str | None = None
    operation: ConfigOperation | str | None = None


@dataclass(frozen=True, slots=True)
class ConfigMutationResult:
    success: bool
    operation: ConfigOperation
    setting: str
    scope: ConfigScope
    domain: ConfigDomain
    path: Path
    value: Any = None
    previous_value: Any = None
    new_value: Any = None
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "success": self.success,
            "operation": self.operation.value,
            "setting": self.setting,
            "scope": self.scope.value,
            "domain": self.domain.value,
            "path": str(self.path),
        }
        secret = _is_secret(self.setting)
        if self.operation is ConfigOperation.GET:
            output["value"] = "***REDACTED***" if secret and self.value not in (None, "") else _redact(self.value)
        else:
            output["previousValue"] = (
                "***REDACTED***" if secret and self.previous_value not in (None, "") else _redact(self.previous_value)
            )
            output["newValue"] = (
                "***REDACTED***" if secret and self.new_value not in (None, "") else _redact(self.new_value)
            )
        if self.diagnostics:
            output["diagnostics"] = list(self.diagnostics)
        return output


_cache_lock = threading.RLock()
_snapshot_cache: dict[str, ConfigurationSnapshot] = {}


def _read_json_strict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"configuration file must contain a JSON object: {path}")
    return data


def _read_json_for_snapshot(path: Path, diagnostics: list[str]) -> dict[str, Any]:
    try:
        return _read_json_strict(path)
    except ConfigurationError as exc:
        diagnostics.append(str(exc))
        return {}


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass  # Cleanup is best-effort and must not replace the primary operation result.
        raise


def _workspace_root(cwd: Path) -> Path:
    from src.config import _find_git_root

    return (_find_git_root(cwd) or cwd).resolve()


def _paths(cwd: Path) -> tuple[Path, list[tuple[ConfigScope, Path, Path]]]:
    from src import config as config_mod

    workspace = _workspace_root(cwd)
    config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")).expanduser()
    return workspace, [
        (
            ConfigScope.USER,
            Path(config_mod.get_config_path()).expanduser(),
            config_dir / "settings.json",
        ),
        (
            ConfigScope.PROJECT,
            workspace / ".claude" / "config.json",
            workspace / ".claude" / "settings.json",
        ),
        (
            ConfigScope.LOCAL,
            workspace / ".claude" / "config.local.json",
            workspace / ".claude" / "settings.local.json",
        ),
    ]


def _is_hook_event_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    # pylint: disable=no-name-in-module  # hooks: pending patch migration
    from clawcodex_ext.hooks.hook_types import ALL_HOOK_EVENTS

    return any(key in ALL_HOOK_EVENTS for key in value)


def _split_legacy_settings(data: Mapping[str, Any]) -> dict[str, Any]:
    settings = data.get("settings", {})
    return _thaw(settings) if isinstance(settings, Mapping) else {}


def _typed_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    from src.settings.constants import DEFAULT_SETTINGS

    merged = _deep_merge(dataclasses.asdict(DEFAULT_SETTINGS), raw)
    hooks = merged.pop("hooks", None)
    hook_runtime = merged.pop("hookRuntime", None)
    if isinstance(hook_runtime, Mapping):
        merged["hooks"] = _thaw(hook_runtime)
    elif isinstance(hooks, Mapping):
        runtime_keys = {
            key: _thaw(value) for key, value in hooks.items() if key in {"enabled", "timeout_ms", "max_concurrent"}
        }
        if runtime_keys:
            merged["hooks"] = runtime_keys
        elif not _is_hook_event_mapping(hooks):
            merged["hooks"] = _thaw(hooks)
    return merged


def get_configuration_snapshot(cwd: str | Path | None = None) -> ConfigurationSnapshot:
    resolved_cwd = Path(cwd or Path.cwd()).resolve()
    key = os.path.normcase(str(resolved_cwd))
    with _cache_lock:
        cached = _snapshot_cache.get(key)
        if cached is not None:
            return cached

    diagnostics: list[str] = []
    workspace, path_rows = _paths(resolved_cwd)
    layers: list[ConfigurationLayer] = []
    for scope, app_path, settings_path in path_rows:
        app_data = _read_json_for_snapshot(app_path, diagnostics)
        settings_data = _read_json_for_snapshot(settings_path, diagnostics)
        layers.append(
            ConfigurationLayer(
                scope=scope,
                app_path=app_path,
                settings_path=settings_path,
                app=_freeze({k: v for k, v in app_data.items() if k != "settings"}),
                legacy_settings=_freeze(_split_legacy_settings(app_data)),
                settings=_freeze(settings_data),
            )
        )

    from src.config import get_default_config

    app = get_default_config()
    settings: dict[str, Any] = {}
    for layer in layers:
        app = _deep_merge(app, layer.app)
        settings = _deep_merge(settings, layer.legacy_settings)
        settings = _deep_merge(settings, layer.settings)

    snapshot = ConfigurationSnapshot(
        cwd=resolved_cwd,
        workspace=workspace,
        app=_freeze(app),
        settings=_freeze(settings),
        typed_settings=_freeze(_typed_settings(settings)),
        layers=tuple(layers),
        diagnostics=tuple(diagnostics),
    )
    with _cache_lock:
        _snapshot_cache[key] = snapshot
    return snapshot


def invalidate_configuration(reason: str, workspace: str | Path | None = None) -> None:
    """Invalidate every configuration representation used by the process."""

    del reason
    with _cache_lock:
        if workspace is None:
            _snapshot_cache.clear()
        else:
            target = Path(workspace).resolve()
            for key, snapshot in list(_snapshot_cache.items()):
                if target in (snapshot.workspace, snapshot.cwd):
                    _snapshot_cache.pop(key, None)

    try:
        from src import config as config_mod

        manager = getattr(config_mod, "_default_manager", None)
        if manager is not None:
            manager.invalidate()
    except Exception as exc:  # noqa: BLE001 - invalidation is best-effort
        logger.debug("Could not invalidate the default configuration manager", exc_info=exc)
    try:
        from src.settings.settings import invalidate_settings_cache

        invalidate_settings_cache()
    except Exception as exc:  # noqa: BLE001 - invalidation is best-effort
        logger.debug("Could not invalidate legacy settings cache", exc_info=exc)
    try:
        from clawcodex_ext.settings.pydantic_adapter import invalidate_settings_cache

        invalidate_settings_cache()
    except Exception as exc:  # noqa: BLE001 - invalidation is best-effort
        logger.debug("Could not invalidate Pydantic settings cache", exc_info=exc)


def _parts(setting: str) -> list[str]:
    if not isinstance(setting, str) or not setting.strip():
        raise ConfigurationError("setting must be a non-empty dotted path")
    parts = setting.split(".")
    if any(not part for part in parts):
        raise ConfigurationError(f"invalid dotted setting path: {setting!r}")
    return parts


def _get_value(data: Mapping[str, Any], setting: str, default: Any = None) -> Any:
    current: Any = data
    for part in _parts(setting):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return _thaw(current)


def _parent(data: dict[str, Any], parts: list[str], *, create: bool) -> dict[str, Any]:
    current = data
    for part in parts[:-1]:
        value = current.get(part)
        if value is None and create:
            value = {}
            current[part] = value
        if not isinstance(value, dict):
            raise ConfigurationError(f"encountered non-object at {part!r}")
        current = value
    return current


def _infer_domain(setting: str) -> ConfigDomain:
    from clawcodex_ext.configuration.contract import infer_configuration_domain

    return ConfigDomain(infer_configuration_domain(setting))


def _is_secret_part(part: str) -> bool:
    normalized = part.lower().replace("-", "_")
    return normalized in _SECRET_PARTS or any(
        normalized.endswith(suffix) for suffix in ("_api_key", "_access_token", "_token", "_secret", "_password")
    )


def _is_secret(setting: str) -> bool:
    return any(_is_secret_part(part) for part in _parts(setting))


def _redact(value: Any, key: str | None = None) -> Any:
    if key is not None and _is_secret_part(key) and value not in (None, ""):
        return "***REDACTED***"
    if isinstance(value, Mapping):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact(v) for v in value]
    return copy.deepcopy(value)


def _normalise_permissions(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    normal = _thaw(data)
    nested = normal.get("rules")
    if isinstance(nested, Mapping):
        for behavior in ("allow", "deny", "ask"):
            if behavior not in normal and behavior in nested:
                normal[behavior] = _thaw(nested[behavior])
    return normal


def _validate_settings_file(data: dict[str, Any]) -> None:
    # pylint: disable=no-name-in-module  # hooks: pending patch migration
    from clawcodex_ext.hooks.config_manager import validate_hook_configs
    from src.settings.types import SettingsSchema
    from src.settings.validation import validate_settings

    hooks = data.get("hooks")
    if hooks is not None:
        if not isinstance(hooks, dict):
            raise ConfigurationError("settings.hooks must be an object of hook event arrays")
        hook_errors = [error for error in validate_hook_configs(hooks) if error.severity == "error"]
        if hook_errors:
            first = hook_errors[0]
            raise ConfigurationError(
                f"invalid hook configuration at {first.event}[{first.index}].{first.field}: {first.message}"
            )

    env = data.get("env")
    if env is not None and (
        not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items())
    ):
        raise ConfigurationError("settings.env must be an object of string values")

    permissions = data.get("permissions")
    if permissions is not None:
        normal = _normalise_permissions(permissions)
        for behavior in ("allow", "deny", "ask"):
            values = normal.get(behavior, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ConfigurationError(f"settings.permissions.{behavior} must be an array of strings")

    typed = _typed_settings(data)
    try:
        validation_errors = validate_settings(SettingsSchema.from_dict(typed))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid settings schema: {exc}") from exc
    if validation_errors:
        first = validation_errors[0]
        raise ConfigurationError(f"invalid setting {first.field}: {first.message}")


def _validate_target(domain: ConfigDomain, data: dict[str, Any]) -> None:
    from clawcodex_ext.configuration.contract import validate_configuration_document

    contract_errors = validate_configuration_document(domain.value, data)
    if contract_errors:
        raise ConfigurationError(contract_errors[0])
    if domain is ConfigDomain.SETTINGS:
        _validate_settings_file(data)
    elif "providers" in data and not isinstance(data["providers"], dict):
        raise ConfigurationError("app.providers must be an object")


def _resolve_request(
    request: ConfigMutationRequest,
) -> tuple[ConfigScope, ConfigDomain, ConfigOperation]:
    try:
        scope = ConfigScope(request.scope)
    except ValueError as exc:
        raise ConfigurationError(f"unknown configuration scope: {request.scope!r}") from exc
    try:
        domain = ConfigDomain(request.domain) if request.domain is not None else _infer_domain(request.setting)
    except ValueError as exc:
        raise ConfigurationError(f"unknown configuration domain: {request.domain!r}") from exc
    from clawcodex_ext.configuration.contract import get_configuration_field

    configuration_field = get_configuration_field(request.setting)
    if configuration_field is not None and domain.value != configuration_field.domain:
        raise ConfigurationError(
            "configuration field "
            f"{configuration_field.name!r} belongs to the {configuration_field.domain!r} domain; "
            f"use domain={configuration_field.domain!r}"
        )
    try:
        if request.operation is None:
            operation = ConfigOperation.GET if request.value is _MISSING else ConfigOperation.SET
        else:
            operation = ConfigOperation(request.operation)
    except ValueError as exc:
        raise ConfigurationError(f"unknown configuration operation: {request.operation!r}") from exc
    return scope, domain, operation


def _mutate_document(
    data: dict[str, Any],
    setting: str,
    operation: ConfigOperation,
    value: Any,
) -> tuple[Any, Any]:
    parts = _parts(setting)
    previous = _get_value(data, setting)
    if operation is ConfigOperation.UNSET and previous is None:
        return None, None
    if operation is ConfigOperation.SET:
        _parent(data, parts, create=True)[parts[-1]] = copy.deepcopy(value)
    elif operation is ConfigOperation.UNSET:
        _parent(data, parts, create=False).pop(parts[-1], None)
    elif operation is ConfigOperation.APPEND_UNIQUE:
        parent = _parent(data, parts, create=True)
        current = parent.setdefault(parts[-1], [])
        if not isinstance(current, list):
            raise ConfigurationError(f"append_unique requires an array setting: {setting}")
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item not in current:
                current.append(copy.deepcopy(item))
    elif operation is ConfigOperation.REMOVE:
        parent = _parent(data, parts, create=False)
        current = parent.get(parts[-1])
        if isinstance(current, list):
            values = value if isinstance(value, list) else [value]
            parent[parts[-1]] = [item for item in current if item not in values]
        elif isinstance(current, dict):
            keys = value if isinstance(value, list) else [value]
            for key in keys:
                current.pop(str(key), None)
        else:
            raise ConfigurationError(f"remove requires an array or object setting: {setting}")
    else:
        raise ConfigurationError(f"unsupported write operation: {operation.value}")
    return previous, _get_value(data, setting)


def mutate_configuration(
    request: ConfigMutationRequest,
    tool_context: Any = None,
) -> ConfigMutationResult:
    scope, domain, operation = _resolve_request(request)
    snapshot = get_configuration_snapshot(
        getattr(tool_context, "cwd", None) or getattr(tool_context, "workspace_root", None)
    )
    layer = snapshot.layer(scope)
    path = layer.app_path if domain is ConfigDomain.APP else layer.settings_path

    if operation is not ConfigOperation.GET:
        from clawcodex_ext.configuration.contract import (
            get_configuration_field,
            managed_configuration_route,
        )

        configuration_field = get_configuration_field(request.setting)
        if configuration_field is not None and scope.value not in configuration_field.scopes:
            raise ConfigurationError(
                f"configuration field {configuration_field.name!r} does not support {scope.value!r} scope; "
                f"allowed scopes: {', '.join(configuration_field.scopes)}"
            )
        managed_by = managed_configuration_route(request.setting)
        if managed_by:
            raise ConfigurationError(
                f"configuration field {_parts(request.setting)[0]!r} is managed by "
                f"{managed_by}; do not mutate it through Config"
            )

    if _is_secret(request.setting) and scope is not ConfigScope.USER:
        raise ConfigurationError("API keys and other secrets may only be written at user scope")
    root = _parts(request.setting)[0]
    if (
        operation is not ConfigOperation.GET
        and domain is ConfigDomain.SETTINGS
        and scope in {ConfigScope.PROJECT, ConfigScope.LOCAL}
        and root in _SENSITIVE_SETTINGS_ROOTS
        and not getattr(tool_context, "workspace_trusted", False)
    ):
        raise ConfigurationError(f"workspace trust is required to modify project/local settings.{root}")

    if operation is ConfigOperation.GET:
        merged = snapshot.app if domain is ConfigDomain.APP else snapshot.settings
        value = _get_value(merged, request.setting)
        return ConfigMutationResult(
            success=True,
            operation=operation,
            setting=request.setting,
            scope=scope,
            domain=domain,
            path=path,
            value=value,
            diagnostics=snapshot.diagnostics,
        )

    if (
        operation in {ConfigOperation.SET, ConfigOperation.APPEND_UNIQUE, ConfigOperation.REMOVE}
        and request.value is _MISSING
    ):
        raise ConfigurationError(f"{operation.value} requires a value")

    document = _read_json_strict(path)
    previous, new_value = _mutate_document(document, request.setting, operation, request.value)
    _validate_target(domain, document)
    try:
        _atomic_write_json(path, document)
    except OSError as exc:
        raise ConfigurationError(f"cannot atomically update {path}: {exc}") from exc
    invalidate_configuration("mutation", snapshot.workspace)
    refreshed = get_configuration_snapshot(snapshot.cwd)
    if tool_context is not None:
        apply_configuration_snapshot(tool_context, refreshed)
    return ConfigMutationResult(
        success=True,
        operation=operation,
        setting=request.setting,
        scope=scope,
        domain=domain,
        path=path,
        previous_value=previous,
        new_value=new_value,
        diagnostics=refreshed.diagnostics,
    )


def set_effort(value: str | None) -> ConfigMutationResult:
    """Persist reasoning effort in the canonical user settings file."""
    return mutate_configuration(
        ConfigMutationRequest(
            setting="effort",
            value=value or "",
            domain=ConfigDomain.SETTINGS,
            scope=ConfigScope.USER,
        )
    )


def _permission_rules(snapshot: ConfigurationSnapshot, trusted: bool) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for layer in snapshot.layers:
        merged = _deep_merge(layer.legacy_settings, layer.settings)
        if layer.scope is not ConfigScope.USER and not trusted:
            continue
        permissions = _normalise_permissions(merged.get("permissions"))
        result[f"{layer.scope.value}Settings"] = permissions
    return result


def _apply_permissions(tool_context: Any, snapshot: ConfigurationSnapshot) -> None:
    # pylint: disable=no-name-in-module  # permissions: pending patch migration
    from clawcodex_ext.permissions.loader import apply_rules_to_context, settings_to_rules
    from clawcodex_ext.permissions.types import PermissionRuleSource, ToolPermissionContext

    current = tool_context.permission_context
    allow = {k: list(v) for k, v in current.always_allow_rules.items() if k not in _CONFIG_SOURCES}
    deny = {k: list(v) for k, v in current.always_deny_rules.items() if k not in _CONFIG_SOURCES}
    ask = {k: list(v) for k, v in current.always_ask_rules.items() if k not in _CONFIG_SOURCES}
    base = ToolPermissionContext(
        mode=current.mode,
        additional_working_directories=dict(current.additional_working_directories),
        always_allow_rules=allow,
        always_deny_rules=deny,
        always_ask_rules=ask,
        is_bypass_permissions_mode_available=current.is_bypass_permissions_mode_available,
        should_avoid_permission_prompts=current.should_avoid_permission_prompts,
        await_automated_checks_before_dialog=current.await_automated_checks_before_dialog,
    )
    rules = []
    for source, permissions in _permission_rules(
        snapshot, bool(getattr(tool_context, "workspace_trusted", False))
    ).items():
        rules.extend(settings_to_rules(permissions, cast(PermissionRuleSource, source)))
    tool_context.permission_context = apply_rules_to_context(base, rules)


def _layer_hook_mapping(layer: ConfigurationLayer) -> dict[str, Any]:
    merged = _deep_merge(layer.legacy_settings, layer.settings)
    hooks = merged.get("hooks")
    return _thaw(hooks) if _is_hook_event_mapping(hooks) else {}


def _apply_hooks(tool_context: Any, snapshot: ConfigurationSnapshot) -> None:
    # pylint: disable=no-name-in-module  # hooks: pending patch migration
    from clawcodex_ext.hooks.config_manager import HookConfigSnapshot, load_hooks_from_mapping
    from clawcodex_ext.hooks.hook_types import HookSource

    hooks: dict[str, list[Any]] = {}
    trusted = bool(getattr(tool_context, "workspace_trusted", False))
    sources = {
        ConfigScope.USER: HookSource.USER_SETTINGS,
        ConfigScope.PROJECT: HookSource.PROJECT_SETTINGS,
        ConfigScope.LOCAL: HookSource.LOCAL_SETTINGS,
    }
    paths: list[str] = []
    for layer in snapshot.layers:
        if layer.scope is not ConfigScope.USER and not trusted:
            continue
        parsed = load_hooks_from_mapping(
            {"hooks": _layer_hook_mapping(layer)},
            source_path=layer.settings_path,
            source=sources[layer.scope],
        )
        paths.append(str(layer.settings_path))
        for event, configs in parsed.hooks.items():
            hooks.setdefault(event, []).extend(configs)
    manager = _FrozenHookManager(
        HookConfigSnapshot(
            hooks=hooks,
            source_path=os.pathsep.join(paths),
        )
    )
    tool_context.hook_config_manager = manager


@dataclass(slots=True)
class _FrozenHookManager:
    snapshot: Any


def _apply_env(tool_context: Any, snapshot: ConfigurationSnapshot) -> None:
    base = getattr(tool_context, "_configuration_env_base", None)
    if base is None:
        base = dict(getattr(tool_context, "env", {}) or {})
        tool_context._configuration_env_base = base
    env = dict(base)
    trusted = bool(getattr(tool_context, "workspace_trusted", False))
    for layer in snapshot.layers:
        if layer.scope is not ConfigScope.USER and not trusted:
            continue
        merged = _deep_merge(layer.legacy_settings, layer.settings)
        values = merged.get("env")
        if isinstance(values, Mapping):
            env.update({str(k): str(v) for k, v in values.items()})
    tool_context.env = env


def apply_configuration_snapshot(
    tool_context: Any,
    snapshot: ConfigurationSnapshot | None = None,
) -> ConfigurationSnapshot | None:
    if tool_context is None:
        return None
    active = snapshot or get_configuration_snapshot(
        getattr(tool_context, "cwd", None) or getattr(tool_context, "workspace_root", None)
    )
    _apply_permissions(tool_context, active)
    _apply_hooks(tool_context, active)
    _apply_env(tool_context, active)
    return active


__all__ = [
    "ConfigDomain",
    "ConfigMutationRequest",
    "ConfigMutationResult",
    "ConfigOperation",
    "ConfigScope",
    "ConfigurationError",
    "ConfigurationSnapshot",
    "apply_configuration_snapshot",
    "get_configuration_snapshot",
    "invalidate_configuration",
    "mutate_configuration",
    "set_effort",
]
