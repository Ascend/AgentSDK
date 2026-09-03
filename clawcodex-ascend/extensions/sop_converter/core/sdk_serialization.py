#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""JSON-serializable encoding for SDK wrapper return values."""

from __future__ import annotations

import dataclasses
import inspect
import json
import textwrap
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

# ---------------------------------------------------------------------------
# Canonical implementations.
#
# The generated SDK wrapper scripts embed these functions as text (see
# ``_render_wrapper_helpers`` at the bottom of this module) instead of
# maintaining a separate inlined copy, so a bug fix here is automatically
# reflected in every generated script.
# ---------------------------------------------------------------------------


def normalize_mapping_inputs(value: Any, *, message_key: str = "query") -> dict[str, Any]:
    """Coerce tool args into a mapping when an ``inputs`` param expects a dict.

    Used by generated SDK wrapper scripts (see ``WRAPPER_SERIALIZATION_HELPERS``)
    and unit tests.  Generic rule: bare strings become ``{message_key: value}``.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {message_key: value}
    raise TypeError(f'inputs must be a dict, e.g. {{"{message_key}": "..."}}; got {type(value).__name__}: {value!r}')


def coerce_mapping_value(value: Any) -> dict[str, Any] | None:
    """Coerce tool args into a dict for mapping-typed SDK parameters."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TypeError(f"expected a JSON object string for mapping parameter; invalid JSON: {exc}") from exc
            if isinstance(parsed, dict):
                return parsed
            raise TypeError(f"expected a JSON object for mapping parameter; got {type(parsed).__name__}")
        raise TypeError(f"expected a dict or JSON object string for mapping parameter; got str: {value!r}")
    raise TypeError(
        f"expected a dict or JSON object string for mapping parameter; got {type(value).__name__}: {value!r}"
    )


def _parse_json_config(value: Any) -> Any:
    """Parse an inline JSON object while preserving already-coerced values."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            return json.loads(text)
    return value


def resolve_env_references(value: Any, *, environ: dict[str, str] | None = None) -> Any:
    """Resolve explicit ``env:NAME`` values recursively.

    Only the ``env:`` syntax is a supported reference. Shell-style forms such
    as ``$NAME`` remain ordinary strings so secrets are never implicitly
    expanded or printed by an agent-facing tool response.
    """
    import os
    import re

    environment = os.environ if environ is None else environ
    if isinstance(value, str) and value.startswith("env:"):
        name = value[4:].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment-variable reference: {value!r}")
        resolved = environment.get(name)
        if resolved is None:
            raise ValueError(f"environment variable is not set: {name}")
        _RESOLVED_ENV_REFERENCES[str(resolved)] = f"env:{name}"
        return resolved
    if isinstance(value, dict):
        resolved_dict: dict[Any, Any] = {}
        for key, item in value.items():
            if (
                str(key).lower() in {"api_key", "apikey", "access_token"}
                and isinstance(item, str)
                and item.startswith("$")
            ):
                raise ValueError("shell-style secret references are unsupported; use env:NAME")
            resolved_dict[key] = resolve_env_references(item, environ=environment)
        return resolved_dict
    if isinstance(value, list):
        return [resolve_env_references(item, environ=environment) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_env_references(item, environ=environment) for item in value)
    return value


def coerce_sdk_type(cls: Any, value: Any) -> Any:
    """Convert JSON-compatible *value* to an SDK annotation type.

    This is the runtime counterpart of the helper embedded into generated SDK
    wrappers. It is invoked right before a saved factory is called directly.
    """
    value = resolve_env_references(value)
    if value is None or cls in (Any, object):
        return value
    try:
        if isinstance(value, cls):
            return value
    except TypeError:
        pass

    origin = get_origin(cls)
    args = get_args(cls)
    if origin in (Union, UnionType):
        candidates = [item for item in args if item is not NoneType and item is not None]
        if len(candidates) == 1:
            return coerce_sdk_type(candidates[0], value)
        for candidate in candidates:
            try:
                return coerce_sdk_type(candidate, value)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"Cannot coerce {value!r} to {cls}")
    if origin in (list, tuple, set, frozenset):
        inner = args[0] if args else Any
        if not hasattr(value, "__iter__"):
            raise TypeError(f"Cannot coerce {value!r} to {cls}: expected a sequence")
        coerced = [coerce_sdk_type(inner, item) for item in value]
        if origin is tuple:
            return tuple(coerced)
        if origin is set:
            return set(coerced)
        if origin is frozenset:
            return frozenset(coerced)
        return coerced
    if origin is dict or cls is dict:
        return value
    if not isinstance(value, dict):
        if dataclasses.is_dataclass(cls) and isinstance(cls, type) and isinstance(value, str):
            field_names = {field.name for field in dataclasses.fields(cls)}
            if "model_provider" in field_names and "model_info" in field_names:
                return coerce_sdk_type(
                    cls,
                    {
                        "model_provider": "",
                        "model_info": {"model": value},
                    },
                )
        return cls(value)
    if dataclasses.is_dataclass(cls) and isinstance(cls, type):
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}
        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            if field.name not in value:
                continue
            annotation = hints.get(field.name, field.type)
            kwargs[field.name] = (
                value[field.name] if annotation in (Any, None) else coerce_sdk_type(annotation, value[field.name])
            )
        return cls(**kwargs)
    if hasattr(cls, "model_validate"):
        return cls.model_validate(value)
    return cls(**value)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert SDK objects to JSON-compatible data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)

    return str(obj)


# Module-level mirror of the wrapper-script redaction helper so the public
# ``dumps_sdk_result`` behaves like the inlined ``_dumps_sdk_result``.
_RESOLVED_ENV_REFERENCES: dict[str, str] = {}


def _redact_sensitive_fields(value: Any) -> Any:
    """Keep factory output and catalog DSL safe for agent-facing transport."""
    sensitive_tokens = ("api_key", "apikey", "access_token", "secret", "password")
    if isinstance(value, dict):
        return {
            str(key): (
                _RESOLVED_ENV_REFERENCES.get(str(item), "<redacted>")
                if any(token in str(key).lower() for token in sensitive_tokens)
                else _redact_sensitive_fields(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive_fields(item) for item in value]
    return value


def _serialize_factory_result(instance: Any) -> Any:
    """Serialize factory function return value for JSON output.

    Attempts to extract meaningful configuration from complex objects
    that don't have model_dump or dataclass serialization.
    """
    result = to_jsonable(instance)
    if isinstance(result, str):
        info: dict[str, Any] = {}
        for attr_name in [
            "agent_config",
            "id",
            "name",
            "description",
            "model",
            "version",
            "controller_type",
            "config",
        ]:
            if hasattr(instance, attr_name):
                attr_val = getattr(instance, attr_name)
                if attr_val is not None:
                    # Best-effort extraction must not fail: un-serializable
                    # attributes are skipped, never raised to the caller.
                    try:
                        info[attr_name] = _redact_sensitive_fields(to_jsonable(attr_val))
                    except Exception:  # noqa: BLE001, S110  # nosec B110
                        pass
        if info:
            info["_runtime_type"] = {
                "module": type(instance).__module__,
                "class_name": type(instance).__name__,
            }
            try:
                import inspect  # pylint: disable=redefined-outer-name,reimported

                invoke = getattr(instance, "invoke", None)
                if callable(invoke):
                    params = list(inspect.signature(invoke).parameters)
                    if params:
                        info["_runtime_invoker"] = {
                            "method": "invoke",
                            "input_param": params[0],
                        }
            except (TypeError, ValueError):
                pass
            info["_repr"] = result
            return info
    return result


def dumps_sdk_result(result: Any) -> str:
    return json.dumps(_redact_sensitive_fields(to_jsonable(result)), ensure_ascii=False, default=str)


WRAPPER_TEAM_DATABASE_COERCION = '''
def _coerce_team_database(value):
    """Coerce an inline mapping or JSON string into a TeamDatabase."""
    import asyncio
    from openjiuwen.agent_teams.tools.database import TeamDatabase
    from openjiuwen.agent_teams.tools.database.config import DatabaseConfig

    if isinstance(value, TeamDatabase):
        return value

    cfg = _parse_json_config(value)
    if isinstance(cfg, dict):
        if "config" in cfg:
            cfg = cfg["config"]
        if not cfg.get("connection_string"):
            cfg = {**cfg, "connection_string": ":memory:"}
        db = TeamDatabase(DatabaseConfig.model_validate(cfg))
        try:
            asyncio.run(db.initialize())
        except RuntimeError as exc:
            if "Event loop is running" in str(exc):
                loop = asyncio.get_running_loop()
                loop.run_until_complete(db.initialize())
            else:
                raise
        return db

    raise TypeError(f"Cannot coerce db from {type(value).__name__}")
'''.lstrip()


WRAPPER_MESSAGER_COERCION = '''
def _coerce_messager(value, *, team_name=None):
    """Coerce an inline mapping or JSON string into a Messager."""
    from openjiuwen.agent_teams.messager.base import (
        MessagerTransportConfig,
        create_messager,
    )
    from openjiuwen.agent_teams.messager.messager import Messager

    if isinstance(value, Messager):
        return value

    cfg = _parse_json_config(value)
    if isinstance(cfg, dict):
        if "backend" not in cfg:
            cfg = {**cfg, "backend": "inprocess"}
        if team_name and not cfg.get("team_name"):
            cfg = {**cfg, "team_name": team_name}
        return create_messager(MessagerTransportConfig.model_validate(cfg))

    raise TypeError(f"Cannot coerce messager from {type(value).__name__}")
'''.lstrip()


# ---------------------------------------------------------------------------
# Wrapper-script helper source generation
# ---------------------------------------------------------------------------

# Imports required by the embedded helpers.  The wrapper template already
# imports ``json`` / ``dataclasses`` (harmless duplicates); the ``typing`` /
# ``types`` imports are needed by the embedded type-coercion helpers.
_HELPER_IMPORTS = (
    "import dataclasses\n"
    "import json\n"
    "from typing import Any, Union, get_args, get_origin, get_type_hints\n"
    "from types import NoneType, UnionType\n"
)

#: ``(inline_name, canonical_func)`` pairs, in definition order.  The inline
#: names are what generated wrapper scripts call (underscore-prefixed);
#: functions whose inline name differs from the canonical one are rendered
#: under their canonical name plus an alias line.
_WRAPPER_SERIALIZATION_HELPERS_SPEC: list[tuple[str, object]] = [
    ("_redact_sensitive_fields", _redact_sensitive_fields),
    ("_to_jsonable", to_jsonable),
    ("_serialize_factory_result", _serialize_factory_result),
    ("_dumps_sdk_result", dumps_sdk_result),
    ("_normalize_mapping_inputs", normalize_mapping_inputs),
]

_WRAPPER_COERCION_HELPERS_SPEC: list[tuple[str, object]] = [
    ("_resolve_env_references", resolve_env_references),
    ("_parse_json_config", _parse_json_config),
    ("_coerce_mapping_value", coerce_mapping_value),
    ("_coerce_sdk_type", coerce_sdk_type),
]


def _render_wrapper_helpers(spec: list[tuple[str, object]]) -> str:
    """Render canonical module functions as self-contained source.

    Generated wrapper scripts only have the wrapped SDK on ``sys.path``, so the
    helpers are embedded as text.  Rendering from the module-level
    implementations keeps generated scripts in sync with bug fixes here.

    Functions whose inline name differs from the canonical one are rendered
    under their canonical name plus an alias line (e.g. ``_to_jsonable =
    to_jsonable``) so the generated script can keep calling the underscore
    names it always used.
    """
    parts: list[str] = [_HELPER_IMPORTS]
    for inline_name, func in spec:
        source = textwrap.dedent(inspect.getsource(func))
        if inline_name == func.__name__:
            parts.append(source)
        else:
            parts.append(f"{source}\n\n{inline_name} = {func.__name__}")
    return "\n\n".join(parts).lstrip()


# Inlined into generated wrapper scripts (scripts only have SDK on sys.path).
WRAPPER_SERIALIZATION_HELPERS = "_RESOLVED_ENV_REFERENCES = {}\n\n" + _render_wrapper_helpers(
    _WRAPPER_SERIALIZATION_HELPERS_SPEC
)

# Runtime type coercion for JSON args -> SDK Pydantic/dataclass instances.
WRAPPER_COERCION_HELPERS = _render_wrapper_helpers(_WRAPPER_COERCION_HELPERS_SPEC)
