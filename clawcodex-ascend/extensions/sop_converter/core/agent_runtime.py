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

"""Runtime helpers for materializing and invoking agent records."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_type_hints

from .resource_catalog import (
    RESOURCE_PAYLOAD_REF_MISSING,
    RESOURCE_SECRET_MISSING,
    ResourceCatalogError,
    ResourceRecord,
    resolve_payload,
)
from .sdk_serialization import coerce_sdk_type


class AgentRuntimeError(RuntimeError):
    """Agent materialization or invocation failure with a stable code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def _materializer_init_kwargs(record: ResourceRecord) -> dict[str, Any]:
    payload = record.payload if isinstance(record.payload, dict) else {}
    materializer = record.materializer if isinstance(record.materializer, dict) else {}
    return dict(payload.get("init_kwargs") or materializer.get("init_kwargs") or {})


def _collect_env_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        if value.startswith("env:") and value[4:].strip():
            refs.add(value[4:].strip())
        if value.startswith("<redacted:env:") and value.endswith(">"):
            refs.add(value[len("<redacted:env:") : -1])
        return refs
    if isinstance(value, Mapping):
        for child in value.values():
            refs.update(_collect_env_refs(child))
        return refs
    if isinstance(value, (list, tuple, set)):
        for child in value:
            refs.update(_collect_env_refs(child))
    return refs


def _validate_record_secrets(record: ResourceRecord) -> None:
    refs = {str(item) for item in (record.secrets or {}).get("env_refs", []) if item}
    refs.update(_collect_env_refs(record.payload))
    refs.update(_collect_env_refs(record.materializer))
    missing = sorted(name for name in refs if os.environ.get(name) is None)
    if missing:
        raise AgentRuntimeError(
            RESOURCE_SECRET_MISSING,
            "required environment variables are not set: " + ", ".join(missing),
        )


def _non_optional_sequence_origin(annotation: Any) -> Any | None:
    """Return list/tuple/set/frozenset origin when *annotation* is non-optional."""
    from types import UnionType
    from typing import Union, get_origin

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        # Optional[List[T]] keeps None (create wrapper does the same).
        return None
    if origin in (list, tuple, set, frozenset):
        return origin
    if annotation in (list, tuple, set, frozenset):
        return annotation
    return None


def _empty_sequence_for_origin(origin: Any) -> Any:
    if origin is list:
        return []
    if origin is tuple:
        return ()
    if origin is set:
        return set()
    if origin is frozenset:
        return frozenset()
    return []


def _coerce_factory_kwargs(factory: Any, init_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply the generated-wrapper coercion contract before a direct factory call.

    Init kwargs store JSON-compatible input arguments. Rematerialization must
    reconstruct the SDK's Pydantic/dataclass objects before calling the SDK
    factory directly;
    otherwise SDK factories receive a plain ``dict`` instead of a typed config.

    Catalog ``init_kwargs`` only persist args the create call received. Create
    wrappers still coerce omitted ``List[...] = None`` parameters with
    ``(param or [])`` before invoking the SDK. Rematerialize must apply the
    same contract so helpers like ``len(workflows)`` never see ``None``.
    """
    try:
        signature = inspect.signature(factory)
        hints = get_type_hints(factory)
    except (TypeError, ValueError, NameError):
        return dict(init_kwargs)

    coerced = dict(init_kwargs)
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name) if hints else None
        if annotation is None and parameter.annotation is not inspect.Parameter.empty:
            annotation = parameter.annotation

        if name not in coerced:
            # Only fill create-wrapper list defaults: omitted + default None +
            # non-optional sequence annotation → []. Leave other defaults to
            # Python (e.g. Optional[str] = None stays omitted).
            if parameter.default is inspect.Parameter.empty or parameter.default is not None:
                continue
            sequence_origin = _non_optional_sequence_origin(annotation)
            if sequence_origin is None:
                continue
            coerced[name] = _empty_sequence_for_origin(sequence_origin)

        if annotation is None or annotation is inspect.Parameter.empty:
            continue
        value = coerced[name]
        sequence_origin = _non_optional_sequence_origin(annotation)
        if value is None and sequence_origin is not None:
            value = _empty_sequence_for_origin(sequence_origin)
        coerced[name] = coerce_sdk_type(annotation, value)
    return coerced


def materialize_agent(
    record: ResourceRecord,
    *,
    catalog_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Materialize an agent using the factory or class stored in a record."""
    if not isinstance(record, ResourceRecord):
        raise AgentRuntimeError("resource_payload_invalid", "agent record is invalid")
    if catalog_dir is not None:
        catalog_dir = Path(catalog_dir)
    payload = record.payload if isinstance(record.payload, dict) else {}
    if payload.get("kind") == "payload_ref":
        if catalog_dir is None:
            raise ResourceCatalogError(
                RESOURCE_PAYLOAD_REF_MISSING,
                "catalog_dir is required to resolve payload_ref",
            )
        record = resolve_payload(record, catalog_dir)
    _validate_record_secrets(record)
    sdk_dir = str((record.sdk or {}).get("source_dir") or "")
    if sdk_dir and sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)

    materializer = record.materializer if isinstance(record.materializer, dict) else {}
    module_name = str(materializer.get("module") or "")
    kind = str(materializer.get("kind") or "python_class")
    init_kwargs = _materializer_init_kwargs(record)
    if not module_name:
        raise AgentRuntimeError(
            "resource_materialize_failed",
            f"agent record {record.resource_id!r} has no materializer module",
        )
    try:
        module = importlib.import_module(module_name)
        if kind == "python_function":
            factory_name = str(materializer.get("name") or "")
            if not factory_name:
                raise AttributeError("factory name is missing")
            factory = getattr(module, factory_name)
            agent = factory(**_coerce_factory_kwargs(factory, init_kwargs))
        else:
            class_name = str(materializer.get("class_name") or "")
            if not class_name:
                raise AttributeError("class name is missing")
            agent_class = getattr(module, class_name)
            agent = agent_class(**_coerce_factory_kwargs(agent_class, init_kwargs))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise AgentRuntimeError("resource_materialize_failed", str(exc)) from exc
    return {"agent": agent}


def _invoke_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("output", "text", "content", "result", "echo"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    payload = getattr(value, "payload", None)
    if isinstance(payload, Mapping):
        for key in ("output", "text", "content", "result", "echo"):
            candidate = payload.get(key)
            if isinstance(candidate, str):
                return candidate
    return ""


def _json_safe_projection(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_projection(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_projection(child) for child in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe_projection(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe_projection(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe_projection(to_dict())
    return str(value)


def invoke_agent(
    agent: Any,
    record: ResourceRecord,
    query: str = "",
    inputs: Any = None,
) -> dict[str, Any]:
    """Invoke a materialized agent according to its persisted invoker contract."""
    invoker = record.invoker if isinstance(record.invoker, dict) else {}
    method_name = str(invoker.get("method") or "invoke")
    input_param = str(invoker.get("input_param") or "query")
    payload = inputs if inputs is not None else query
    if input_param == "inputs" and not isinstance(payload, Mapping):
        payload = {"query": payload}
    try:
        method = getattr(agent, method_name, None)
        if method is None:
            for alternate in ("invoke", "run", "__call__"):
                method = getattr(agent, alternate, None)
                if method is not None:
                    method_name = alternate
                    break
        if method is None:
            raise AttributeError("agent exposes none of invoke/run/__call__")
        result = method(**{input_param: payload})
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
    except Exception as exc:
        raise AgentRuntimeError("agent_invoke_failed", str(exc)) from exc
    raw = _json_safe_projection(result)
    return {
        "agent_id": record.resource_id,
        "text": _invoke_text(result),
        "raw": raw,
        "output": raw,
        "method": method_name,
    }


__all__ = ["AgentRuntimeError", "invoke_agent", "materialize_agent"]
