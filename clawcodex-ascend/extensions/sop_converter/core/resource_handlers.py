#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.

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

"""Runtime extension registry for resource types."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .resource_catalog import ResourceRecord

RESOURCE_TYPE_UNREGISTERED = "resource_type_unregistered"


def normalize_resource_type(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


class ResourceHandlerError(RuntimeError):
    """Resource handler lookup failure with a stable machine-readable code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ResourceHandler:
    resource_type: str
    materialize: Callable[[ResourceRecord], dict[str, Any]]
    invoke: Callable[..., dict[str, Any]]
    public_output_schema: dict[str, Any]
    error_codes: frozenset[str]

    def __post_init__(self) -> None:
        normalized = normalize_resource_type(self.resource_type)
        if not normalized:
            raise ValueError("resource_type is required")
        if not callable(self.materialize) or not callable(self.invoke):
            raise TypeError("materialize and invoke must be callable")
        object.__setattr__(self, "resource_type", normalized)


_HANDLERS: dict[str, ResourceHandler] = {}
_REGISTRY_LOCK = threading.RLock()
_BUILTINS_READY = False


def register_resource_handler(
    handler: ResourceHandler,
    *,
    replace: bool = False,
) -> None:
    """Register one normalized resource type."""
    key = normalize_resource_type(handler.resource_type)
    with _REGISTRY_LOCK:
        if key in _HANDLERS and not replace:
            raise ValueError(f"resource handler already registered: {key}")
        _HANDLERS[key] = handler


def _invoke_agent_record(
    record: ResourceRecord,
    query: str = "",
    inputs: Any = None,
) -> dict[str, Any]:
    from .agent_runtime import invoke_agent, materialize_agent

    materialized = materialize_agent(record)
    return invoke_agent(
        materialized["agent"],
        record,
        query=query,
        inputs=inputs,
    )


def ensure_builtin_handlers() -> None:
    """Install Agent as the first registry row without import-time cycles."""
    global _BUILTINS_READY
    with _REGISTRY_LOCK:
        if _BUILTINS_READY:
            return
        from .agent_runtime import materialize_agent

        handler = ResourceHandler(
            resource_type="agent",
            materialize=materialize_agent,
            invoke=_invoke_agent_record,
            public_output_schema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                    "text": {"type": "string"},
                    "output": {},
                    "raw": {},
                    "method": {"type": "string"},
                },
                "required": ["agent_id", "text", "output"],
            },
            error_codes=frozenset(
                {
                    "resource_materialize_failed",
                    "resource_payload_invalid",
                    "resource_secret_missing",
                    "agent_invoke_failed",
                }
            ),
        )
        _HANDLERS.setdefault("agent", handler)
        _HANDLERS.setdefault("agentconfig", handler)
        _BUILTINS_READY = True


def get_resource_handler(resource_type: str) -> ResourceHandler | None:
    ensure_builtin_handlers()
    key = normalize_resource_type(resource_type)
    with _REGISTRY_LOCK:
        handler = _HANDLERS.get(key)
        if handler is not None:
            return handler
        # Existing catalogs use fully-qualified AgentConfig type identities.
        # This suffix alias is the compatibility mapping for the built-in row.
        if key.endswith(("agent", "agentconfig")):
            return _HANDLERS.get("agent")
        return None


def require_resource_handler(resource_type: str) -> ResourceHandler:
    handler = get_resource_handler(resource_type)
    if handler is None:
        normalized = normalize_resource_type(resource_type) or "<empty>"
        raise ResourceHandlerError(
            RESOURCE_TYPE_UNREGISTERED,
            f"{RESOURCE_TYPE_UNREGISTERED}: no handler registered for {normalized}",
        )
    return handler


def registered_resource_types() -> tuple[str, ...]:
    ensure_builtin_handlers()
    with _REGISTRY_LOCK:
        return tuple(sorted(_HANDLERS))


__all__ = [
    "RESOURCE_TYPE_UNREGISTERED",
    "ResourceHandler",
    "ResourceHandlerError",
    "ensure_builtin_handlers",
    "get_resource_handler",
    "normalize_resource_type",
    "register_resource_handler",
    "registered_resource_types",
    "require_resource_handler",
]
