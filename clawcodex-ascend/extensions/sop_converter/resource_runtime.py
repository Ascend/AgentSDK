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

"""Generic resource materialize/invoke helpers for resume-resource."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

from .resource_catalog import (
    RESOURCE_PAYLOAD_REF_MISSING,
    ResourceCatalogError,
    ResourceRecord,
    resolve_payload,
)
from .resource_handlers import require_resource_handler


def _materialize_record(
    record: ResourceRecord,
    *,
    catalog_dir: Path | None,
) -> ResourceRecord:
    payload = record.payload if isinstance(record.payload, dict) else {}
    if payload.get("kind") != "payload_ref":
        return record
    if catalog_dir is None:
        raise ResourceCatalogError(
            RESOURCE_PAYLOAD_REF_MISSING,
            "catalog_dir is required to resolve payload_ref",
        )
    return resolve_payload(record, catalog_dir)


def materialize_resource(
    record: ResourceRecord,
    resource_type: str = "",
    *,
    catalog_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Materialize a catalog record through the registered ResourceHandler."""
    if not isinstance(record, ResourceRecord):
        raise TypeError("record must be a ResourceRecord")
    if catalog_dir is not None:
        catalog_dir = Path(catalog_dir)
    record = _materialize_record(record, catalog_dir=catalog_dir)
    handler = require_resource_handler(resource_type or record.resource_type)
    materialized = handler.materialize(record)
    if not isinstance(materialized, dict):
        raise TypeError(f"handler.materialize for {handler.resource_type!r} must return a dict")
    return materialized


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


def _invoke_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "output", "content", "result", "echo"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return ""


def invoke_resource(
    record: ResourceRecord,
    resource_type: str = "",
    query: str = "",
    inputs: Any = None,
) -> dict[str, Any]:
    """Invoke a catalog record via its ResourceHandler and normalize public output."""
    if not isinstance(record, ResourceRecord):
        raise TypeError("record must be a ResourceRecord")
    handler = require_resource_handler(resource_type or record.resource_type)
    raw_result = handler.invoke(record, query=query, inputs=inputs)
    if not isinstance(raw_result, dict):
        raw_result = {
            "output": raw_result,
            "raw": raw_result,
            "text": str(raw_result),
        }

    try:
        from clawcodex_ext.tool_system.schema_validation import validate_json_schema

        validate_json_schema(
            raw_result,
            handler.public_output_schema,
            root_name=f"resource.{handler.resource_type}.invoke",
        )
    except Exception as exc:
        from .composite_runtime import CompositeWorkflowError

        raise CompositeWorkflowError(
            "workflow_output_schema_mismatch",
            str(exc),
        ) from exc

    projected = {key: _json_safe_projection(value) for key, value in raw_result.items()}
    if "output" not in projected:
        projected["output"] = {
            key: value for key, value in projected.items() if key not in {"text", "raw", "method", "agent_id"}
        } or projected.get("text", "")
    if "raw" not in projected:
        projected["raw"] = projected["output"]
    if "text" not in projected:
        projected["text"] = _invoke_text(projected) or _invoke_text(raw_result)

    projected["resource_type"] = handler.resource_type
    projected["resource_ref"] = str(record.resource_id)
    projected["resource_id"] = str(record.resource_id)
    return projected


__all__ = ["invoke_resource", "materialize_resource"]
