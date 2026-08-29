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

"""Lifecycle-kind inference for SOP-converted tools.

Given a parsed :class:`SourceOperation`, decide whether it is
the *create* end of a lifecycle (e.g. ``build_agent`` / ``create_team_session``),
the *invoke* end (e.g. ``run_agent`` / ``invoke_existing_agent``), or neither.

The result drives two behaviours:

* **create** → the wrapper script's call_impl is augmented with a
  ``--catalog-metadata`` payload so the subprocess records the resulting
  ``agent_id`` into the :mod:`agent_catalog` for later retrieval.
* **invoke** → the wrapper script is left alone (callers will use the
  ``invoke-existing-agent`` macro tool, which itself is a separate composite
  tool that loads the catalog before importing the SDK class).  It also
  lets generated invoke-kind wrappers use the same catalog path as an automatic
  fallback when the SDK reports that the in-memory agent no longer exists.
* **none** → no special treatment; the wrapper script is the default
  ``python3 <script> <method> '{json_args}'`` template.

Heuristics (type-contract driven):

* ``create``  — name starts with ``build_`` / ``create_`` / ``init_`` /
  ``register_`` / ``ensure_``; AND return type contains ``Dict`` /
  ``Mapping`` (the SDK returns a config object the catalog can serialize).
* ``invoke``  — name starts with ``invoke_`` / ``run_`` / ``call_`` / ``send_``;
  AND one of:
    (a) at least one required parameter ends with ``_id`` / ``id`` (legacy
        name-based heuristic — kept as fallback for SDKs without type hints); or
    (b) at least one required parameter whose ``type_hint`` matches the
        ``return_type`` of a known create-kind op in the same component set
        (§8 type-contract path — works for any SDK whose create/invoke share
        a resource type, e.g. ``AgentConfig`` / ``TeamSession``).
* otherwise — ``"none"``.

The function is intentionally a pure predicate. Callers that need richer
metadata (e.g. the exact parameter name carrying the id, or the matched
``resource_type``) should layer that on top using :func:`_invoke_id_param`
and :func:`derive_resource_type`.
"""

from __future__ import annotations

import re
from typing import Literal

from ..source_parser import SourceOperation

LifecycleKind = Literal["create", "invoke", "none"]


_CREATE_PREFIXES = ("build_", "create_", "init_", "register_", "ensure_", "load_")
_INVOKE_PREFIXES = ("invoke_", "run_", "call_", "send_")

# Return-type substrings that signal a "config/dict" return. Matched
# case-insensitively against the unparsed ast type hint.
_DICT_RETURN_HINTS = (
    "dict",
    "mapping",
    "mutabledict",
    "agentconfig",
    "agentbuilder",
    "config",
)

_GENERIC_CONTAINER_RETURN_TOKENS = frozenset({"dict", "mapping", "mutabledict"})

# Parameter-name suffix patterns that signal the tool consumes a stable
# identifier (and is therefore the "invoke" end of a create→invoke chain).
# Kept as fallback for SDKs without type hints (§8.6 compatibility).
_ID_PARAM_RE = re.compile(r"^(?:[a-z]+_)?id$|^[a-z]+_id$", re.IGNORECASE)

# Built-in / primitive types that should never be treated as resource handles.
# Used by :func:`derive_resource_type` to skip non-resource return/param types.
_PRIMITIVE_TYPE_TOKENS = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "none",
        "any",
        "list",
        "dict",
        "tuple",
        "set",
        "mapping",
        "sequence",
        "optional",
        "union",
        "iterable",
        "iterator",
        "generator",
        "asynciterator",
        "asyncgenerator",
        "awaitable",
        "string",
        "integer",
        "boolean",
        "number",
        "object",
    }
)

_GENERIC_RESOURCE_TOKENS = frozenset(
    {
        "config",
        "settings",
        "options",
        "params",
        "kwargs",
        "args",
        "input",
        "inputs",
        "output",
        "outputs",
        "result",
        "results",
        "data",
        "payload",
        "request",
        "response",
        "context",
        "state",
    }
)


def _looks_like_dict_return(return_type: str | None) -> bool:
    if not return_type:
        return False
    lowered = return_type.lower()
    return any(token in lowered for token in _DICT_RETURN_HINTS)


def _normalize_type_hint(type_hint: str | None) -> str:
    """Normalize a type-hint string to a comparable resource-type token.

    Strips ``Optional[...]`` / ``Union[...]`` wrappers and generic parameters
    (``Dict[str, AgentConfig]`` → ``dict``; ``AgentConfig`` → ``agentconfig``).
    Returns the empty string for missing / unparseable hints.
    """
    if not type_hint:
        return ""
    text = type_hint.strip()
    # Peel Optional[...] / Union[...] / Annotated[...] wrappers.
    for wrapper in ("Optional", "Union", "Annotated"):
        prefix = wrapper + "["
        while text.startswith(prefix):
            inner = text[len(prefix) :]
            if inner.endswith("]"):
                text = inner[:-1].strip()
                # Union[A, B] / Annotated[T, ...] — comma-separated args.
                # Skip None so Optional/Union[..., None] yields the payload type.
                if wrapper in ("Union", "Annotated"):
                    arms = [
                        a.strip()
                        for a in _split_top_level(text, ",")
                        if a.strip() and a.strip() not in ("None", "NoneType")
                    ]
                    if arms:
                        text = arms[0]
            else:
                break
    # PEP 604 unions written as ``Foo | None``.
    if "|" in text:
        arms = [a.strip() for a in _split_top_level(text, "|") if a.strip() and a.strip() not in ("None", "NoneType")]
        if arms:
            text = arms[0]
    # Drop generic parameters: Foo[Bar, Baz] → Foo.
    bracket = text.find("[")
    if bracket > 0:
        text = text[:bracket].strip()
    # Last-name segment: module.AgentConfig → agentconfig.
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _candidate_type_tokens(type_hint: str | None) -> list[str]:
    """Return non-primitive type tokens embedded in a type hint."""
    if not type_hint:
        return []
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", type_hint):
        low = re.sub(r"[^a-z0-9]+", "", token.lower())
        if (
            low
            and low not in _PRIMITIVE_TYPE_TOKENS
            and low not in _GENERIC_CONTAINER_RETURN_TOKENS
            and low not in _GENERIC_RESOURCE_TOKENS
        ):
            tokens.append(low)
    return tokens


def _operation_complex_param_types(op: SourceOperation) -> list[str]:
    out: list[str] = []
    for param in op.parameters:
        if not param.required:
            continue
        pt = _normalize_type_hint(param.type_hint)
        if pt and pt not in _PRIMITIVE_TYPE_TOKENS and pt not in _GENERIC_RESOURCE_TOKENS:
            out.append(pt)
    return out


def _looks_like_resource_factory(op: SourceOperation) -> bool:
    """True when a create/build op appears to materialize a reusable resource."""
    name = op.name or ""
    if not any(name.startswith(p) for p in _CREATE_PREFIXES):
        return False
    if _looks_like_dict_return(op.return_type):
        return True
    rt = _normalize_type_hint(op.return_type)
    if rt and rt not in _PRIMITIVE_TYPE_TOKENS and rt not in _DICT_RETURN_HINTS and rt not in _GENERIC_RESOURCE_TOKENS:
        return True
    return bool(_operation_complex_param_types(op))


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split *text* on *sep* at bracket depth 0."""
    parts: list[str] = []
    depth = 0
    current = []
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth = max(0, depth - 1)
        if depth == 0 and ch == sep:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def derive_resource_type(op: SourceOperation) -> str:
    """Return the resource type this op produces (create-kind) or consumes (invoke-kind).

    For create-kind ops, derived from ``op.return_type``.
    For invoke-kind ops, derived from the first required parameter whose
    ``type_hint`` is non-primitive.
    Returns the empty string if no resource type can be derived.

    The result is a normalized lowercase token (e.g. ``"agentconfig"``)
    suitable for matching create↔invoke pairs across any SDK.
    """
    # Create-kind: take the return type, but skip bare dict/mapping (too generic).
    if any((op.name or "").startswith(p) for p in _CREATE_PREFIXES):
        rt = _normalize_type_hint(op.return_type)
        if rt and rt not in _PRIMITIVE_TYPE_TOKENS and rt not in _GENERIC_CONTAINER_RETURN_TOKENS:
            return rt
        # Fallback: dict-like return that still carries a resource name in
        # the unnormalized hint (e.g. ``Dict[str, AgentConfig]``).
        for token in _candidate_type_tokens(op.return_type):
            return token
        # §8.6: when return type is a bare dict, fall back to the first
        # non-primitive parameter type — this covers the common SDK pattern
        # ``create_agent(config: AgentConfig) -> Dict[str, Any]`` where the
        # resource type is carried by the input config, not the return value.
        for pt in _operation_complex_param_types(op):
            return pt
        return ""
    # Invoke-kind: first non-primitive required param type.
    for param in op.parameters:
        if not param.required:
            continue
        pt = _normalize_type_hint(param.type_hint)
        if pt and pt not in _PRIMITIVE_TYPE_TOKENS:
            return pt
    return ""


def invoke_lifecycle_id_param(op: SourceOperation) -> str | None:
    """Return the parameter name that carries the agent/team/session id, if any.

    Prefers a parameter whose ``type_hint`` matches a known resource type
    (§8 type-contract path). Falls back to the legacy ``*_id`` name rule
    (§3.2.3 compatibility) when no type hint matches.
    """
    resource_type = derive_resource_type(op)
    if resource_type:
        for param in op.parameters:
            if not param.required:
                continue
            if _normalize_type_hint(param.type_hint) == resource_type:
                return param.name
    # Legacy fallback: parameter name ends with _id / is exactly "id".
    for param in op.parameters:
        if not param.required:
            continue
        if _ID_PARAM_RE.match(param.name):
            return param.name
    return None


def invoke_lifecycle_query_arg(op: SourceOperation) -> str:
    """Return the most likely payload/query parameter for an invoke-kind op."""
    id_param = invoke_lifecycle_id_param(op)
    preferred = ("query", "prompt", "inputs", "input", "task", "message")
    param_names = [param.name for param in op.parameters if param.name != id_param]
    for name in preferred:
        if name in param_names:
            return name
    return param_names[0] if param_names else "query"


def inject_resource_ref_schema(
    input_schema: dict,
    *,
    resource_type: str,
    create_tool_name: str = "",
    consume_param: str | None = None,
) -> dict:
    """Add the stable resource handle contract to an invoke schema.

    The SDK-specific consume parameter remains in ``properties`` for legacy
    callers, but ``resource_ref`` replaces it in ``required`` when supplied.
    A fresh schema is returned so callers do not mutate a shared ToolSpec.
    """
    if not resource_type or not isinstance(input_schema, dict):
        return input_schema

    properties = dict(input_schema.get("properties") or {})
    if "resource_ref" not in properties:
        description = "Handle from the create tool that produces this resource_type"
        if create_tool_name:
            description = f"Handle from `{create_tool_name}` (resource_type={resource_type})"
        properties["resource_ref"] = {
            "type": "string",
            "description": description,
        }
    if "resource_type" not in properties:
        properties["resource_type"] = {
            "type": "string",
            "description": f"Resource type (default {resource_type})",
            "default": resource_type,
        }

    output = dict(input_schema)
    output["properties"] = properties
    required = list(output.get("required") or [])
    if consume_param:
        required = [name for name in required if name != consume_param]
    if "resource_ref" not in required:
        required.append("resource_ref")
    output["required"] = required
    return output


def infer_lifecycle_kind(
    op: SourceOperation,
    *,
    known_create_types: frozenset[str] | None = None,
) -> LifecycleKind:
    """Classify an op as ``"create"`` / ``"invoke"`` / ``"none"``.

    See module docstring for the heuristics.  Determinism matters: same
    input always yields the same classification so the catalog hook fires
    consistently across ``sop convert`` reruns.

    ``known_create_types`` (§8 type-contract path): when the caller has
    pre-computed the set of resource types produced by create-kind ops in
    the same component set, invoke-kind ops are also classified via type
    matching — an invoke op whose parameter type_hint matches any element
    of ``known_create_types`` is classified as ``"invoke"`` even if no
    parameter name ends with ``_id``.  When ``None``, only the legacy
    name-based heuristic applies.
    """
    name = op.name or ""
    if _looks_like_resource_factory(op):
        return "create"
    # §8 type-contract path: when create-types are known, type matching is a
    # strong enough signal that we skip the name-prefix check entirely.
    # This covers SDKs whose invoke method is named ``invoke`` (no trailing
    # underscore) or uses a non-standard verb.
    if known_create_types:
        for param in op.parameters:
            if not param.required:
                continue
            if _normalize_type_hint(param.type_hint) in known_create_types:
                return "invoke"
    if any(name.startswith(p) for p in _INVOKE_PREFIXES):
        # Legacy name-based fallback (§3.2.3).
        if invoke_lifecycle_id_param(op) is not None:
            return "invoke"
    return "none"


def lifecycle_fallback_payload(
    op: SourceOperation,
    *,
    source_dir: str,
    bundle_id: str | None = None,
    module_name: str | None = None,
    class_name: str | None = None,
    tool_name: str | None = None,
    known_create_types: frozenset[str] | None = None,
    lifecycle_kind_override: LifecycleKind | None = None,
    extra_metadata: dict | None = None,
) -> dict | None:
    """Build JSON metadata for invoke-kind catalog fallback.

    The generated wrapper receives this payload via ``--catalog-fallback`` and
    uses it only when the normal SDK call returns or raises an agent-not-found
    style failure.  The payload intentionally mirrors only the routing fields
    needed to call :func:`invoke_existing_agent`.
    """
    lifecycle_kind = lifecycle_kind_override or infer_lifecycle_kind(
        op,
        known_create_types=known_create_types,
    )
    if lifecycle_kind != "invoke":
        return None
    id_param = invoke_lifecycle_id_param(op) or "agent_id"
    resource_type = derive_resource_type(op)
    metadata: dict = {
        "sdk_source_dir": source_dir,
        "bundle_id": bundle_id or "",
        "module_name": module_name or "",
        "class_name": class_name or op.class_name or "",
        "source_tool": tool_name or "",
        # §8 type-contract fields (preferred by _try_catalog_fallback).
        "resource_type": resource_type,
        "handle_field": id_param,
        # Legacy id_arg kept for §8.6 backward compatibility.
        "id_arg": id_param,
        "query_arg": invoke_lifecycle_query_arg(op),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def lifecycle_metadata_payload(
    op: SourceOperation,
    *,
    source_dir: str,
    bundle_id: str | None = None,
    module_name: str | None = None,
    class_name: str | None = None,
    tool_name: str | None = None,
    known_create_types: frozenset[str] | None = None,
    lifecycle_kind_override: LifecycleKind | None = None,
    extra_metadata: dict | None = None,
) -> dict | None:
    """Build the JSON payload passed to the wrapper via ``--catalog-metadata``.

    Returns ``None`` for non-create ops (callers should not even attach the
    flag in that case).  The payload mirrors :class:`AgentCatalogEntry` minus
    ``agent_id`` and ``created_at``, which the wrapper fills in from the
    SDK's return value at call time.

    The wrapper is responsible for extracting the ``agent_id`` from the
    return value and for inserting it into the ``agent_id`` field of this
    payload before writing the catalog row.  This function therefore emits
    the *static* parts the wrapper cannot infer.
    """
    lifecycle_kind = lifecycle_kind_override or infer_lifecycle_kind(
        op,
        known_create_types=known_create_types,
    )
    if lifecycle_kind != "create":
        return None
    metadata: dict = {
        "sdk_source_dir": source_dir,
        "model": "",
        "provider": "",
        "class_name": class_name or op.class_name or "",
        "module_name": module_name or "",
        "query_arg": "query",
        "invoke_method": _guess_invoke_method(op),
        "schema_version": 1,
        # §8 type-contract fields: record so the create wrapper writes them
        # into the catalog entry, enabling invoke-kind fallback by resource_type.
        "resource_type": derive_resource_type(op),
        "handle_field": _guess_create_handle_field(op),
    }
    # Module-level factory functions return a runtime resource but have no
    # ``class_name`` in SourceOperation. Persist the factory entrypoint so
    # the catalog can recreate the resource with its original arguments.
    if not (class_name or op.class_name) and module_name and op.name:
        metadata["factory"] = {
            "module": module_name,
            "name": op.name,
        }
    if tool_name:
        metadata["source_tool"] = tool_name
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def _guess_invoke_method(op: SourceOperation) -> str:
    """Pick a sensible default ``invoke_method`` for the materialized Agent.

    The wrapper itself is the source of truth at call time — it tries
    ``invoke`` first, then ``run``.  This just supplies a hint for the
    catalog row's ``invoke_method`` field so consumers that don't try both
    have a reasonable starting point.
    """
    name = (op.name or "").lower()
    if name.startswith("run_"):
        return "run"
    return "invoke"


def _guess_create_handle_field(op: SourceOperation) -> str:
    """Return a compatibility default until the wrapper sees the real result."""
    resource_type = derive_resource_type(op)
    if resource_type.endswith(("agent", "agentconfig")):
        return "agent_id"
    return "id"


__all__ = [
    "LifecycleKind",
    "derive_resource_type",
    "infer_lifecycle_kind",
    "inject_resource_ref_schema",
    "invoke_lifecycle_id_param",
    "invoke_lifecycle_query_arg",
    "lifecycle_fallback_payload",
    "lifecycle_metadata_payload",
]
