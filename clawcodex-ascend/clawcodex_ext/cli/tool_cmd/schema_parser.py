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

# pylint: disable=try-except-raise
"""JSON Schema → argparse adapter.

 maps a tool's ``input_schema`` (JSON Schema) to a CLI argument
parser so the user can invoke ``/detect_modality --path /data/raw`` or
``clawcodex-dev tool detect_modality --path /data/raw`` instead of
crafting raw JSON.

Mapping rules (mirrors spec §1.5):

================================== ===========================================
JSON Schema field argparse argument
================================== ===========================================
``{"type": "string"}`` ``--name STR`` (or ``--name CHOICE`` when
 ``enum`` present, or ``--name JSON`` when
 ``format`` is ``"json"``)
``{"type": "integer"}``/``number`` ``--name INT`` / ``--name FLOAT``
``{"type": "boolean"}`` ``--name`` (store_true flag)
``{"type": "array", "items": …}`` ``--name ITEM [ITEM...]`` (nargs="+")
``{"type": "object"}`` ``--name JSON`` (raw JSON string)
``required: true`` argument is required (no default)
``required: false`` / absent argument is optional; ``default`` used if
 present in schema
``enum: [a, b, c]`` ``choices=[a, b, c]`` (string types only)
``description: "..."`` argument ``help`` text
================================== ===========================================

Edge cases
----------
* ``properties`` empty / schema ``{}`` → no arguments (tool is a no-op
 command, useful for status / info tools).
* ``additionalProperties: true`` → no schema constraint enforced at CLI
 level; we just expose the named properties.
* Missing ``type`` → defaults to ``"string"`` (most permissive).
* ``anyOf`` / ``oneOf`` / ``$ref`` → not supported in this slice; tools
 with such schemas fall back to a single ``--input JSON`` argument.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Mapping

log = logging.getLogger(__name__)


def _is_simple_schema(schema: Mapping[str, Any]) -> bool:
    """Return True if *schema* has only fields we know how to map.

    If False, we fall back to a single ``--input JSON`` argument so the
    tool remains callable from the CLI (just less convenient).
    """
    if not isinstance(schema, Mapping):
        return False
    for key in schema:
        if key not in {
            "type",
            "properties",
            "required",
            "enum",
            "description",
            "default",
            "items",
            "additionalProperties",
        }:
            return False
    return True


def _parse_json_or_str(raw: str) -> Any:
    """Parse a CLI value as JSON if it looks like a JSON literal,
    otherwise return the raw string. Used for ``object`` type fields.
    """
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if not stripped:
        return raw
    # Heuristic: a value that starts with ``"`` or is a valid JSON literal
    # should be parsed. ``[`` / ``{`` are unambiguous JSON containers.
    if stripped[0] in '"[{0123456789tfn-':
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    return raw


def build_arg_parser(tool_name: str, schema: Mapping[str, Any] | None) -> argparse.ArgumentParser:
    """Build an ``argparse.ArgumentParser`` for *tool_name* from *schema*.

    The returned parser accepts ``--key value`` (or ``--key`` for bool
    flags) plus a trailing ``[args ...]`` catch-all that the caller can
    ignore or merge into a top-level ``*args`` field if the schema has
    one.
    """
    parser = argparse.ArgumentParser(
        prog=f"/{tool_name}",
        description=f"Auto-generated CLI for tool '{tool_name}'.",
        add_help=True,
        exit_on_error=False,  # raise ``ArgumentError`` instead of sys.exit
    )

    if not schema or not isinstance(schema, Mapping):
        return parser

    if not _is_simple_schema(schema):
        # Fallback: single --input JSON argument for complex schemas.
        parser.add_argument(
            "--input",
            type=str,
            default=None,
            help="Tool input as a JSON object string (e.g. '{\"key\": \"value\"}').",
        )
        return parser

    properties = schema.get("properties")
    additional = schema.get("additionalProperties", False)

    if not isinstance(properties, Mapping) or not properties:
        # Schema is non-empty but has no ``properties`` (e.g. complex
        # ``anyOf``/``oneOf``/``$ref``). The fallback --input JSON arg
        # was already added above; nothing more to do.
        if additional:
            parser.description = parser.description + " (Schema allows additional properties.)"
        return parser

    required = set(schema.get("required") or ())

    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, Mapping):
            prop_schema = {}
        arg_name = f"--{prop_name.replace('_', '-')}"
        is_required = prop_name in required
        kwargs: dict[str, Any] = {
            "dest": prop_name,
            "default": argparse.SUPPRESS if is_required else prop_schema.get("default"),
            "required": False,  # we handle required manually for nicer errors
        }
        desc = prop_schema.get("description")
        if isinstance(desc, str) and desc:
            kwargs["help"] = desc

        prop_type = prop_schema.get("type")
        enum = prop_schema.get("enum")

        if prop_type == "boolean":
            kwargs["action"] = "store_true"
            kwargs.pop("default", None)
            kwargs["default"] = prop_schema.get("default", False)
            parser.add_argument(arg_name, **kwargs)
            continue

        if prop_type == "integer":
            kwargs["type"] = int
        elif prop_type in ("number", "float"):
            kwargs["type"] = float
        elif prop_type == "array":
            item_type = (
                (prop_schema.get("items") or {}).get("type") if isinstance(prop_schema.get("items"), Mapping) else None
            )
            if item_type in ("integer", "number", "float"):
                kwargs["type"] = int if item_type == "integer" else float
            else:
                kwargs["type"] = str
            kwargs["nargs"] = "+"
        elif prop_type == "object":
            kwargs["type"] = str
            kwargs["help"] = (kwargs.get("help", "") + " (parsed as JSON)").strip()
        else:
            # Default: string (covers "string" and unknown/missing types).
            kwargs["type"] = str
            if isinstance(enum, list) and enum and all(isinstance(e, str) for e in enum):
                kwargs["choices"] = enum
                if not desc:
                    kwargs["help"] = f"one of {', '.join(enum)}"

        if is_required:
            # Required → no default; argparse will error if missing.
            kwargs.pop("default", None)
        parser.add_argument(arg_name, **kwargs)

    if additional:
        # Schema allows extra properties; expose them as --key value pairs
        # by appending a parser for the *known* extras. We do this by
        # parsing the remaining argv after known args via parse_known_args
        # in the caller. The flag is exposed via a help note.
        parser.description = parser.description + " (Schema allows additional properties.)"

    return parser


def parse_tool_args(
    tool_name: str,
    schema: Mapping[str, Any] | None,
    argv: list[str],
) -> dict[str, Any]:
    """Parse *argv* against *schema* and return a tool input dict.

    Returns the parsed dict (always with str keys). On parse failure,
    raises :class:`argparse.ArgumentError` so the caller can format a
    friendly usage message.
    """
    parser = build_arg_parser(tool_name, schema)

    if not schema or not isinstance(schema, Mapping):
        if argv:
            log.debug("tool %r received extra args but has no schema: %r", tool_name, argv)
        return {}

    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        # Complex / fallback schema (anyOf, no properties). The parser
        # carries a single --input JSON arg; parse it.
        if argv:
            try:
                ns, unknown = parser.parse_known_args(argv)
            except argparse.ArgumentError:
                raise
            result: dict[str, Any] = {}
            if "input" in ns and getattr(ns, "input", None) is not None:
                # Strip wrapping quotes if the user passed a quoted JSON
                # string. We accept both `"hello"` and `hello`; the
                # latter is also valid JSON.
                result["input"] = _parse_json_or_str(getattr(ns, "input"))
            if unknown:
                result["_extra"] = list(unknown)
            return result
        return {}

    required = set(schema.get("required") or ())

    try:
        # Use parse_known_args so extra args (rare) don't error out.
        ns, unknown = parser.parse_known_args(argv)
    except argparse.ArgumentError:
        raise

    result = {}
    for prop_name in properties:
        if prop_name in ns:
            value = getattr(ns, prop_name)
            prop_schema = properties[prop_name] or {}
            if isinstance(prop_schema, Mapping) and prop_schema.get("type") == "object" and isinstance(value, str):
                value = _parse_json_or_str(value)
            if value is not argparse.SUPPRESS:
                result[prop_name] = value

    # Validate required fields are present. ``parse_known_args`` returns
    # a Namespace where unset attributes are ``None`` (since we configured
    # them with ``required=False`` to drive our own validation), so we
    # must check for ``None`` explicitly — not just key absence.
    for req in required:
        if req not in result or result[req] is None:
            raise argparse.ArgumentError(
                None,
                f"the following arguments are required: --{req.replace('_', '-')}",
            )

    if unknown:
        # Forward unknown args as the ``_extra`` list so the caller can
        # decide whether to fold them into an "additional properties" bucket.
        result["_extra"] = list(unknown)

    return result
