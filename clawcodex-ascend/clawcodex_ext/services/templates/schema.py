#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Template schema — the canonical contract for template files.

This module is the single source of truth for what a template payload
looks like on disk. The :class:`Template` dataclass in
:mod:`src.services.templates.models` enforces per-field validation, but
the *shape* of the document (which keys are allowed at which level,
how a file can bundle multiple templates, how strict the parser is
about unknown keys) lives here.

The contract has three pieces:

1. **Top-level shape** — a template file is EITHER a single mapping
   (one template) OR a list of mappings (a bundle of N templates). Any
   other top-level shape is a :class:`TemplateCorruptError`.
2. **Top-level keys** — each template mapping is allowed to declare
   exactly: ``id``, ``title``, ``description``, ``fields``, ``metadata``,
   ``source``. ``id`` and ``title`` are required; everything else is
   optional. In strict mode (the default) any other key is rejected;
   in lenient mode unknown keys are silently dropped.
3. **Nested keys** — the ``fields`` and ``metadata`` mappings must
   have string keys; ``fields`` keys must additionally match the
   Python-identifier pattern enforced by :class:`Template`.

The :data:`SCHEMA_DESCRIPTION` constant below embeds the contract as
a single string so a future ``/template schema`` CLI command (P85-D)
can ``print(SCHEMA_DESCRIPTION)`` it directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .exceptions import TemplateCorruptError, TemplateValidationError
from .models import Template

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

TEMPLATE_SCHEMA_VERSION: str = "1.0"
"""Bumped on any breaking change to the template file format."""

TEMPLATE_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "title",
        "description",
        "fields",
        "metadata",
        "source",
        # convenience keys. They are normalized into metadata/fields
        # before constructing Template so the storage model remains stable.
        "kind",
        "variables",
        "tags",
        "category",
        "schema_version",
        "min_clawcodex_version",
        "output_path_template",
        "content_template",
        "content_template_ref",
    }
)
"""The only keys allowed at the top of a template mapping."""

TEMPLATE_FIELD_KEYS: frozenset[str] = TEMPLATE_TOP_LEVEL_KEYS
"""The keys allowed inside a ``fields`` or ``metadata`` mapping.

These are the same set as the top-level keys for symmetry — both
``fields`` and ``metadata`` are free-form Mappings, so any string
key is allowed. The dataclass enforces Python-identifier shape for
``fields`` keys; ``metadata`` keys only need to be non-empty strings.
"""

SCHEMA_DESCRIPTION: str = (
    "# ClawCodex Template Schema — version "
    f"{TEMPLATE_SCHEMA_VERSION}\n"
    "\n"
    "A template file is EITHER a single template mapping OR a list of\n"
    "template mappings. Each mapping has the following shape:\n"
    "\n"
    "  id:          string, required, pattern [A-Za-z0-9._-]{1,64}\n"
    "  title:       string, required, max 200 chars\n"
    "  description: string, optional, max 2000 chars\n"
    "  fields:      object, optional, keys must be valid Python\n"
    "               identifiers (default: {})\n"
    "  metadata:    object, optional, keys must be non-empty strings\n"
    "               (default: {})\n"
    "  source:      string, optional, one of built-in / user / project /\n"
    "               managed / plugin (default: user)\n"
    "\n"
    "An agent definition can reference a template via:\n"
    "\n"
    "  template: <id>     # at the top level of the agent dict\n"
    "\n"
    "The reference is resolved at load time by TemplateResolver; a\n"
    "missing template id raises TemplateNotFoundError.\n"
    "\n"
    "Example (single template):\n"
    "\n"
    "  id: general-purpose\n"
    "  title: General Purpose Agent\n"
    "  description: A versatile agent for general tasks\n"
    "  fields:\n"
    "    tools: [Read, Write, Edit, Bash]\n"
    "    max_turns: 20\n"
    "  metadata:\n"
    "    version: 1.0\n"
    "  source: built-in\n"
    "\n"
    "Example (bundle — list of templates in one file):\n"
    "\n"
    "  - id: explore\n"
    "    title: Explore Agent\n"
    "    fields: { tools: [Read, Grep, Glob] }\n"
    "  - id: plan\n"
    "    title: Plan Agent\n"
    "    fields: { tools: [Read, Grep, Glob] }\n"
)


# ---------------------------------------------------------------------------
# Payload parsers
# ---------------------------------------------------------------------------


def _check_strict_keys(
    data: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    strict: bool,
    what: str,
) -> None:
    """Reject unknown top-level keys when ``strict`` is True."""
    if not strict:
        return
    unknown = set(data.keys()) - allowed
    if unknown:
        raise TemplateValidationError(f"{what} has unknown key(s): {sorted(unknown)}; allowed: {sorted(allowed)}")


def parse_template_payload(
    data: Any,
    *,
    strict: bool = True,
) -> Template:
    """Parse a single template mapping into a :class:`Template`.

    Args:
        data: A mapping matching the top-level shape of the schema.
        strict: When True (default), reject unknown keys with
            :class:`TemplateValidationError`. When False, unknown
            keys are silently dropped (forward-compat with future
            schema additions).

    Raises:
        TemplateValidationError: On shape violations (non-dict input,
            unknown keys in strict mode, etc.). Note that
            :class:`Template`'s own ``__post_init__`` raises plain
            ``ValueError`` for id / title / field-name pattern
            violations; both error types are caught by callers that
            want a uniform "validation" exception.
    """
    if not isinstance(data, Mapping):
        raise TemplateValidationError(f"template payload must be a mapping (got {type(data).__name__})")
    _check_strict_keys(data, TEMPLATE_TOP_LEVEL_KEYS, strict=strict, what="template")
    normalized = dict(data)
    metadata = dict(normalized.get("metadata") or {})
    fields = dict(normalized.get("fields") or {})
    for key in (
        "kind",
        "variables",
        "tags",
        "category",
        "schema_version",
        "min_clawcodex_version",
        "output_path_template",
    ):
        if key in normalized:
            metadata.setdefault(key, normalized.pop(key))
    for key in ("content_template", "content_template_ref"):
        if key in normalized:
            fields.setdefault(key, normalized.pop(key))
    if metadata:
        normalized["metadata"] = metadata
    if fields:
        normalized["fields"] = fields
    try:
        return Template.from_dict(normalized)
    except ValueError as exc:
        # Per-field validation (id pattern, title length, etc.) bubbles
        # up as ValueError from Template.__post_init__. Wrap so callers
        # can catch the single TemplatesError family. TemplateValidation
        # errors raised by from_dict's strict-mode branch pass through
        # unchanged.
        raise TemplateValidationError(str(exc)) from exc


def parse_template_file_payload(
    data: Any,
    *,
    strict: bool = True,
) -> list[Template]:
    """Parse a template file body (single mapping or list of mappings).

    Args:
        data: Either a single mapping (one template) or a list of
            mappings (a bundle of N templates).
        strict: Forwarded to :func:`parse_template_payload` for each
            entry.

    Raises:
        TemplateCorruptError: If ``data`` is not a mapping and not a
            list (e.g. a string, scalar, or arbitrary nested shape).
        TemplateValidationError: Per-entry validation failure.
    """
    if isinstance(data, Mapping):
        return [parse_template_payload(data, strict=strict)]
    if isinstance(data, list):
        if not data:
            # An empty list is a degenerate but valid bundle — zero
            # templates. Lenient discovery treats this as "nothing
            # to register" rather than an error.
            return []
        out: list[Template] = []
        for index, entry in enumerate(data):
            if not isinstance(entry, Mapping):
                raise TemplateCorruptError(
                    f"template bundle entry #{index} must be a mapping (got {type(entry).__name__})"
                )
            try:
                out.append(parse_template_payload(entry, strict=strict))
            except TemplateValidationError as exc:
                # Re-raise with the bundle index in the message so a
                # user can locate the bad entry quickly.
                raise TemplateValidationError(f"template bundle entry #{index}: {exc}") from exc
        return out
    raise TemplateCorruptError(f"template file must be a mapping or a list of mappings (got {type(data).__name__})")


def parse_template_file(
    path: Path | str,
    *,
    strict: bool = True,
) -> list[Template]:
    """Read a template file from disk and parse it.

    Dispatches on suffix: ``.yml`` / ``.yaml`` use PyYAML's
    ``safe_load``; everything else is treated as JSON. The original
    parse error is chained via ``__cause__`` so a caller can surface
    the exact line/column of a YAML error.

    Raises:
        TemplateCorruptError: File not found, unreadable, malformed
            YAML/JSON, or top-level shape is not a mapping or list.
        TemplateValidationError: Per-entry validation failure.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise TemplateCorruptError(f"template file does not exist: {file_path}")
    if not file_path.is_file():
        raise TemplateCorruptError(f"template path is not a regular file: {file_path}")

    suffix = file_path.suffix.lower()
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateCorruptError(f"cannot read template file {file_path}: {exc}") from exc

    if suffix in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised in env without PyYAML
            raise TemplateCorruptError(f"yaml parser unavailable, cannot read {file_path}") from exc
        try:
            data: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise TemplateCorruptError(f"invalid YAML in {file_path}: {exc}") from exc
    else:
        # Default to JSON for .json, extensionless, and any unknown
        # suffix. Mirrors TemplateRegistry._parse_path's policy.
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TemplateCorruptError(f"invalid JSON in {file_path}: {exc}") from exc

    if data is None:
        # An empty file (yaml.safe_load returns None) is a degenerate
        # but valid case — zero templates.
        return []
    try:
        return parse_template_file_payload(data, strict=strict)
    except (TemplateValidationError, TemplateCorruptError):  # pylint: disable=try-except-raise
        # Already has the right type; re-raise unchanged.
        raise


__all__ = [
    "SCHEMA_DESCRIPTION",
    "TEMPLATE_FIELD_KEYS",
    "TEMPLATE_SCHEMA_VERSION",
    "TEMPLATE_TOP_LEVEL_KEYS",
    "parse_template_file",
    "parse_template_file_payload",
    "parse_template_payload",
]
