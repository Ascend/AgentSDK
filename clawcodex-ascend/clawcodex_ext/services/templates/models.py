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

"""Template data model.

A :class:`Template` is a reusable, named bundle of agent configuration
fields (tools, model, prompt fragments, max_turns, etc.) that can be
referenced from an agent definition via ``agent: template_name``. The
template itself is intentionally **not** an agent definition: the
resolver (see :mod:`resolver`) is responsible for merging a base
template with an inline override into a concrete agent config.

The schema is field-driven and permissive: anything serialisable as
JSON/YAML is a valid field value, but the template carries a strict
``id``, ``title``, optional ``description``, and an optional ``fields``
mapping that lists the field names the template declares. Validation
rejects empty ids, oversized ids, ids with whitespace or path-unsafe
characters, and any field name that is not a valid Python identifier.

This module is the single source of truth for what a template looks
like on disk; the registry and resolver layer only deal with
:class:`Template` instances, never raw dicts.

extensions (P95-A): in addition to the legacy agent-config shape,
:class:`Template` may now declare a *kind* (agent / skill / workflow /
prompt / issue / generic) plus *variable schema* / *category* / *tags*
/ *schema_version* / *min_clawcodex_version* / *output_path_template*
inside ``metadata``. The legacy surface (id / title / description /
fields / source) is unchanged, so existing tests and on-disk
templates continue to load. New helpers (:func:`kind_of`,
:func:`category_of`, :func:`tags_of`, :func:`output_path_template_of`,
:func:`content_template_of`, :func:`get_manifest`) project those
metadata fields into typed records (:class:`TemplateKind`,
:class:`TemplateVariable`, :class:`TemplateManifest`,
:class:`RenderedTemplate`) for the renderer / generator / catalogue.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# Reuse the id pattern that ultraplan/models.py established for plan ids
# so cross-service lookups stay consistent.
_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
_TITLE_MAX = 200
_DESCRIPTION_MAX = 2_000
_FIELDS_MAX = 64
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
# variable-name shape — letters / digits / underscore, must start
# with a letter or underscore. The renderer walks ``{{ name }}``
# placeholders; the names extracted there are matched against declared
# variables using this same pattern.
_VARIABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_TAG_MAX_LEN = 32
_TAGS_MAX = 16

# the closed set of template kinds. ``generic`` is the catch-all
# for templates that don't fit one of the typed slots; the renderer /
# generator treat it as a content-only template (no special output
# routing).
TemplateKind = Literal["agent", "skill", "workflow", "prompt", "issue", "generic"]
_TEMPLATE_KINDS: frozenset[str] = frozenset({"agent", "skill", "workflow", "prompt", "issue", "generic"})

# Schema version understood by this build. ``compatibility.py`` bumps
# SUPPORTED_SCHEMA_VERSIONS when an on-disk version becomes supported /
# unsupported. Templates whose ``schema_version`` is not in this set
# are rejected by :func:`check_compatibility`.
CURRENT_SCHEMA_VERSION: str = "1"


def _validate_id(value: str, *, what: str = "id") -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    if not _ID_RE.match(value):
        raise ValueError(f"{what} has invalid characters or length: {value!r} (expected [A-Za-z0-9._-]{{1,64}})")


def _validate_title(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("title must be a non-empty string")
    if len(value) > _TITLE_MAX:
        raise ValueError(f"title exceeds {_TITLE_MAX} characters (got {len(value)})")


def _validate_description(value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError("description must be a string when provided")
    if len(value) > _DESCRIPTION_MAX:
        raise ValueError(f"description exceeds {_DESCRIPTION_MAX} characters (got {len(value)})")


def _validate_fields(value: Mapping[str, Any] | None) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("fields must be a mapping when provided")
    if len(value) > _FIELDS_MAX:
        raise ValueError(f"fields exceeds {_FIELDS_MAX} entries (got {len(value)})")
    for name in value:
        if not isinstance(name, str) or not _FIELD_NAME_RE.match(name):
            raise ValueError(f"field name must match {_FIELD_NAME_RE.pattern!r}: got {name!r}")


def _validate_metadata(value: Mapping[str, Any] | None) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping when provided")
    for key in value:
        if not isinstance(key, str) or not key:
            raise ValueError(f"metadata keys must be non-empty strings: {key!r}")


def _coerce_kind(raw: Any) -> TemplateKind:
    """Coerce ``raw`` to a valid :data:`TemplateKind` literal.

    Falls back to ``"generic"`` when ``raw`` is missing or unknown. This
    is a forward-compat hedge: a future clawcodex version may add a new
    kind and an older build that does not yet recognise it should still
    be able to load the template (and silently treat it as a generic
    content blob) rather than refuse to parse it.
    """
    if raw is None:
        return "generic"
    if isinstance(raw, str) and raw in _TEMPLATE_KINDS:
        return raw  # type: ignore[return-value]
    if isinstance(raw, str):
        return "generic"
    raise ValueError(f"template kind must be a string (got {type(raw).__name__})")


def _normalize_tags(raw: Any) -> tuple[str, ...]:
    """Coerce ``raw`` (string / iterable / None) to a tag tuple.

    Accepts the legacy comma-separated string (``"python,fix"``) and a
    proper iterable of strings. Returns an empty tuple when ``raw`` is
    ``None`` or empty.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = tuple(p.strip() for p in raw.split(",") if p.strip())
        return parts
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, str)):
        out: list[str] = []
        for tag in raw:
            if not isinstance(tag, str) or not tag:
                raise ValueError("tags must be non-empty strings")
            if len(tag) > _TAG_MAX_LEN:
                raise ValueError(f"tag exceeds {_TAG_MAX_LEN} chars: {tag!r}")
            out.append(tag)
        if len(out) > _TAGS_MAX:
            raise ValueError(f"tags exceeds {_TAGS_MAX} entries (got {len(out)})")
        return tuple(out)
    raise ValueError("tags must be a string or iterable of strings")


# ---------------------------------------------------------------------------
# (P95-A): extended data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateVariable:
    """A single declared input for a template.

    (P95-A): templates may declare their inputs explicitly so the
    renderer can fail fast on missing values and the TUI picker can
    build a typed form. A variable is *required* by default; setting
    ``required=False`` (or providing a non-``None`` ``default``) marks
    it as optional. ``pattern`` is a regex the supplied value must
    match (string-coerced). ``choices`` is an explicit enum of accepted
    values. ``secret=True`` redacts the supplied value from any preview
    / log surface — the renderer still substitutes it into the rendered
    body, but it never appears in ``RenderedTemplate.warnings`` and
    never round-trips through the preview path.
    """

    name: str
    description: str = ""
    required: bool = True
    default: str | int | float | bool | None = None
    pattern: str | None = None
    choices: tuple[str, ...] = ()
    secret: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _VARIABLE_NAME_RE.match(self.name):
            raise ValueError(f"variable name must match {_VARIABLE_NAME_RE.pattern!r}: got {self.name!r}")
        if not isinstance(self.description, str):
            raise TypeError("variable description must be a string")
        if not isinstance(self.required, bool):
            raise TypeError("variable required must be a bool")
        if not isinstance(self.secret, bool):
            raise TypeError("variable secret must be a bool")
        if self.default is not None and not isinstance(self.default, (str, int, float, bool)):
            raise ValueError("variable default must be str / int / float / bool / None")
        if self.pattern is not None:
            if not isinstance(self.pattern, str):
                raise ValueError("variable pattern must be a string when provided")
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"variable pattern is not a valid regex: {exc}") from exc
        if not isinstance(self.choices, tuple):
            raise TypeError("variable choices must be a tuple of strings")
        for choice in self.choices:
            if not isinstance(choice, str):
                raise TypeError("variable choice entries must be strings")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "secret": self.secret,
        }
        if self.default is not None:
            out["default"] = self.default
        if self.pattern is not None:
            out["pattern"] = self.pattern
        if self.choices:
            out["choices"] = list(self.choices)
        return out

    @classmethod
    def from_dict(cls, data: Any) -> TemplateVariable:
        if not isinstance(data, Mapping):
            raise TypeError("variable payload must be a mapping")
        if "name" not in data:
            raise ValueError("variable payload missing required key: name")
        choices_raw = data.get("choices", ())
        if isinstance(choices_raw, list):
            choices: tuple[str, ...] = tuple(choices_raw)
        elif isinstance(choices_raw, tuple):
            choices = choices_raw
        elif choices_raw == ():
            choices = ()
        else:
            raise ValueError("variable choices must be a list/tuple when provided")
        return cls(
            name=data["name"],
            description=str(data.get("description", "")),
            required=bool(data.get("required", True)),
            default=data.get("default"),
            pattern=data.get("pattern"),
            choices=choices,
            secret=bool(data.get("secret", False)),
        )


@dataclass(frozen=True)
class TemplateManifest:
    """The "what is this template" projection of a :class:`Template`.

    (P95-A): the manifest is what the catalogue / picker / CLI
    consume. It is derived from the on-disk :class:`Template` (via
    :func:`get_manifest`) so the storage surface stays a single
    dataclass but consumers get a typed view that is safe to render,
    search, and filter on.

    Attributes:
        id: Stable template id (see :data:`_ID_RE`).
        title: Human-readable name.
        kind: One of the :data:`TemplateKind` literals.
        description: Optional long-form description.
        variables: Declared input variables (ordered as declared).
        tags: Search/filter tags.
        category: Optional coarse grouping label (e.g. ``"search"``,
            ``"edit"``). The built-in agents already use this in
            ``metadata.category``; surfaces it directly.
        schema_version: On-disk schema version (defaults to ``"1"``).
        min_clawcodex_version: Optional minimum clawcodex build id; the
            compatibility gate enforces this.
        output_path_template: Optional path pattern (relative to the
            workspace root) into which the rendered content is written.
            May contain ``{{ var }}`` placeholders; the renderer
            substitutes them when materialising the absolute path.
        source: Provenance label (built-in / user / project / managed /
            plugin).
    """

    id: str
    title: str
    kind: TemplateKind = "generic"
    description: str | None = None
    variables: tuple[TemplateVariable, ...] = ()
    tags: tuple[str, ...] = ()
    category: str | None = None
    schema_version: str = CURRENT_SCHEMA_VERSION
    min_clawcodex_version: str | None = None
    output_path_template: str | None = None
    source: str = "user"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "source": self.source,
            "schema_version": self.schema_version,
        }
        if self.description is not None:
            out["description"] = self.description
        if self.variables:
            out["variables"] = [v.to_dict() for v in self.variables]
        if self.tags:
            out["tags"] = list(self.tags)
        if self.category is not None:
            out["category"] = self.category
        if self.min_clawcodex_version is not None:
            out["min_clawcodex_version"] = self.min_clawcodex_version
        if self.output_path_template is not None:
            out["output_path_template"] = self.output_path_template
        return out


@dataclass(frozen=True)
class RenderedTemplate:
    """The output of :meth:`TemplateRenderer.render`.

    Attributes:
        template_id: The id of the source :class:`Template`.
        kind: The template kind (passed through from the manifest).
        content: The rendered body (variable substitution complete).
        output_path: Absolute path the content should be written to.
            ``None`` when the template carries no ``output_path_template``
            or the caller asked for a preview.
        variables_used: Mapping of variable name -> supplied value
            (stringified). ``secret=True`` variables are **not** stored
            here — they live only in the rendered body. Tests / debug
            output can rely on this contract.
        warnings: Tuple of human-readable warnings (shadow keys, choice
            fallbacks, missing optional variables, etc.). Empty when
            the render was clean.
    """

    template_id: str
    kind: TemplateKind
    content: str
    output_path: Any = None  # Path | None — kept loose to avoid Path import cycles
    variables_used: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Template:
    """A reusable template.

    Attributes:
        id: Stable identifier referenced by ``agent: <id>`` in agent
            definitions. Id-pattern enforced (see :data:`_ID_RE`).
        title: Short human-readable name (max 200 chars).
        description: Optional long-form description (max 2_000 chars).
        fields: Optional mapping of field name -> default value. Used
            both to declare what the template contributes and to give
            a baseline value when no inline override is supplied. Field
            names must be valid Python identifiers. also recognises
            two special keys here: ``output_path_template`` and
            ``content_template`` — when present the renderer / generator
            use them as the materialised path and rendered body
            respectively (so on-disk templates can be plain YAML/JSON
            without touching the new ``manifest`` block).
        metadata: Optional free-form metadata. reads the following
            keys (all optional): ``kind``, ``category``, ``tags``,
            ``schema_version``, ``min_clawcodex_version``,
            ``output_path_template``, ``variables``. Unknown keys are
            tolerated for forward-compat.
        source: Provenance label — ``built-in``, ``user``, ``project``,
            ``managed``, or ``plugin``. Mirrors ``AgentSource`` in
            :mod:`src.agent.agent_definitions` but kept as a plain
            string here so this module has no upstream dependency.
    """

    id: str
    title: str
    description: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "user"

    def __post_init__(self) -> None:
        _validate_id(self.id)
        _validate_title(self.title)
        _validate_description(self.description)
        _validate_fields(self.fields)
        _validate_metadata(self.metadata)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "source": self.source,
        }
        if self.description is not None:
            out["description"] = self.description
        if self.fields:
            out["fields"] = dict(self.fields)
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out

    @classmethod
    def from_dict(cls, data: Any) -> Template:
        if not isinstance(data, dict):
            raise ValueError("template payload must be a JSON object")
        # Required fields. Missing -> fail loudly (corrupt payload).
        if "id" not in data:
            raise ValueError("template payload missing required key: id")
        if "title" not in data:
            raise ValueError("template payload missing required key: title")
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description"),
            fields=data.get("fields") or {},
            metadata=data.get("metadata") or {},
            source=data.get("source", "user"),
        )


# ---------------------------------------------------------------------------
# (P95-A) projections: helper accessors that lift metadata into the
# typed model. Kept as free functions so a :class:`Template` (frozen,
# stored everywhere) can be projected without copying.
# ---------------------------------------------------------------------------


def kind_of(template: Template) -> TemplateKind:
    """Return the ``kind`` declared on ``template.metadata``."""
    return _coerce_kind(template.metadata.get("kind"))


def category_of(template: Template) -> str | None:
    """Return the optional ``category`` declared on ``template.metadata``."""
    raw = template.metadata.get("category")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError("metadata.category must be a string when provided")
    return raw or None


def tags_of(template: Template) -> tuple[str, ...]:
    """Return the ``tags`` declared on ``template.metadata``.

    Accepts the legacy comma-separated string (``"python,fix"``) and a
    proper iterable of strings. Returns an empty tuple when absent.
    """
    return _normalize_tags(template.metadata.get("tags"))


def variables_of(template: Template) -> tuple[TemplateVariable, ...]:
    """Parse the optional ``variables`` block on ``template.metadata``.

    Each entry is re-parsed via :meth:`TemplateVariable.from_dict` so a
    single bad entry surfaces as a :class:`TemplateValidationError` (in
    the caller's frame) rather than silently dropping the variable.
    """
    raw = template.metadata.get("variables")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("metadata.variables must be a list of mappings")
    out: list[TemplateVariable] = []
    for index, entry in enumerate(raw):
        try:
            out.append(TemplateVariable.from_dict(entry))
        except ValueError as exc:
            raise ValueError(f"variables[{index}]: {exc}") from exc
    return tuple(out)


def output_path_template_of(template: Template) -> str | None:
    """Return the ``output_path_template`` (preferred metadata)."""
    raw_meta = template.metadata.get("output_path_template")
    if raw_meta is not None:
        if not isinstance(raw_meta, str):
            raise ValueError("metadata.output_path_template must be a string")
        return raw_meta or None
    raw_fields = template.fields.get("output_path_template")
    if raw_fields is not None:
        if not isinstance(raw_fields, str):
            raise ValueError("fields.output_path_template must be a string")
        return raw_fields or None
    return None


def content_template_of(template: Template) -> str | None:
    """Return the ``content_template`` (preferred metadata, then fields)."""
    raw_meta = template.metadata.get("content_template")
    if raw_meta is not None:
        if not isinstance(raw_meta, str):
            raise ValueError("metadata.content_template must be a string")
        return raw_meta or None
    raw_fields = template.fields.get("content_template")
    if raw_fields is not None:
        if not isinstance(raw_fields, str):
            raise ValueError("fields.content_template must be a string")
        return raw_fields or None
    return None


def schema_version_of(template: Template) -> str:
    """Return the on-disk ``schema_version`` (defaults to current)."""
    raw = template.metadata.get("schema_version")
    if raw is None:
        return CURRENT_SCHEMA_VERSION
    if not isinstance(raw, str):
        raise TypeError("metadata.schema_version must be a string when provided")
    return raw


def min_clawcodex_version_of(template: Template) -> str | None:
    """Return the optional ``min_clawcodex_version`` declared on metadata."""
    raw = template.metadata.get("min_clawcodex_version")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError("metadata.min_clawcodex_version must be a string when provided")
    return raw or None


def get_manifest(template: Template) -> TemplateManifest:
    """Project ``template`` onto a typed :class:`TemplateManifest`.

    The manifest is the shape catalogue / renderer / generator consume.
    Building it on demand keeps :class:`Template` as the single storage
    surface (so existing tests and on-disk YAML keep working) while
    letting the surface operate on typed records.
    """
    return TemplateManifest(
        id=template.id,
        title=template.title,
        kind=kind_of(template),
        description=template.description,
        variables=variables_of(template),
        tags=tags_of(template),
        category=category_of(template),
        schema_version=schema_version_of(template),
        min_clawcodex_version=min_clawcodex_version_of(template),
        output_path_template=output_path_template_of(template),
        source=template.source,
    )


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "RenderedTemplate",
    "Template",
    "TemplateKind",
    "TemplateManifest",
    "TemplateVariable",
    "category_of",
    "content_template_of",
    "get_manifest",
    "kind_of",
    "min_clawcodex_version_of",
    "output_path_template_of",
    "schema_version_of",
    "tags_of",
    "variables_of",
]
