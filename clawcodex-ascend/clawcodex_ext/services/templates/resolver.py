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

"""Template resolver: merge a base template with an inline override.

The resolver takes a base :class:`Template` plus an optional override
mapping and produces a fully-merged configuration dict that downstream
code can hand to an agent spawner. The merge follows these rules:

* **Override fields win** — every key in the override replaces the
  template's default value for that key.
* **Lists are replaced wholesale** — an override list does not get
  appended to the base list. This keeps the semantics obvious: an
  inline ``tools: [Read]`` overrides ``tools: [Bash, Read, Write]``
  entirely. (If a future caller needs list-concat semantics, they can
  compose it at the caller layer by reading the base value first.)
* **Dicts merge recursively** — a nested mapping in the override
  replaces the matching nested mapping in the base, with the same
  override-wins / list-replace rules applied at every level.
* **Non-overridden template fields are preserved** — keys present in
  the base but not in the override are carried through unchanged.
* **Override-only fields are accepted but warned about** — keys in the
  override that are not declared in ``base.fields`` are passed through
  with a flag in the result so callers can detect "shadow" overrides.
* **Falsy is allowed** — an override of ``False`` or ``0`` or ``""``
  is honoured; only the *absence* of a key signals "no override".

The resolver is pure: no I/O, no mutation of the inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .exceptions import TemplateResolutionError
from .models import Template
from .registry import TemplateRegistry


def _merge(base: Any, override: Any) -> Any:
    """Recursively merge two values, override wins."""
    # List / tuple: replace wholesale.
    if isinstance(base, list) and isinstance(override, list):
        return list(override)
    if isinstance(base, tuple) and isinstance(override, tuple):
        return tuple(override)
    # Dict: deep merge.
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        out: dict[Any, Any] = dict(base)
        for k, v in override.items():
            if k in out:
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out
    # Scalars and type-mismatched pairs: override wins.
    return override


@dataclass(frozen=True)
class ResolvedTemplate:
    """The output of a resolution operation.

    Attributes:
        template_id: The base template's id (preserved for tracing).
        fields: The merged field mapping.
        shadow_keys: Override keys not declared in the base's
            ``fields`` mapping. Callers can use this to flag drift or
            surface warnings to the user.
        base_template: The base :class:`Template` instance, kept for
            callers that want to introspect title / description /
            metadata after resolution.
    """

    template_id: str
    fields: dict[str, Any] = field(default_factory=dict)
    shadow_keys: list[str] = field(default_factory=list)
    base_template: Template | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)


class TemplateResolver:
    """Pure resolver: base + override -> ResolvedTemplate."""

    def resolve(
        self,
        template: Template,
        override: Mapping[str, Any] | None = None,
    ) -> ResolvedTemplate:
        if not isinstance(template, Template):
            raise TypeError("resolve() expects a Template instance")
        if override is None:
            override = {}
        if not isinstance(override, Mapping):
            raise TemplateResolutionError("override must be a mapping when provided")

        merged: dict[str, Any] = {}
        for key, value in template.fields.items():
            merged[key] = value

        shadow: list[str] = []
        for key, value in override.items():
            if not isinstance(key, str):
                raise TemplateResolutionError(f"override keys must be strings (got {type(key).__name__})")
            if key in merged:
                merged[key] = _merge(merged[key], value)
            else:
                merged[key] = value
                shadow.append(key)
        shadow.sort()
        return ResolvedTemplate(
            template_id=template.id,
            fields=merged,
            shadow_keys=shadow,
            base_template=template,
        )

    # ------------------------------------------------------------------
    # Convenience: resolve by id from a registry
    # ------------------------------------------------------------------

    def resolve_from_registry(
        self,
        registry: TemplateRegistry,
        template_id: str,
        override: Mapping[str, Any] | None = None,
    ) -> ResolvedTemplate:
        if not isinstance(registry, TemplateRegistry):
            raise TypeError("resolve_from_registry expects a TemplateRegistry")
        template = registry.get(template_id)
        return self.resolve(template, override)


__all__ = [
    "ResolvedTemplate",
    "TemplateResolver",
]
