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

"""Safe, expression-free rendering for F-95 templates."""  # pylint: disable=r0401

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .compatibility import check_compatibility
from .exceptions import TemplateRenderError, TemplateUnsafePathError
from .models import (
    RenderedTemplate,
    Template,
    TemplateVariable,
    content_template_of,
    get_manifest,
    output_path_template_of,
)

_PLACEHOLDER_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_MAX_RENDER_BYTES = 2_000_000
_MISSING = object()


def _content_template_ref_of(template: Template) -> str | None:
    raw = template.metadata.get("content_template_ref")
    if raw is None:
        raw = template.fields.get("content_template_ref")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TemplateRenderError("content_template_ref must be a string")
    return raw or None


def _read_ref(path_text: str, *, workspace_root: Path | None = None) -> str:
    path = Path(path_text)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    if workspace_root is not None:
        candidates.append(workspace_root / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        except OSError as exc:
            raise TemplateRenderError(f"cannot read content_template_ref {candidate}: {exc}") from exc
    raise TemplateRenderError(f"content_template_ref not found: {path_text}")


def _validate_value(var: TemplateVariable, value: Any) -> str:
    text = str(value)
    if var.choices and text not in var.choices:
        raise TemplateRenderError(f"variable {var.name!r} must be one of {list(var.choices)!r}; got {text!r}")
    if var.pattern is not None and re.fullmatch(var.pattern, text) is None:
        raise TemplateRenderError(f"variable {var.name!r} does not match pattern {var.pattern!r}: {text!r}")
    return text


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class TemplateRenderer:
    """Render ``{{ name }}`` placeholders without eval/Jinja execution."""

    def __init__(self, *, max_render_bytes: int = _MAX_RENDER_BYTES) -> None:
        self.max_render_bytes = max_render_bytes

    def render(
        self,
        template: Template,
        variables: Mapping[str, object],
        *,
        workspace_root: Path | str | None = None,
    ) -> RenderedTemplate:
        check_compatibility(template)
        manifest = get_manifest(template)
        root = Path(workspace_root).expanduser().resolve() if workspace_root is not None else None
        content_template = content_template_of(template)
        ref = _content_template_ref_of(template)
        if content_template is None and ref is not None:
            content_template = _read_ref(ref, workspace_root=root)
        if content_template is None:
            content_template = str(template.fields.get("content", ""))

        values, public_values, warnings = self._resolve_variables(
            template,
            variables,
            content_template=content_template,
        )
        content = self._substitute(content_template, values)
        if len(content.encode("utf-8")) > self.max_render_bytes:
            raise TemplateRenderError(f"rendered template exceeds {self.max_render_bytes} bytes")

        output_path = None
        path_template = output_path_template_of(template)
        if path_template:
            rendered_path = self._substitute(path_template, values)
            if root is not None:
                output_path = self._resolve_output_path(rendered_path, root)
            else:
                output_path = Path(rendered_path)

        return RenderedTemplate(
            template_id=template.id,
            kind=manifest.kind,
            content=content,
            output_path=output_path,
            variables_used=public_values,
            warnings=tuple(warnings),
        )

    def preview(
        self,
        template: Template,
        variables: Mapping[str, object],
        *,
        workspace_root: Path | str | None = None,
    ) -> RenderedTemplate:
        """Render without writing files."""

        return self.render(template, variables, workspace_root=workspace_root)

    def _resolve_variables(
        self,
        template: Template,
        variables: Mapping[str, object],
        *,
        content_template: str,
    ) -> tuple[dict[str, str], dict[str, str], list[str]]:
        declared = {v.name: v for v in get_manifest(template).variables}
        referenced = set(_PLACEHOLDER_RE.findall(content_template))
        path_template = output_path_template_of(template)
        if path_template:
            referenced.update(_PLACEHOLDER_RE.findall(path_template))
        values: dict[str, str] = {}
        public_values: dict[str, str] = {}
        warnings: list[str] = []

        for name, var in declared.items():
            raw = variables.get(name, _MISSING)
            if raw is _MISSING:
                if var.default is not None:
                    raw = var.default
                elif var.required:
                    raise TemplateRenderError(f"missing required variable: {name}")
                else:
                    warnings.append(f"optional variable {name!r} was not provided")
                    raw = ""
            text = _validate_value(var, raw)
            values[name] = text
            if not var.secret:
                public_values[name] = text

        for name in sorted(referenced):
            if name in values:
                continue
            if name in variables:
                text = str(variables[name])
                values[name] = text
                public_values[name] = text
                continue
            raise TemplateRenderError(f"missing required variable: {name}")

        for name, raw in variables.items():
            if name not in values:
                text = str(raw)
                values[name] = text
                public_values[name] = text
                warnings.append(f"unused variable {name!r}")
        return values, public_values, warnings

    def _substitute(self, text: str, values: Mapping[str, str]) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], text)

    def _resolve_output_path(self, rendered_path: str, workspace_root: Path) -> Path:
        path = Path(rendered_path)
        if path.is_absolute():
            candidate = path.expanduser().resolve()
        else:
            candidate = (workspace_root / path).expanduser().resolve()
        if not _is_relative_to(candidate, workspace_root):
            raise TemplateUnsafePathError(f"output path escapes workspace: {candidate} (root: {workspace_root})")
        return candidate


__all__ = ["TemplateRenderer"]
