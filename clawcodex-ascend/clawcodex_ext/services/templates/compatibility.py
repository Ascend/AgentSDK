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

"""Compatibility checks for templates."""

from __future__ import annotations

from .exceptions import TemplateCompatibilityError
from .models import CURRENT_SCHEMA_VERSION, Template, get_manifest

SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({CURRENT_SCHEMA_VERSION})


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw in value.strip().removeprefix("v").split("."):
        if not raw:
            continue
        digits = ""
        for ch in raw:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def _current_version() -> str:
    try:
        from clawcodex_ext._version import __version__  # pylint: disable=no-name-in-module

        return __version__
    except Exception:
        return "0.0.0"


def check_compatibility(
    template: Template,
    *,
    current_version: str | None = None,
) -> None:
    """Raise when ``template`` declares an unsupported schema/version."""

    manifest = get_manifest(template)
    if manifest.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise TemplateCompatibilityError(
            f"template {template.id!r} uses unsupported schema_version "
            f"{manifest.schema_version!r}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if manifest.min_clawcodex_version:
        running = current_version or _current_version()
        if _version_tuple(running) < _version_tuple(manifest.min_clawcodex_version):
            raise TemplateCompatibilityError(
                f"template {template.id!r} requires clawcodex {manifest.min_clawcodex_version}, running {running}"
            )


__all__ = ["SUPPORTED_SCHEMA_VERSIONS", "check_compatibility"]
