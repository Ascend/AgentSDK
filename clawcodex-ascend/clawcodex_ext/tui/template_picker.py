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

"""Minimal template picker helpers.

The full interactive picker can grow on top of this adapter; for now it
exposes the same catalogue ordering that the TUI can render in a select list.
"""

from __future__ import annotations

from dataclasses import dataclass

from clawcodex_ext.services.templates import (
    TemplateCatalogue,
    TemplateKind,
    TemplateRegistry,
    get_manifest,
)


@dataclass(frozen=True)
class TemplatePickerItem:
    id: str
    label: str
    description: str
    kind: TemplateKind


def build_template_picker_items(
    registry: TemplateRegistry,
    *,
    kind: TemplateKind | None = None,
) -> list[TemplatePickerItem]:
    catalogue = TemplateCatalogue(registry)
    items: list[TemplatePickerItem] = []
    for template in catalogue.list(kind=kind):
        manifest = get_manifest(template)
        items.append(
            TemplatePickerItem(
                id=template.id,
                label=template.title,
                description=template.description or "",
                kind=manifest.kind,
            )
        )
    return items


__all__ = ["TemplatePickerItem", "build_template_picker_items"]
