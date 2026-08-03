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

"""Minimal F-95 template picker helpers.

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
