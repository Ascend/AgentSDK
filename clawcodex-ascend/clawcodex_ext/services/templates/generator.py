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

"""Materialize rendered templates onto disk."""

from __future__ import annotations

from pathlib import Path

from .exceptions import TemplateOverwriteError, TemplateRenderError, TemplateUnsafePathError
from .models import RenderedTemplate


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class TemplateGenerator:
    """Write rendered template content with containment and overwrite checks."""

    def __init__(self, *, workspace_root: Path | str, allowed_roots: tuple[Path | str, ...] = ()) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.allowed_roots = tuple(
            [self.workspace_root] + [Path(root).expanduser().resolve() for root in allowed_roots]
        )

    def generate(self, rendered: RenderedTemplate, *, overwrite: bool = False) -> Path:
        if rendered.output_path is None:
            raise TemplateRenderError("rendered template has no output_path")
        target = Path(rendered.output_path).expanduser().resolve()
        if not any(_is_relative_to(target, root) for root in self.allowed_roots):
            raise TemplateUnsafePathError(f"output path is outside allowed roots: {target}")
        if target.exists() and not overwrite:
            raise TemplateOverwriteError(f"output file already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered.content, encoding="utf-8")
        return target


__all__ = ["TemplateGenerator"]
