#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.


"""Extractor registry."""

from __future__ import annotations

import logging
from pathlib import Path

from ..discriminator import _detect_adapter_name
from ..scan_context import SourceScanContext
from .adapters.generic import GenericPipelineExtractor
from .base import WorkflowExtractorBase

logger = logging.getLogger(__name__)

_PROJECT_ADAPTERS: dict[str, type[WorkflowExtractorBase]] = {
    # Custom extractors can be registered via register_adapter()
}


class ExtractorRegistry:
    @staticmethod
    def get_extractor(
        source_dir: Path,
        *,
        name: str | None = None,
        scan: SourceScanContext | None = None,
        mode: str = "fwa",
        allow_coarse: bool = False,
    ) -> WorkflowExtractorBase:
        resolved = name or _detect_adapter_name(source_dir)
        cls = _PROJECT_ADAPTERS.get(resolved, GenericPipelineExtractor)
        return cls(scan=scan, mode=mode, allow_coarse=allow_coarse)

    @staticmethod
    def register_adapter(project_name: str, extractor_cls: type[WorkflowExtractorBase]) -> None:
        _PROJECT_ADAPTERS[project_name] = extractor_cls

    @staticmethod
    def available_adapters() -> list[str]:
        return list(_PROJECT_ADAPTERS.keys())
