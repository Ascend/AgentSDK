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


"""Resolve workflow artifacts inside a POS bundle directory."""

from __future__ import annotations

import logging
from pathlib import Path

from .bundle_manifest import read_bundle_manifest

logger = logging.getLogger(__name__)


def bundle_dir_from_workflow_yaml(workflow_yaml: Path) -> Path:
    return workflow_yaml.resolve().parent


def resolve_bundle_workflow_yaml(bundle_path: Path) -> Path | None:
    """Return ``workflow.yaml`` for *bundle_path* if present."""
    bundle_path = bundle_path.resolve()
    manifest = read_bundle_manifest(bundle_path)
    if manifest is not None and manifest.workflow_yaml:
        candidate = bundle_path / manifest.workflow_yaml
        if candidate.is_file():
            return candidate

    direct = bundle_path / "workflow.yaml"
    if direct.is_file():
        return direct
    return None


def discover_workflow_yaml(workspace: Path) -> Path | None:
    """Find the first workflow bundle under ``workspace/.clawcodex/*/``."""
    workspace = workspace.resolve()
    clawcodex_root = workspace / ".clawcodex"
    if not clawcodex_root.is_dir():
        return None

    candidates: list[Path] = []
    for child in sorted(clawcodex_root.iterdir()):
        if not child.is_dir():
            continue
        wf = resolve_bundle_workflow_yaml(child)
        if wf is not None:
            candidates.append(wf)

    if not candidates:
        return None
    if len(candidates) > 1:
        logger.info(
            "Multiple workflow bundles found under %s; using %s",
            clawcodex_root,
            candidates[0],
        )
    return candidates[0]


def workflow_artifacts_enabled(
    *,
    has_mapped_stages: bool,
    workflow_mode: str | None,
) -> bool:
    """Whether workflow artifacts should be emitted."""
    if has_mapped_stages:
        return True
    return workflow_mode in ("fwa", "hybrid")
