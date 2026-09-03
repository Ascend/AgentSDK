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

"""Resolve bundle-local and home-fallback paths for the agent catalog."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from extensions.sop_converter.bundle_context import BundleContext

HOME_ROOT_ENV = "CLAWCODEX_HOME"
HOME_ONLY_ENV = "CLAWCODEX_CATALOG_HOME_ONLY"
DEFAULT_HOME = Path.home() / ".clawcodex"


@dataclass(frozen=True)
class CatalogLocation:
    """Resolved catalog path together with its selection reason."""

    path: Path
    reason: str
    writable: bool | None = None

    def ensure_parent(self) -> None:
        """Create the catalog parent directory when it does not exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)


def _clawcodex_home() -> Path:
    raw = os.environ.get(HOME_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_HOME.resolve()


def _is_home_only_forced() -> bool:
    return os.environ.get(HOME_ONLY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _home_fallback_path(bundle_id: str | None) -> Path:
    leaf = (bundle_id or "default").strip() or "default"
    return _clawcodex_home() / "sop-agents" / leaf / "agent-catalog.json"


def resolve_catalog_path(
    bundle: "BundleContext | Path | str | None" = None,
    *,
    bundle_id: str | None = None,
    home_only: bool | None = None,
) -> CatalogLocation:
    """Choose the catalog location without creating directories."""
    if home_only is None:
        home_only = _is_home_only_forced()

    bundle_path: Path | None = None
    resolved_bundle_id = bundle_id
    if isinstance(bundle, str) and not bundle.strip():
        bundle = None
    if bundle is not None:
        if isinstance(bundle, (str, Path)):
            bundle_path = Path(bundle).expanduser()
            if resolved_bundle_id is None:
                resolved_bundle_id = bundle_path.name
        else:
            bundle_path = Path(bundle.bundle_path).expanduser()
            if resolved_bundle_id is None:
                resolved_bundle_id = bundle.bundle_name

    if home_only or bundle_path is None:
        path = _home_fallback_path(resolved_bundle_id)
        reason = "home-forced" if home_only else "no-bundle"
        return CatalogLocation(path=path, reason=reason, writable=_probe_writable(path))

    path = bundle_path / ".clawcodex" / "agent-catalog.json"
    return CatalogLocation(path=path, reason="bundle-local", writable=_probe_writable(path))


def _probe_writable(path: Path) -> bool | None:
    parent = path.parent
    if not parent.exists():
        return None
    return os.access(parent, os.W_OK)


__all__ = [
    "CatalogLocation",
    "HOME_ONLY_ENV",
    "HOME_ROOT_ENV",
    "resolve_catalog_path",
]
