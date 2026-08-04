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

"""F-97 LODESTONE — configuration loading & persistence.

*   ``LodestoneConfig`` is the only source of truth (declared in
    :mod:`clawcodex_ext.services.lodestone.models`).
*   :func:`load_config` reads ``~/.clawcodex/lodestone.json``; missing
    or invalid files fall back to :func:`default_config`.
*   :func:`save_config` writes back atomically (write to ``.tmp``
    + :func:`os.replace`).
*   The ``LODESTONE`` environment variable acts as a kill-switch:
    ``LODESTONE=off`` forces :attr:`LodestoneConfig.enabled` to ``False``
    on load, regardless of what's persisted.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import LodestoneConfig

log = logging.getLogger(__name__)


def config_dir() -> Path:
    """Return the per-user configuration directory (creates it lazily)."""
    base = Path(os.environ.get("CLAWCODEX_CONFIG_DIR") or Path.home() / ".clawcodex")
    base.mkdir(parents=True, exist_ok=True)
    return base


def config_path() -> Path:
    return config_dir() / "lodestone.json"


def default_config() -> LodestoneConfig:
    """Conservative default: enabled, vscode preferred, file fallback."""
    return LodestoneConfig(
        enabled=True,
        default_editor="vscode",
        fallback_editor="file",
        auto_remote=True,
        disabled_kinds=(),
        renderer="auto",
        custom_targets=(),
        default_tracker_host="gitcode.com",
        default_tracker_repo=None,
        extra_hosts=(),
    )


def load_config(path: Path | None = None) -> LodestoneConfig:
    """Load config from disk, falling back to defaults on any failure.

    Order of precedence (lowest → highest):

    1.  :func:`default_config`
    2.  disk file at ``path`` (or ``~/.clawcodex/lodestone.json``)
    3.  ``LODESTONE=off`` environment kill-switch
    """
    cfg = default_config()
    target = path or config_path()
    if target.exists():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            cfg = _from_dict(raw)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            log.debug("failed to read lodestone config %s: %s", target, exc)
            cfg = default_config()

    if (os.environ.get("LODESTONE") or "").lower() in {"off", "0", "false", "no"}:
        cfg = LodestoneConfig(**{**asdict(cfg), "enabled": False})
    return cfg


def save_config(cfg: LodestoneConfig, path: Path | None = None) -> Path:
    """Persist ``cfg`` atomically; return the path written."""
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115  # pylint: disable=consider-using-with
        mode="w",
        delete=False,
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
        encoding="utf-8",
    )
    try:
        json.dump(_to_dict(cfg), tmp, ensure_ascii=False, indent=2)
        tmp.close()
        os.replace(tmp.name, target)
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    return target


def _to_dict(cfg: LodestoneConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    # Convert tuple fields to lists for JSON friendliness.
    for key, value in list(payload.items()):
        if isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def _from_dict(raw: dict[str, Any]) -> LodestoneConfig:
    """Inverse of :func:`_to_dict` with extra resilience."""
    defaults = asdict(default_config())
    defaults.update({k: v for k, v in raw.items() if k in defaults})
    # Re-tuplify list fields.
    for field_name in (
        "disabled_kinds",
        "custom_targets",
        "extra_hosts",
        "custom_placeholder_resolvers",
    ):
        if isinstance(defaults.get(field_name), list):
            defaults[field_name] = tuple(defaults[field_name])
    if not isinstance(defaults.get("default_tracker_repo"), (list, tuple)):
        defaults["default_tracker_repo"] = None
    elif isinstance(defaults["default_tracker_repo"], list) and len(defaults["default_tracker_repo"]) == 2:
        defaults["default_tracker_repo"] = tuple(defaults["default_tracker_repo"])
    return LodestoneConfig(**defaults)


__all__ = [
    "config_dir",
    "config_path",
    "default_config",
    "load_config",
    "save_config",
]
