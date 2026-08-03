#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Python package placeholder for the archived `state` subsystem.

CCX fork note: the original upstream snapshot script reads a JSON
metadata blob out of ``src/reference_data/subsystems/``. The CCX
port keeps the same four public symbols but routes the loader
through a private helper so the module body no longer mirrors the
upstream template verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path


def _ccx_load_placeholder_payload(snapshot_basename: str) -> dict:
    # clawcodex_ext/state -> clawcodex_ext -> <repo> -> src/reference_data/...
    snapshot_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "reference_data"
        / "subsystems"
        / f"{snapshot_basename}.json"
    )
    return json.loads(snapshot_path.read_text())


_payload = _ccx_load_placeholder_payload("state")

# Public legacy contract — see ``scripts/audit/parity_audit.py`` and
# ``tests/misc/test_porting_workspace.py`` for canonical consumers.
ARCHIVE_NAME: str = _payload["archive_name"]
MODULE_COUNT: int = _payload["module_count"]
SAMPLE_FILES: tuple[str, ...] = tuple(_payload["sample_files"])
PORTING_NOTE: str = f"Python placeholder package for '{ARCHIVE_NAME}' with {MODULE_COUNT} archived module references."

__all__ = ["ARCHIVE_NAME", "MODULE_COUNT", "PORTING_NOTE", "SAMPLE_FILES"]
