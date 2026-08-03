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

"""Helpers for external-agent REPL debugging."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, TextIO


_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"", "0", "false", "no", "off"}


def agent_debug_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = str(env.get("CLAWCODEX_AGENT_DEBUG", "")).strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSEY:
        return False
    return False


def resolve_repl_history_file(
    environ: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    explicit = str(env.get("CLAWCODEX_HISTORY_FILE", "")).strip()
    if explicit:
        return Path(explicit).expanduser()

    home_path = Path.home() if home is None else home
    if not agent_debug_enabled(env):
        return home_path / ".clawcodex" / "history"

    debug_dir = str(env.get("CLAWCODEX_AGENT_DEBUG_DIR", "")).strip()
    base = Path(debug_dir).expanduser() if debug_dir else Path(tempfile.gettempdir()) / "clawcodex-agent-debug"
    return base / "history"


def resolve_agent_debug_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    debug_dir = str(env.get("CLAWCODEX_AGENT_DEBUG_DIR", "")).strip()
    if debug_dir:
        return Path(debug_dir).expanduser()
    return Path(tempfile.gettempdir()) / "clawcodex-agent-debug"


def apply_agent_debug_environment(
    environ: dict[str, str] | None = None,
    *,
    debug_dir: Path | None = None,
) -> dict[str, str]:
    env = os.environ if environ is None else environ
    base = debug_dir if debug_dir is not None else resolve_agent_debug_dir(env)

    env["CLAWCODEX_AGENT_DEBUG"] = "1"
    env["CLAWCODEX_AGENT_DEBUG_DIR"] = str(base)
    env["CLAWCODEX_HOME"] = str(base)
    env["CLAWCODEX_HISTORY_FILE"] = str(base / "history")
    env["CLAWCODEX_SESSIONS_DIR"] = str(base / "sessions")
    env["CLAW_TELEMETRY_STORAGE_DIR"] = str(base / "telemetry")
    return env


def emit_agent_debug_marker(
    name: str,
    payload: Mapping[str, object] | None = None,
    *,
    stream: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    if not agent_debug_enabled(environ):
        return

    target = sys.stderr if stream is None else stream
    body = json.dumps(dict(payload or {}), sort_keys=True, separators=(",", ":"))
    target.write(f"CLAWCODEX_AGENT_DEBUG::{name}::{body}\n")
    target.flush()
