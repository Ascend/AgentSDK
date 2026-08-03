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

"""Layer-2/Layer-1 freeze-configuration resolution (F-108 §十八 P108-E).

Centralises the env-var → settings dataclass resolution the freeze
subsystem depends on, so a single import surface can answer:

* What is the current Layer-2 budget for an agent loop? → ``resolve_freeze_settings().agent_loop_timeout_s``
* Is freeze diagnostics enabled for this run? → ``bool(os.environ.get("CLAWCODEX_FREEZE_DIAG", ""))``

The dataclass shape mirrors :class:`clawcodex_ext.settings.types.FreezeSettings`
so consumers that already load settings can substitute one for the
other at the boundary. This module only reads env vars + dataclass
defaults — it never instantiates a ``get_settings()`` call, so importing
is safe from cold paths (CLI subcommand dispatch, REPL init, TUI
worker).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# F-108 §十八 acceptance §5: ``CLAWCODEX_FREEZE_DIAG=1`` flips the
# watchdog on from an existing run without code changes.
DIAG_ENV_VAR = "CLAWCODEX_FREEZE_DIAG"

# F-108 §十八 design decision #5: ``0`` disables the Layer-2 budget
# without removing the watch — a fast-recovery escape hatch for users
# who hit a known-false-positive.
ENV_VAR_FOR: dict[str, str] = {
    "agent_loop_timeout_s": "CLAWCODEX_AGENT_LOOP_TIMEOUT",
    "turn_timeout_s": "CLAWCODEX_TURN_TIMEOUT",
    "tool_timeout_s": "CLAWCODEX_TOOL_TIMEOUT",
    "permission_timeout_s": "CLAWCODEX_PERMISSION_TIMEOUT",
    "threshold_s": "CLAWCODEX_FREEZE_THRESHOLD",
}


def env_var_for(field_name: str) -> str | None:
    """Return the env-var name mapped to ``field_name``, or None if unknown.

    Used by the CLI ``diag viewer`` to surface which env var
    overrides which knob. Returns ``None`` for ``dump_dir`` (no env
    var — managed via settings.json only).
    """
    return ENV_VAR_FOR.get(field_name)


@dataclass(frozen=True)
class FreezeSettings:
    """Frozen view of the F-108 freeze knobs.

    Kept distinct from :class:`clawcodex_ext.settings.types.FreezeSettings`
    so we don't import the full settings module in cold paths.
    ``resolve_freeze_settings()`` merges both: any field populated by
    the persisted ``SettingsSchema`` wins over the env var, which wins
    over the dataclass default.
    """

    agent_loop_timeout_s: float = 600.0
    turn_timeout_s: float = 300.0
    tool_timeout_s: float = 120.0
    permission_timeout_s: float = 30.0
    threshold_s: float = 60.0
    dump_dir: str | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "agent_loop_timeout_s": self.agent_loop_timeout_s,
            "turn_timeout_s": self.turn_timeout_s,
            "tool_timeout_s": self.tool_timeout_s,
            "permission_timeout_s": self.permission_timeout_s,
            "threshold_s": self.threshold_s,
            "dump_dir": self.dump_dir,
        }


# ``resolve_freeze_settings()`` returns this singleton when no settings
# file is reachable and no env var is set. Tests pin against the
# defaults via ``DEFAULT_FREEZE_SETTINGS`` so a regression that
# accidentally widens a budget surfaces as a snapshot diff.
DEFAULT_FREEZE_SETTINGS = FreezeSettings()


def _coerce_env(name: str, raw: str) -> float | None:
    """Parse a non-negative float env var, returning None on garbage."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return v


def resolve_freeze_settings(
    *,
    settings_factory: Callable[[], object | None] | None = None,
) -> FreezeSettings:
    """Resolve the layer-1/2 budgets honouring settings → env → default.

    Resolution order per knob:

    1. Settings file (if ``settings_factory`` returns an object with a
       ``freeze`` dataclass — the F-108 P108-E block we added to
       :class:`SettingsSchema`).
    2. Env var (from :data:`ENV_VAR_FOR`).
    3. :data:`DEFAULT_FREEZE_SETTINGS`.

    ``settings_factory`` is a callable (not the settings object) so the
    cold-path call doesn't import the whole settings module. The
    watchdog thread passes ``None`` to fall back to env+default; the
    CLI ``diag viewer`` passes ``get_settings`` so the JSON dump
    reflects the persisted config.

    Zero values are accepted (Layer-2 budget disabled); negative or
    unparseable values silently fall through to the next layer so a
    mis-set env var never widens a budget.
    """
    persisted = _safe_settings_freeze(settings_factory)
    # Make a mutable copy so we can layer in env overrides without
    # mutating the dataclass default.
    out = FreezeSettings(
        agent_loop_timeout_s=_pick(
            persisted.get("agent_loop_timeout_s"),
            _env("agent_loop_timeout_s"),
            DEFAULT_FREEZE_SETTINGS.agent_loop_timeout_s,
        ),
        turn_timeout_s=_pick(
            persisted.get("turn_timeout_s"),
            _env("turn_timeout_s"),
            DEFAULT_FREEZE_SETTINGS.turn_timeout_s,
        ),
        tool_timeout_s=_pick(
            persisted.get("tool_timeout_s"),
            _env("tool_timeout_s"),
            DEFAULT_FREEZE_SETTINGS.tool_timeout_s,
        ),
        permission_timeout_s=_pick(
            persisted.get("permission_timeout_s"),
            _env("permission_timeout_s"),
            DEFAULT_FREEZE_SETTINGS.permission_timeout_s,
        ),
        threshold_s=_pick(
            persisted.get("threshold_s"),
            _env("threshold_s"),
            DEFAULT_FREEZE_SETTINGS.threshold_s,
        ),
        dump_dir=pick_string(
            persisted.get("dump_dir"),
            os.environ.get("CLAWCODEX_FREEZE_DUMP_DIR"),
            DEFAULT_FREEZE_SETTINGS.dump_dir,
        ),
    )
    return out


def _env(field_name: str) -> float | None:
    raw_name = ENV_VAR_FOR.get(field_name)
    if raw_name is None:
        return None
    raw = os.environ.get(raw_name)
    if raw is None or raw.strip() == "":
        return None
    return _coerce_env(raw_name, raw)


def _pick(
    persisted: object | None,
    env_value: float | None,
    default: float,
) -> float:
    if isinstance(persisted, (int, float)) and persisted >= 0:
        return float(persisted)
    if env_value is not None:
        return env_value
    return default


def pick_string(persisted: object, env_value: str | None, default: str | None) -> str | None:
    if isinstance(persisted, str) and persisted:
        return persisted
    if isinstance(env_value, str) and env_value:
        return env_value
    return default


def _safe_settings_freeze(
    factory: Callable[[], object | None] | None,
) -> dict[str, object]:
    """Pull the ``freeze`` block out of the persisted settings.

    Returns ``{}`` (not None) when the settings module is unavailable
    so :func:`resolve_freeze_settings` doesn't have to special-case.
    Type-narrowed here to keep :func:`_pick` simple.
    """
    if factory is None:
        return {}
    try:
        obj = factory()
    except Exception:
        return {}
    if obj is None:
        return {}
    freeze = getattr(obj, "freeze", None)
    if freeze is None:
        return {}
    out: dict[str, object] = {}
    for name in (
        "agent_loop_timeout_s",
        "turn_timeout_s",
        "tool_timeout_s",
        "permission_timeout_s",
        "threshold_s",
        "dump_dir",
    ):
        v = getattr(freeze, name, None)
        if v is not None:
            out[name] = v
    return out


def dump_path(*, dump_dir: str | Path | None) -> Path:
    """Resolve the freeze-dump directory, creating it if missing.

    Defaults to ``$TMPDIR/clawcodex-freeze`` (or platform equivalent
    via :func:`tempfile.gettempdir`) when neither ``dump_dir`` nor the
    ``CLAWCODEX_FREEZE_DUMP_DIR`` env var is provided. Tests should
    always pass ``tmp_path`` to keep the workspace clean.
    """
    import tempfile

    if dump_dir is None:
        dump_dir = os.environ.get("CLAWCODEX_FREEZE_DUMP_DIR")
    if dump_dir is None or (isinstance(dump_dir, str) and not dump_dir.strip()):
        dump_dir = Path(tempfile.gettempdir()) / "clawcodex-freeze"
    p = Path(dump_dir).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Disk-full / read-only mount — fall back to a per-pid scratch
        # under the temp dir so the watchdog still produces output.
        p = Path(tempfile.gettempdir()) / f"clawcodex-freeze-{os.getpid()}"
        p.mkdir(parents=True, exist_ok=True)
    return p
