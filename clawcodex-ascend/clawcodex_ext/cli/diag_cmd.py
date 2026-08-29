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

# pylint: disable=consider-using-in
"""``clawcodex-dev diag`` CLI subcommand.

Layer 4 of the freeze detection plan. Subcommands:

* ``clawcodex-dev diag freeze-report [--last N] [--dump-dir PATH]``
  — show the most recent N freeze dumps.
* ``clawcodex-dev diag viewer [--last N] [--dump-dir PATH]``
  — alias of ``freeze-report`` (the plan calls it ``viewer``).
* ``clawcodex-dev diag status``
  — show resolved freeze settings + watchdog enable state.

Output is plain text (no Rich markup) so the command stays
scriptable. Diagnostic payloads are JSON so a downstream tool can
parse them deterministically.

Layer 0–3 do the actual detection / recovery; this command only
exposes them to humans / postmortem scripts. It is gated by the
``clawcodex_ext.diagnostics`` package; nothing here mutates the
canonical query loop.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from clawcodex_ext.cli.subcommand_registry import register
from clawcodex_ext.diagnostics import (
    DEFAULT_FREEZE_CHECK_INTERVAL_S,
    DEFAULT_FREEZE_DIAG_ENV,
    DEFAULT_FREEZE_SETTINGS,
    FreezeDetector,
    dump_path,
    env_var_for,
    resolve_freeze_settings,
)


_USAGE = (
    "usage: clawcodex-dev diag <freeze-report|viewer|status> [options]\n\n"
    "Subcommands:\n"
    "  freeze-report [--last N] [--dump-dir PATH] [--json]\n"
    "                         Print the most recent N freeze dumps found in\n"
    "                         the dump dir (default 1). With --json prints\n"
    "                         the raw dump payload.\n"
    "  viewer        [--last N] [--dump-dir PATH] [--json]\n"
    "                         Alias of freeze-report.\n"
    "  status                  Show resolved freeze settings + watchdog state.\n"
    "                         With --json prints the resolver output.\n"
    "\n"
    "Options:\n"
    "  --last N                Number of dumps to show (default 1; 0 = all).\n"
    "  --dump-dir PATH         Override $CLAWCODEX_FREEZE_DUMP_DIR / settings.\n"
    "  --json                  Emit JSON output instead of human text.\n"
    "  --resolved-from settings|env|default\n"
    "                         Restrict the status report to one resolution layer.\n"
)


# ----------------------------------------------------------------------
# Public CLI entry — wired into :mod:`subcommand_registry` below.
# ----------------------------------------------------------------------


def run_diag_command(argv: list[str]) -> int:
    """Dispatch the ``diag`` subcommand. Returns exit code."""
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_USAGE)
        return 0
    sub = argv[0]
    rest = argv[1:]
    if sub == "freeze-report" or sub == "viewer":
        return _run_freeze_report(rest)
    if sub == "status":
        return _run_status(rest)
    sys.stderr.write(f"unknown diag subcommand: {sub}\n\n{_USAGE}")
    return 2


# ----------------------------------------------------------------------
# Subcommand implementations.
# ----------------------------------------------------------------------


def _parse_common(argv: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Parse ``--last``, ``--dump-dir``, ``--json`` from ``argv``.

    Returns (options, residual). Unknown flags fall through to
    ``residual`` so a future addition can detect them without
    forking the parser.
    """
    opts: dict[str, Any] = {"last": 1, "dump_dir": None, "json": False}
    residual: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--last" and i + 1 < len(argv):  # nosec B105
            try:
                opts["last"] = int(argv[i + 1])
            except ValueError:
                opts["last"] = 1
            i += 2
            continue
        if token == "--dump-dir" and i + 1 < len(argv):  # nosec B105
            opts["dump_dir"] = argv[i + 1]
            i += 2
            continue
        if token == "--json":  # nosec B105
            opts["json"] = True
            i += 1
            continue
        residual.append(token)
        i += 1
    return opts, residual


def _run_freeze_report(argv: list[str]) -> int:
    opts, residual = _parse_common(argv)
    if residual:
        sys.stderr.write(f"unrecognised options: {' '.join(residual)}\n\n{_USAGE}")
        return 2
    last_n: int = opts["last"] if isinstance(opts["last"], int) else 1
    target_dir = _resolve_dump_dir(opts["dump_dir"])
    if not target_dir.exists():
        sys.stderr.write(f"no freeze dumps found (dir does not exist: {target_dir})\n")
        return 0
    dumps = sorted(target_dir.glob("freeze-*.json"), key=_dump_sort_key)
    if not dumps:
        sys.stderr.write(f"no freeze dumps in {target_dir}\n")
        return 0
    if last_n > 0:
        dumps = dumps[-last_n:]
    if opts["json"]:
        # ``print`` here so consumers can pipe; ``dumps`` is the
        # list of dicts the postmortem tool will parse.
        out = []
        for path in dumps:
            try:
                out.append(json.loads(path.read_text()))
            except Exception as exc:
                out.append({"file": str(path), "error": f"unreadable: {exc}"})
        json.dump(out, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0
    sys.stdout.write(f"Freeze dumps in {target_dir}:\n")
    for path in dumps:
        _print_dump_text(path)
    return 0


def _run_status(argv: list[str]) -> int:
    opts, residual = _parse_common(argv)
    if residual:
        sys.stderr.write(f"unrecognised options: {' '.join(residual)}\n\n{_USAGE}")
        return 2
    settings = resolve_freeze_settings()
    diag_env = bool(os.environ.get(DEFAULT_FREEZE_DIAG_ENV, "").strip())
    detector_alive = False
    inst = FreezeDetector._INSTANCE  # noqa: SLF001 — internal but stable
    detector_alive = bool(inst and inst._watchdog is not None and inst._watchdog.is_alive())  # noqa: SLF001
    if opts["json"]:
        payload = {
            "diag_env_var": DEFAULT_FREEZE_DIAG_ENV,
            "diag_env_enabled": diag_env,
            "check_interval_s": DEFAULT_FREEZE_CHECK_INTERVAL_S,
            "settings": settings.as_dict(),
            "env_var_map": {
                k: v
                for k, v in (
                    ("agent_loop_timeout_s", env_var_for("agent_loop_timeout_s")),
                    ("turn_timeout_s", env_var_for("turn_timeout_s")),
                    ("tool_timeout_s", env_var_for("tool_timeout_s")),
                    ("permission_timeout_s", env_var_for("permission_timeout_s")),
                    ("threshold_s", env_var_for("threshold_s")),
                )
                if v
            },
            "detector_alive": detector_alive,
            "default_settings": DEFAULT_FREEZE_SETTINGS.as_dict(),
            "dump_dir": str(dump_path(dump_dir=settings.dump_dir)),
        }
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0
    sys.stdout.write("Freeze detector status\n")
    sys.stdout.write(f"  diag env var ({DEFAULT_FREEZE_DIAG_ENV}): {diag_env}\n")
    sys.stdout.write(f"  watchdog alive: {detector_alive}\n")
    sys.stdout.write("  resolved settings:\n")
    for name, value in settings.as_dict().items():
        sys.stdout.write(f"    {name}: {value} (env: {env_var_for(name)})\n")
    sys.stdout.write(f"  dump dir: {dump_path(dump_dir=settings.dump_dir)}\n")
    sys.stdout.write(f"  check interval (default): {DEFAULT_FREEZE_CHECK_INTERVAL_S}s\n")
    return 0


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------


def _resolve_dump_dir(override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    return dump_path(dump_dir=None)


def _dump_sort_key(path: Path) -> tuple[int, int, int]:
    """Sort key matching the filename scheme ``freeze-<pid>-<idx>-<ts>.json``.

    Order is (ts, idx, pid) so a multi-detector process buckets by
    time first, then by index. The returned tuple is naturally
    ordered chronologically.
    """
    parts = path.stem.split("-")
    try:
        pid = int(parts[1]) if len(parts) > 1 else 0
        idx = int(parts[2]) if len(parts) > 2 else 0
        ts = int(parts[3]) if len(parts) > 3 else 0
    except (ValueError, IndexError):
        pid, idx, ts = 0, 0, 0
    return (ts, idx, pid)


def _print_dump_text(path: Path) -> None:
    """Render one freeze dump in human-readable form to stdout."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        sys.stdout.write(f"  - {path.name}: <unreadable: {exc}>\n")
        return
    detected_at = data.get("detected_at_unix", 0)
    elapsed = data.get("elapsed_seconds", 0.0)
    threshold = data.get("threshold_seconds", 0.0)
    threads = data.get("thread_stacks") or []
    extra = data.get("extra") or {}
    tripped = extra.get("tripped_count", 0)
    iso = _dt.datetime.fromtimestamp(detected_at, tz=_dt.timezone.utc).isoformat() if detected_at else "<unknown>"
    sys.stdout.write(f"  - {path.name}\n")
    sys.stdout.write(f"      detected_at: {iso}\n")
    sys.stdout.write(f"      elapsed:    {elapsed:.1f}s (threshold {threshold:.1f}s)\n")
    sys.stdout.write(f"      tripped:    {tripped}\n")
    sys.stdout.write(f"      threads:    {len(threads)}\n")
    for frame in threads[:5]:
        sys.stdout.write(f"        • tid={frame.get('tid')} name={frame.get('thread_name', '?')!r}\n")
    if len(threads) > 5:
        sys.stdout.write(f"        … ({len(threads) - 5} more threads)\n")


# ----------------------------------------------------------------------
# Subcommand registration.
# ----------------------------------------------------------------------


@register("diag")
def _diag_subcommand(argv: list[str]) -> int:
    """``clawcodex-dev diag <sub>`` fast-path entry."""
    return run_diag_command(argv)


__all__ = ["run_diag_command"]
