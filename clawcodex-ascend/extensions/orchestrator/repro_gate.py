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

"""Repro-first gate — reproduce the bug before you may fix it.

Capability extension #1 (the keystone): a fix pipeline is only as
trustworthy as its ability to *observe* the defect. This module adds a
reproduction stage in front of the normal fix run:

1. A dedicated agent pass turns the issue description into an
   executable check: it writes reproduction assets under
   ``.orchestrator_control/repro/`` and puts a single shell command into
   ``repro_command.txt``. Contract: the command **exits non-zero while
   the bug exists and zero once it is fixed** (i.e. it is a failing
   test).
2. The orchestrator runs that command itself. Only a demonstrated
   failure opens the gate to the fix stage; the command is then re-run
   during pre-push verification and must have turned green.
3. When the agent cannot reproduce (or produced nothing executable),
   the gate closes: the issue is marked failed with a
   "cannot reproduce" report posted back to the tracker — instead of an
   unverifiable "fix" MR.

Everything lives under ``.orchestrator_control/`` so the existing
artifact guards keep reproduction scaffolding out of commits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPRO_DIR = ".orchestrator_control/repro"
REPRO_COMMAND_FILE = f"{REPRO_DIR}/repro_command.txt"
NOT_REPRODUCIBLE_FILE = f"{REPRO_DIR}/not_reproducible.json"

_OUTPUT_TAIL_CHARS = 3_000


@dataclass(frozen=True)
class ReproGateResult:
    """Outcome of evaluating the reproduction contract."""

    verdict: str  # "reproduced" | "not_reproducible" | "not_demonstrated" | "missing"
    command: str | None = None
    output: str = ""
    payload: dict[str, Any] | None = None  # not_reproducible.json contents

    @property
    def proceed(self) -> bool:
        return self.verdict == "reproduced"


def build_repro_prompt(issue: Any) -> str:
    """Prompt for the reproduction stage.

    Deliberately narrow: the agent's ONLY deliverable is an executable
    judgment of the described behavior — no source modifications.
    """
    identifier = getattr(issue, "identifier", None) or getattr(issue, "id", "")
    title = getattr(issue, "title", "") or ""
    description = getattr(issue, "description", "") or ""
    return f"""You are the REPRODUCTION stage for issue {identifier}.

Issue: {title}

{description}

Your ONLY task is to turn the described problem into a repeatable,
executable check. Do NOT fix anything. Do NOT modify existing source
files.

1. Explore the repository and reproduce the described behavior.
2. Create your reproduction assets under `{REPRO_DIR}/` (e.g. a
   `repro_test.py` exercising the buggy behavior).
3. Write ONE shell command (a single line, run from the repository
   root) into `{REPRO_COMMAND_FILE}`. The command MUST:
   - exit with a NON-ZERO code right now (it demonstrates the bug), and
   - exit 0 once the bug is properly fixed.
   A failing test is the ideal shape for this command.
4. Run the command yourself and confirm it currently exits non-zero.

If you cannot reproduce the problem after honest attempts (the
referenced code does not exist, the described behavior is actually
correct, the steps are not actionable), write
`{NOT_REPRODUCIBLE_FILE}` instead:

```json
{{"reason": "<one line>",
  "attempts": ["<what you tried>", "..."]}}
```

Then stop. Never invent a reproduction for a bug you could not
actually observe — an honest "cannot reproduce" is a valid, useful
outcome.
"""


async def _run_shell(command: str, cwd: str, timeout_ms: int) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_ms / 1000.0,
        )
    except asyncio.TimeoutError:
        return 124, f"repro command timed out after {timeout_ms}ms"
    output = "\n".join(part.decode("utf-8", errors="replace").strip() for part in (stdout, stderr) if part).strip()
    if len(output) > _OUTPUT_TAIL_CHARS:
        output = f"…(truncated)…\n{output[-_OUTPUT_TAIL_CHARS:]}"
    return proc.returncode or 0, output


async def evaluate_repro_gate(workspace_root: Path | str, timeout_ms: int = 300_000) -> ReproGateResult:
    """Check the reproduction contract the agent left in the workspace.

    Verdicts:

    - ``reproduced`` — ``repro_command.txt`` exists and the command
      exits non-zero: the bug is demonstrated, the fix stage may run.
    - ``not_demonstrated`` — the command exits 0: whatever the agent
      wrote does not actually show a failure.
    - ``not_reproducible`` — the agent explicitly reported it could not
      reproduce (``not_reproducible.json``).
    - ``missing`` — the agent produced neither artifact.
    """
    root = Path(workspace_root)

    marker = root / NOT_REPRODUCIBLE_FILE
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(payload, dict):
                payload = {"reason": str(payload)}
        except (OSError, json.JSONDecodeError):
            payload = {"reason": "not reproducible (unparseable report)"}
        return ReproGateResult(verdict="not_reproducible", payload=payload)

    command_file = root / REPRO_COMMAND_FILE
    if not command_file.is_file():
        return ReproGateResult(verdict="missing")
    try:
        command = command_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ReproGateResult(verdict="missing")
    if not command:
        return ReproGateResult(verdict="missing")
    # Take the first non-comment line: the contract says ONE command.
    command = next(
        (line.strip() for line in command.splitlines() if line.strip() and not line.strip().startswith("#")),
        "",
    )
    if not command:
        return ReproGateResult(verdict="missing")

    rc, output = await _run_shell(command, str(root), timeout_ms)
    if rc == 0:
        return ReproGateResult(verdict="not_demonstrated", command=command, output=output)
    return ReproGateResult(verdict="reproduced", command=command, output=output)


def append_repro_hint(workspace_root: Path | str, command: str) -> None:
    """Surface the established reproduction to the fix stage.

    ``.operator_hints.md`` is already prepended to every agent prompt
    (and excluded from commits), so appending here needs no extra
    prompt plumbing.
    """
    hints = Path(workspace_root) / ".operator_hints.md"
    section = (
        "\n\n## Reproduction established\n"
        f"A reproduction command is in place and currently FAILS:\n\n"
        f"```\n{command}\n```\n\n"
        "Run it from the repository root at any time. Your fix is only\n"
        "complete when this command exits 0; it is re-run automatically\n"
        "before push and will block the MR while it still fails.\n"
    )
    try:
        with open(hints, "a", encoding="utf-8") as f:
            f.write(section)
    except OSError:
        logger.warning("failed to write repro hint to %s", hints, exc_info=True)


def format_repro_gate_comment(issue: Any, result: ReproGateResult) -> str:
    """Tracker comment posted when the gate closes."""
    identifier = getattr(issue, "identifier", None) or getattr(issue, "id", "")
    lines = [f"## ClawCodex could not reproduce {identifier}".rstrip(), ""]
    if result.verdict == "not_reproducible":
        payload = result.payload or {}
        reason = str(payload.get("reason", "")).strip()
        if reason:
            lines += [f"**Reason**: {reason}", ""]
        attempts = payload.get("attempts")
        if isinstance(attempts, list) and attempts:
            lines += ["**Attempted**:"] + [f"- {item}" for item in attempts[:10]] + [""]
    elif result.verdict == "not_demonstrated":
        lines += [
            "The reproduction stage produced a check, but it exits 0 — the",
            "described failure could not be demonstrated:",
            "",
            f"```\n{result.command}\n{result.output}\n```".strip(),
            "",
        ]
    else:  # missing / timeout-shaped outcomes
        lines += [
            "The reproduction stage did not produce an executable check for",
            "the described behavior.",
            "",
        ]
    lines += [
        "_No fix was attempted and no merge request was opened: a fix for an"
        " unobserved bug cannot be verified. If the problem is real, please"
        " add concrete reproduction steps (exact commands, inputs, expected"
        " vs actual output) and re-trigger with the `agent:retry` label._",
    ]
    return "\n".join(lines)
