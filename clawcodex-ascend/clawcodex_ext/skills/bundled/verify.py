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

"""Bundled ``/verify`` skill for evidence-backed implementation verification."""

from __future__ import annotations

from ..bundled_skills import BundledSkillDefinition, register_bundled_skill


VERIFY_PROMPT = """# Verification Assignment

Independently verify the implementation requested in the current conversation. Use
the inherited conversation as the contract and the current working tree as evidence.

Inspect the change, run the applicable project checks, exercise the public behavior,
and perform at least one relevant adversarial probe. Read `examples/cli.md` or
`examples/server.md` when it matches the target.

When the user request supplies exact commands or file paths, run those commands first
from the project workspace and report their output before doing broader inspection.
Do not spend verification turns rediscovering the workspace or repeating equivalent
Git/status scans that do not advance the requested proof.
If those explicit checks cover the requested contract and succeed, report the verdict
immediately instead of starting optional inspection afterward. Always reserve a final
text response for the evidence summary and required verdict line; never end on a tool
call or tool result.

Return concrete commands, observed output, expected-versus-actual comparisons, and
PASS/FAIL per check. Distinguish inspection from runtime proof and finish with exactly
one line: `VERDICT: PASS`, `VERDICT: FAIL`, or `VERDICT: PARTIAL`.

Do not modify the project.
"""


CLI_EXAMPLE = """# CLI verification example

Verify a CLI through the same entry point users run, not by importing only an internal
helper.

1. Run `--help` and confirm the command, options, defaults, and examples are accurate.
2. Run one representative successful invocation and capture stdout, stderr, and the
   process exit code.
3. Run malformed or missing input and confirm a non-zero exit plus actionable error.
4. Try a boundary relevant to the parser: empty text, Unicode, zero, negative value,
   a long value, or repeated flags.
5. If the command persists state, run it twice and inspect the resulting state for
   idempotency or duplication.

Prefer the repository's documented executable (for example its venv-installed entry
point) so packaging and command registration are covered by the check.
"""


SERVER_EXAMPLE = """# Server verification example

Verify a server by starting it and making real requests against a disposable/local
instance.

1. Start the documented development server and retain its startup output.
2. Call a representative endpoint and validate the response body/schema, not only the
   HTTP status.
3. Call an error path with malformed input or a missing identifier and verify status,
   body, and absence of an internal crash.
4. Repeat a mutating request or issue parallel requests when idempotency/concurrency is
   relevant; inspect the resulting state for duplicates or lost writes.
5. Check logs and shut the server down cleanly. Do not point destructive probes at a
   shared or production environment.

Use a unique temporary resource name and clean it up when the public API supports safe
cleanup.
"""


def _build_verify_prompt(args: str, context: object | None = None) -> str:
    workspace_root = getattr(context, "workspace_root", None)
    cwd = getattr(context, "cwd", None) or workspace_root
    runtime_context = """## Runtime Location

The `Base directory for this skill` shown above contains reference material only. It
is not the project under verification. Run repository inspection, Git commands,
tests, and public behavior checks from the caller's project workspace."""
    if workspace_root is not None:
        runtime_context += f"\n\nProject workspace root: `{workspace_root}`."
    if cwd is not None and cwd != workspace_root:
        runtime_context += f"\nCaller working directory: `{cwd}`."

    parts = [VERIFY_PROMPT, runtime_context]
    if args:
        parts.append(f"## User Request\n\n{args}")
    return "\n\n".join(parts)


def register_verify_skill() -> bool:
    return register_bundled_skill(
        BundledSkillDefinition(
            name="verify",
            description="Verify a code change does what it should by running the app.",
            when_to_use=(
                "Use after implementation work to prove the requested behavior through "
                "real execution, required project checks, and an adversarial probe."
            ),
            user_invocable=True,
            context="fork",
            agent="verification",
            files={
                "examples/cli.md": CLI_EXAMPLE,
                "examples/server.md": SERVER_EXAMPLE,
            },
            get_prompt_for_command=_build_verify_prompt,
        )
    )


__all__ = [
    "CLI_EXAMPLE",
    "SERVER_EXAMPLE",
    "VERIFY_PROMPT",
    "register_verify_skill",
]
