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

"""Consolidation prompt builder — F-100.

Mirrors ``typescript/src/services/autoDream/consolidationPrompt.ts`` +
the prefix used by ``dream.ts`` (the manual /dream skill).

The prompt is a 4-phase structured reflection:

1. **Orient** — ``ls`` the memory dir, read ``MEMORY.md`` index, skim
   existing topic files.
2. **Gather** — pull signal from daily logs, drifted memories, narrow
   transcript greps.
3. **Consolidate** — merge new signal into existing topic files,
   convert relative dates to absolute, delete contradicted facts.
4. **Prune + index** — keep ``MEMORY.md`` under the line + byte caps
   as a one-line-per-entry index.

Reuses :data:`src.memdir.memdir.DIR_EXISTS_GUIDANCE`,
:data:`src.memdir.memdir.ENTRYPOINT_NAME`, and
:data:`src.memdir.memdir.MAX_ENTRYPOINT_LINES` constants so the prompt
stays in lock-step with the upstream memory module's caps.
"""

from __future__ import annotations

from clawcodex_ext.memdir.memdir import (
    DIR_EXISTS_GUIDANCE,
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_LINES,
)

__all__ = [
    "DREAM_PROMPT_PREFIX",
    "build_consolidation_prompt",
]


# Used by the manual /dream skill (Phase C / 100.4). Extracted so the
# autoDream service and the manual skill share the same prefix
# without one importing the other.
DREAM_PROMPT_PREFIX = (
    "# Dream: Memory Consolidation (manual run)\n\n"
    "You are performing a manual dream — a reflective pass over your "
    "memory files. Unlike the automatic background dream, this run has "
    "full tool permissions and the user is watching. Synthesize what "
    "you've learned recently into durable, well-organized memories so "
    "that future sessions can orient quickly.\n\n"
)


def build_consolidation_prompt(
    memory_root: str,
    transcript_dir: str,
    extra: str = "",
    *,
    manual: bool = False,
) -> str:
    """Build the 4-phase consolidation prompt.

    Args:
        memory_root: Absolute path to the auto-memory directory.
        transcript_dir: Absolute path to the per-cwd session
            transcript directory.
        extra: Optional free-form context appended under "Additional
            context". Used by the auto-dream service to inject the
            tool-constraints block + session list, and by the manual
            /dream skill to inject user-supplied context.
        manual: When True, prepend :data:`DREAM_PROMPT_PREFIX`. The
            auto service passes ``False`` (it injects its own
            session-list block).
    """
    base = _build_base_prompt(memory_root, transcript_dir)
    if manual:
        base = DREAM_PROMPT_PREFIX + base
    if extra:
        base = base + f"\n\n## Additional context\n\n{extra}"
    return base


def _build_base_prompt(memory_root: str, transcript_dir: str) -> str:
    return f"""# Dream: Memory Consolidation

You are performing a dream — a reflective pass over your memory files. Synthesize what you've learned recently into durable, well-organized memories so that future sessions can orient quickly.

Memory directory: `{memory_root}`
{DIR_EXISTS_GUIDANCE}

Session transcripts: `{transcript_dir}` (large JSONL files — grep narrowly, don't read whole files)

---

## Phase 1 — Orient

- `ls` the memory directory to see what already exists
- Read `{ENTRYPOINT_NAME}` to understand the current index
- Skim existing topic files so you improve them rather than creating duplicates
- If `logs/` or `sessions/` subdirectories exist (assistant-mode layout), review recent entries there

## Phase 2 — Gather recent signal

Look for new information worth persisting. Sources in rough priority order:

1. **Daily logs** (`logs/YYYY/MM/YYYY-MM-DD.md`) if present — these are the append-only stream
2. **Existing memories that drifted** — facts that contradict something you see in the codebase now
3. **Transcript search** — if you need specific context (e.g., "what was the error message from yesterday's build failure?"), grep the JSONL transcripts for narrow terms:
   `grep -rn "<narrow term>" {transcript_dir}/ --include="*.jsonl" | tail -50`

Don't exhaustively read transcripts. Look only for things you already suspect matter.

## Phase 3 — Consolidate

For each thing worth remembering, write or update a memory file at the top level of the memory directory. Use the memory file format and type conventions from your system prompt's auto-memory section — it's the source of truth for what to save, how to structure it, and what NOT to save.

Focus on:
- Merging new signal into existing topic files rather than creating near-duplicates
- Converting relative dates ("yesterday", "last week") to absolute dates so they remain interpretable after time passes
- Deleting contradicted facts — if today's investigation disproves an old memory, fix it at the source

## Phase 4 — Prune and index

Update `{ENTRYPOINT_NAME}` so it stays under {MAX_ENTRYPOINT_LINES} lines AND under ~25KB. It's an **index**, not a dump — each entry should be one line under ~150 characters: `- [Title](file.md) — one-line hook`. Never write memory content directly into it.

- Remove pointers to memories that are now stale, wrong, or superseded
- Demote verbose entries: if an index line is over ~200 chars, it's carrying content that belongs in the topic file — shorten the line, move the detail
- Add pointers to newly important memories
- Resolve contradictions — if two files disagree, fix the wrong one

---

Return a brief summary of what you consolidated, updated, or pruned. If nothing changed (memories are already tight), say so."""
