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

"""Scope-aware memory prompt builder — F-13 memory scope isolation.

This module sits entirely in ``clawcodex_ext/`` and relies on the existing
``src.memdir.load_memory_prompts()`` interface. No changes to ``src/memdir/``
are required.

Integration is handled via the memory-section builder registry defined in
``src/context_system/prompt_assembly.py``.  Calling
:func:`install_memory_extension` (done automatically by
``clawcodex_ext.__init__``) registers a builder that produces scope-aware
memory sections.  No ``memory_scopes`` parameter is needed in the upstream
``build_full_system_prompt`` / ``build_full_system_prompt_blocks`` signatures.
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# Mirror of src/agent/parse_agent_markdown.VALID_MEMORY_SCOPES.
# Kept here so the ext module does not import from src for validation.
VALID_MEMORY_SCOPES: frozenset[str] = frozenset({"user", "project", "local"})

# Default scopes used when building scope-aware memory.  Can be overridden
# by calling ``set_default_memory_scopes()`` before the first prompt build.
_default_memory_scopes: list[str] = ["user", "project", "local"]


def _validate_scopes(memory_scopes: Iterable[str]) -> list[str]:
    """Filter ``memory_scopes`` to known-valid values.

    Logs a warning for unknown scopes so the user can fix agent definitions,
    but does **not** raise — unrecognised scopes degrade gracefully.
    """
    validated: list[str] = []
    for scope in memory_scopes:
        s = str(scope).strip()
        if s in VALID_MEMORY_SCOPES:
            validated.append(s)
        else:
            logger.warning(
                "Unknown memory scope %r — must be one of %s; skipping",
                s,
                ", ".join(sorted(VALID_MEMORY_SCOPES)),
            )
    return validated


def set_default_memory_scopes(scopes: list[str]) -> None:
    """Override the default memory scopes used by the registered builder.

    Call this before the first prompt build (e.g. at app startup) to
    customise which scopes are included.  The default is
    ``["user", "project", "local"]``.
    """
    global _default_memory_scopes
    _default_memory_scopes = list(scopes)


def build_scope_aware_memory_prompt(
    memory_scopes: list[str] | None = None,
) -> str | None:
    """Build a combined memory prompt for the given memory scopes.

    Args:
        memory_scopes:  List of scope names (``"user"``, ``"project"``,
                        ``"local"``).  ``None`` or an empty list returns
                        ``None`` (no-op / no-op path).

    Returns:
        A single concatenated memory-prompt string, or ``None`` if
        auto-memory is disabled, no scopes were given, or no prompt could
        be loaded for any scope.
    """
    if not memory_scopes:
        return None

    validated = _validate_scopes(memory_scopes)
    if not validated:
        return None

    # Lazy import: ``src.memdir`` may not be importable in all contexts
    # (e.g. CLI subcommands that don't need the full agent loop).
    try:
        from clawcodex_ext.memdir.memdir import load_memory_prompts
    except ImportError:
        logger.debug("src.memdir not available — skipping scope-aware memory")
        return None

    try:
        prompts = load_memory_prompts(memory_scopes=validated)
    except Exception:
        logger.exception("Failed to load scope-aware memory prompts")
        return None

    if not prompts:
        return None

    # Join multiple scope prompts into one block, separated by a blank line.
    # Each individual prompt already has its own heading and structure.
    combined = "\n\n".join(prompts)
    return combined


def install_memory_extension() -> None:
    """Register the scope-aware memory builder with the prompt-assembly registry.

    Idempotent — safe to call more than once.
    """
    from clawcodex_ext.context_system.section_registry import register_section

    def _wrapped(_ctx: dict) -> str | None:
        prompt = build_scope_aware_memory_prompt(_default_memory_scopes)
        return prompt  # str | None — builder contract is (runtime_ctx) -> str | None

    register_section("memory", builder=_wrapped)
