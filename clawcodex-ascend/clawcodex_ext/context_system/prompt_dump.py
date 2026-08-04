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

"""Prompt dump / observability interface (P119-C).

Exposes the effective system prompt as structured data so the self-iteration
framework can (1) compute prompt drift diffs, (2) write regression baselines,
and (3) feed evaluators — without scraping string output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from clawcodex_ext.context_system.cache_boundary import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
from clawcodex_ext.context_system.section_registry import get_section_order


@dataclass
class SectionSnapshot:
    """A structured snapshot of one block in the assembled system prompt.

    Each snapshot corresponds to one block returned by
    ``build_full_system_prompt_blocks()``.  The ``content`` field is
    empty unless ``include_content=True`` was passed to
    :func:`dump_effective_system_prompt`.
    """

    id: str
    """Section identifier (e.g. ``"intro"``, ``"tool_docs"``) or
    ``"__boundary__"`` for the dynamic-boundary marker."""

    order: float
    """Canonical sort order of this section (0-90 for known sections,
    block index for boundary/appended)."""

    cache_scope: str
    """Cache scope: ``"global"``, ``"session"``, or ``"request"``."""

    byte_len: int
    """UTF-8 byte length of the section content."""

    sha256: str
    """SHA-256 hex digest of the section content (byte-stable)."""

    content: str = ""
    """Section content text.  Empty unless ``include_content=True``."""

    source: Literal["default", "boundary", "appended"] = "default"
    """Provenance: built by a default section builder, the dynamic boundary
    marker, or the appended SDK prompt."""


# Keep the public ``format=`` keyword for observability API compatibility.
def dump_effective_system_prompt(  # pylint: disable=redefined-builtin
    format: Literal["blocks", "str", "structured"] = "structured",
    *,
    append_system_prompt: str | None = None,
    include_content: bool = False,
    **extra: Any,
) -> list[SectionSnapshot] | list[dict[str, Any]] | str:
    """Build the effective system prompt and return it in the requested *format*.

    Parameters
    ----------
    format:
        - ``"structured"`` → ``list[SectionSnapshot]`` (default)
        - ``"blocks"`` → raw block list from ``build_full_system_prompt_blocks``
        - ``"str"`` → concatenated plain-text prompt
    append_system_prompt:
        Appended SDK prompt text.  Also used to classify the trailing
        block as ``source="appended"`` in structured mode.
    include_content:
        If ``True``, populate ``SectionSnapshot.content`` with the full
        section text.  Defaults to ``False`` for safety — prevent
        accidental prompt leakage into logs.
    **extra:
        Additional keyword arguments forwarded to
        ``build_full_system_prompt_blocks``
        (e.g. ``cwd``, ``tools``, ``skills``, ``provider``, ``query_source``).

    Returns
    -------
    Depending on *format*:
    - ``list[SectionSnapshot]`` (structured)
    - ``list[dict]`` (blocks — raw output of ``build_full_system_prompt_blocks``)
    - ``str`` (plain-text prompt)
    """
    from clawcodex_ext.context_system.prompt_assembly import (
        build_full_system_prompt_blocks,
    )

    blocks = build_full_system_prompt_blocks(
        append_system_prompt=append_system_prompt,
        **extra,
    )

    if format == "blocks":
        return blocks

    if format == "str":
        return "\n\n".join(b.get("text", "") for b in blocks)

    # format == "structured"
    snapshots: list[SectionSnapshot] = []
    for i, block in enumerate(blocks):
        text: str = block.get("text", "")
        section_id: str = block.get("_section_id", "")

        # Determine source and id
        if text == SYSTEM_PROMPT_DYNAMIC_BOUNDARY:
            source: Literal["default", "boundary", "appended"] = "boundary"
            sid = "__boundary__"
            order = float(i)
        elif i == len(blocks) - 1 and append_system_prompt and not section_id:
            source = "appended"
            sid = "__appended__"
            order = float(i)
        else:
            source = "default"
            sid = section_id if section_id else f"block_{i}"
            order = get_section_order(sid) if section_id else float(i)

        # Determine cache scope
        cache_scope = block.get("_cache_scope", "")
        if not cache_scope:
            cc = block.get("cache_control", {})
            if isinstance(cc, dict) and cc.get("scope") == "global":
                cache_scope = "global"
            else:
                cache_scope = "session"

        snapshots.append(
            SectionSnapshot(
                id=sid,
                order=order,
                cache_scope=cache_scope,
                byte_len=len(text.encode("utf-8")),
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                content=text if include_content else "",
                source=source,
            )
        )

    return snapshots
