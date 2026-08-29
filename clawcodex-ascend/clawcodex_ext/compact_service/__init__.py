#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""
Compact service — boundary markers and command-facing compaction wrapper.

``compact_service.messages`` provides boundary-marker dataclasses and
factory functions used by the compaction pipeline (``services/compact/``).

``compact_service.service`` provides the ``compact_conversation()`` wrapper
that the ``/compact`` command handler expects: it accepts a ``Conversation``
object, delegates to the unified pipeline, and mutates the conversation in place.
"""

from __future__ import annotations

from .messages import (
    CompactBoundaryMetadata,
    PreservedSegment,
    annotate_boundary_with_preserved_segment,
    create_compact_boundary_message,
    create_compact_summary_message,
    get_messages_after_boundary,
    is_compact_boundary_message,
)

# NOTE: service.py is intentionally NOT imported here to avoid a circular
# import chain: compact_service.__init__ → service → services.compact.compact
# → compact_service.messages. Import it directly when needed:
#   from clawcodex_ext.compact_service.service import compact_conversation

__all__ = [
    "CompactBoundaryMetadata",
    "PreservedSegment",
    "annotate_boundary_with_preserved_segment",
    "create_compact_boundary_message",
    "create_compact_summary_message",
    "get_messages_after_boundary",
    "is_compact_boundary_message",
]
