#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""P65-A — Langfuse observability integration.

A plug-in :class:`AnalyticsSink` that translates :class:`AnalyticsEvent`
records into Langfuse traces / spans / generations. Implements CCB's
"Agent Loop level tracing" surface area (model, prompt, completion,
token usage, latency) on top of clawcodex's existing analytics
event stream.

Graceful degradation
--------------------
The ``langfuse`` SDK is **optional**. When it is not importable (or
when the env-var credentials are not set), :class:`LangfuseSink`
collapses to a :class:`NullSink` so the rest of the system runs
unaffected. The public API never raises on missing config — silent
no-op is the contract.

Configuration
-------------
Reads three environment variables at construction time:

* ``LANGFUSE_PUBLIC_KEY``  — required to actually flush traces
* ``LANGFUSE_SECRET_KEY``  — required to actually flush traces
* ``LANGFUSE_HOST``        — defaults to ``"https://cloud.langfuse.com"``

Either pass them via :func:`init_langfuse` or rely on the
constructor picking them up from ``os.environ``.

Mapping contract
----------------
``AnalyticsEvent.type`` decides which Langfuse primitive receives
the event:

* ``TURN_END``                → ``trace.generation(...)`` (LLM call)
* ``TOOL_USE``                → ``trace.span(...)``
* ``AGENT_SPAWN`` / ``AGENT_COMPLETE`` → ``trace.span(...)``
* everything else             → ``trace.event(...)``

Training data export
--------------------
:func:`export_training_data` reads the buffered trace map and
emits JSONL records consumable by SFT / DPO pipelines. The
exporter never depends on a live Langfuse server — it operates
on the sink's local buffer (the same data the sink already
forwarded to Langfuse, captured here for offline use).
"""

from __future__ import annotations

from .client import (
    LangfuseConfig,
    get_langfuse_client,
    init_langfuse,
    is_langfuse_available,
    reset_langfuse_client,
)
from .exporter import (
    TrainingDataExporter,
    export_training_data,
)
from .sink import LangfuseSink

__all__ = [
    "LangfuseConfig",
    "LangfuseSink",
    "TrainingDataExporter",
    "export_training_data",
    "get_langfuse_client",
    "init_langfuse",
    "is_langfuse_available",
    "reset_langfuse_client",
]
