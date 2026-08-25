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

"""Pydantic models shared by the local session visualizer APIs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Forward references for OperationCategory.color / .label properties
# ---------------------------------------------------------------------------

# Populated after the enum is defined; see ``OperationCategory`` below.
_CATEGORY_COLORS: dict = {}
_CATEGORY_LABELS: dict = {}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BarType(str, Enum):
    """Type of operation represented by a timeline bar."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USER = "user"
    SYSTEM = "system"
    PHASE = "phase"
    TURN = "turn"
    SESSION = "session"
    WAIT = "wait"
    CUSTOM = "custom"


class BarStatus(str, Enum):
    """Execution status of a timeline bar."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    RUNNING = "running"
    PENDING = "pending"
    SKIPPED = "skipped"


class OperationCategory(str, Enum):
    """High-level operation category for the multi-session waterfall view.

    Eight buckets mapped 1:1 to the legend in the reference visualization:
    READ / EXECUTE / WRITE / ORCHESTRATE /
    LLM_TEXT / TURN / BACKGROUND / OTHER.

    The LLM_TEXT, TURN, and BACKGROUND buckets were added in 2026-06-11 to
    split the previous catch-all OTHER bucket, which had 46/55 operations in
    orchestrator sessions, into meaningful sub-types.
    """

    READ = "read"
    EXECUTE = "execute"
    WRITE = "write"
    ORCHESTRATE = "orchestrate"
    LLM_TEXT = "llm_text"
    TURN = "turn"
    BACKGROUND = "background"
    OTHER = "other"

    @property
    def color(self) -> str:
        """Canonical legend color, aligned with style.css dark theme."""
        return _CATEGORY_COLORS[self]

    @property
    def label(self) -> str:
        """Chinese display label for legend and pills."""
        return _CATEGORY_LABELS[self]


_CATEGORY_COLORS: dict[OperationCategory, str] = {
    OperationCategory.READ: "#3fb950",
    OperationCategory.EXECUTE: "#58a6ff",
    OperationCategory.WRITE: "#d29922",
    OperationCategory.ORCHESTRATE: "#f778ba",
    OperationCategory.LLM_TEXT: "#79c0ff",
    OperationCategory.TURN: "#bc8cff",
    OperationCategory.BACKGROUND: "#8b949e",
    OperationCategory.OTHER: "#6e7681",
}

_CATEGORY_LABELS: dict[OperationCategory, str] = {
    OperationCategory.READ: "读取",
    OperationCategory.EXECUTE: "执行",
    OperationCategory.WRITE: "写入",
    OperationCategory.ORCHESTRATE: "编排",
    OperationCategory.LLM_TEXT: "推理",
    OperationCategory.TURN: "轮次",
    OperationCategory.BACKGROUND: "后台",
    OperationCategory.OTHER: "其他",
}


class AnomalyType(str, Enum):
    """Category of detected anomaly."""

    NO_OP = "no_op"
    STAGNATION = "stagnation"
    LOOP = "loop"
    READ_ONLY_SPIRAL = "read_only_spiral"
    RATE_LIMIT = "rate_limit"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_TURNS = "max_turns"
    LONG_TOOL = "long_tool"
    CUSTOM = "custom"


class AnomalySeverity(str, Enum):
    """Severity level of an anomaly."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TimelineBar(BaseModel):
    """A single operation bar on the timeline.

    Rendered as one ECharts custom-series bar.
    """

    id: str = Field(description="Unique bar identifier")
    type: BarType = Field(description="Operation type")
    label: str = Field(description="Human-readable label")
    start_time: float = Field(description="Unix timestamp (seconds)")
    end_time: float = Field(description="Unix timestamp (seconds)")
    duration_ms: int = Field(description="Computed duration in milliseconds")
    agent_id: str | None = Field(default=None, description="Owning agent/runner")
    group_id: str | None = Field(default=None, description="Logical group (phase/turn)")
    parent_id: str | None = Field(default=None, description="Parent bar for nesting")
    status: BarStatus = Field(default=BarStatus.SUCCESS)
    depth: int = Field(default=0, description="Nesting depth for tree rendering")
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific payload (tool_name, params excerpt, etc.)",
    )
    color: str | None = Field(
        default=None,
        description="Optional override color (hex string)",
    )
    category: OperationCategory | None = Field(
        default=None,
        description="High-level operation category (read/execute/write/orchestrate/other). "
        "Filled by OperationCategorizer; consumed by the multi-session waterfall legend.",
    )

    # Fields used by the curved timeline renderer.
    # All Optional, all default-initialised so existing callers and serialised
    # payloads are byte-compatible. Populated by transcript_parser where
    # upstream data is available; the multi-session view builder also
    # computes ``absolute_time`` from ``start_time`` at serialisation time.
    user_role: str | None = Field(
        default=None,
        description="For user-message bars: 'user' (prompt) or 'system' (subtype).",
    )
    user_text: str | None = Field(
        default=None,
        description="Resolved user-prompt text (joined from text blocks / bare string).",
    )
    system_text: str | None = Field(
        default=None,
        description="Resolved system-injected text (away_summary body, etc.).",
    )
    model: str | None = Field(
        default=None,
        description="Model label carried on the assistant record (e.g. 'claude-opus-4-7').",
    )
    absolute_time: str | None = Field(
        default=None,
        description="ISO-8601 string of ``start_time``; computed by the builder at serialisation time.",
    )
    duration_unrecorded: bool = Field(
        default=False,
        description="True when no real duration was stamped (placeholder ms only). "
        "The bezier view renders the bar at minimum visible width and labels the duration '未记录'.",
    )
    duration_heuristic: bool = Field(
        default=False,
        description="True when the duration was synthesised from a heuristic (next-bar gap). "
        "The bezier view labels the duration '估算'.",
    )
    ts_unrecorded: bool = Field(
        default=False,
        description="True when the underlying record had no parseable timestamp. "
        "The bezier view labels all time fields '未记录'.",
    )
    input_text: str | None = Field(
        default=None,
        description="For LLM events: the user prompt (or prior tool result) that triggered this turn. "
        "Reserved for a future parentUuid-chain resolver; v1 always None.",
    )
    token_in: int | None = Field(
        default=None,
        description="Input tokens consumed in this turn. v1 always None (parser does not yet extract per-event usage).",
    )
    token_out: int | None = Field(
        default=None,
        description="Output tokens generated in this turn. v1 always None.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "tc-001",
                "type": "tool_call",
                "label": "Read src/main.py",
                "start_time": 1717500000.0,
                "end_time": 1717500001.2,
                "duration_ms": 1200,
                "agent_id": "agent-1",
                "group_id": "turn-3",
                "parent_id": None,
                "status": "success",
                "depth": 0,
                "detail": {"tool_name": "Read", "path": "src/main.py"},
            }
        }
    }


class Anomaly(BaseModel):
    """Detected anomaly in a session.

    6 fields; surfaced in the anomaly panel.
    """

    type: AnomalyType
    severity: AnomalySeverity
    session_id: str
    description: str
    timestamp: float
    suggestion: str = ""
    bar_id: str | None = None  # link to triggering bar


class OperationStats(BaseModel):
    """Aggregated operation statistics for a session.

    8 fields; displayed in the stats bar.

    Note: ``total_duration_ms`` is the *sum of per-bar ``duration_ms``
    values* (parser-supplied approximations — text bars are 100ms,
    message bars 50ms, etc).  ``wall_clock_duration_ms`` is the true
    wall-clock span from the earliest bar start to the latest bar end,
    which can diverge significantly from ``total_duration_ms`` when
    the session's metadata ``start_time`` is clock-skewed relative to
    its transcript (e.g. a resume that re-opens an old session).
    """

    total_ops: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    avg_duration_ms: float = 0.0
    max_concurrent: int = 0
    total_duration_ms: int = 0
    wall_clock_duration_ms: int = 0
    context_tokens: int = 0
    cost_usd: float = 0.0


class AgentTreeNode(BaseModel):
    """Node in the multi-agent tree.

    The flat parent-linked representation also carries layout fields populated
    by AgentTreeLayout for the waterfall view.
    """

    agent_id: str
    name: str
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    session_ref: str  # session/run_id this node belongs to
    status: BarStatus = BarStatus.SUCCESS
    depth: int = 0
    stats: OperationStats = Field(default_factory=OperationStats)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # ---- layout (populated by AgentTreeLayout) ----
    spawn_x: float | None = Field(
        default=None,
        description="Relative-time x where this sub-agent was spawned (parent's Agent call).",
    )
    join_x: float | None = Field(
        default=None,
        description="Relative-time x where this sub-agent joined back to the parent.",
    )
    depth_y: int = Field(
        default=0,
        description="Y row in the waterfall view (0 = parent agent row, 1+ = sub-agent rows).",
    )
    role: str = Field(
        default="",
        description="Short role pill label, e.g. '评审' or '核对'.",
    )
    role_color: str = Field(
        default="",
        description="Hex color for the role pill (matches the parent swimlane accent).",
    )


class SessionVizData(BaseModel):
    """Complete visualization data for one session.

    23 fields; the primary payload returned by /api/viz/sessions/{sid}.
    """

    session_id: str
    title: str = ""
    workspace: str = ""
    model: str = ""
    provider: str = ""
    start_time: float = 0.0
    end_time: float | None = None
    duration_ms: int = 0
    status: str = "unknown"  # running, completed, failed, etc.
    agent_name: str = ""
    tags: list[str] = Field(default_factory=list)

    # Aggregated stats
    stats: OperationStats = Field(default_factory=OperationStats)

    # Timeline bars (primary rendering data)
    timeline: list[TimelineBar] = Field(default_factory=list)

    # Anomalies
    anomalies: list[Anomaly] = Field(default_factory=list)

    # Multi-agent tree represented as a flat list with parent references.
    agent_tree: list[AgentTreeNode] = Field(default_factory=list)

    # Linked local artifacts
    report_path: str | None = None  # Verification report path
    tool_events_path: str | None = None  # Tool-event audit NDJSON path
    debug_log_path: str | None = None  # Debug timeline NDJSON path

    # Raw data pointers (not serialized in full; frontend fetches on demand)
    transcript_path: str | None = None
    snapshot_path: str | None = None
    # New-format: parent session's sub-agent transcripts may live under
    # ``<sessions_dir>/<sid>/subagents/agent-*.jsonl`` (nested) or in the
    # flat ``~/.clawcodex/transcripts/`` fallback. This field records the
    # resolved directory so downstream consumers (e.g. agent tree
    # builder) can load child transcripts without re-deriving paths.
    transcripts_dir: str | None = None

    # Computed fields
    turn_count: int = 0
    tool_count: int = 0
    phase_count: int = 0
    detected_mode: str = ""  # e.g. "headless", "interactive"
    config_summary: dict[str, Any] = Field(default_factory=dict)
    parse_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal damaged or unsupported transcript records.",
    )

    # Session termination metadata
    end_reason: str | None = None
    end_summary: str = ""

    # Multi-agent waterfall layout summary (populated by AgentTreeLayout)
    agent_layout_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregated layout info: spawn_time, join_time, subagent_count, by_role.",
    )

    # Orchestrator issue association
    issue_id: str = ""
    verification_status: str = ""

    model_config = {"populate_by_name": True}


class WorkspaceInfo(BaseModel):
    """Lightweight workspace descriptor for the workspace list endpoint."""

    id: str
    name: str
    path: str
    session_count: int = 0
    last_updated: float = 0.0


class ShareLink(BaseModel):
    """Share link record for a single session."""

    id: str
    session_id: str
    created_at: float
    expires_at: float
    payload: dict[str, Any] = Field(default_factory=dict)


class ImportStatus(BaseModel):
    """Async import job status."""

    task_id: str
    status: str  # pending, running, completed, failed
    progress: int = 0  # 0-100
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class DashboardEntryViz(BaseModel):
    """A single entry returned by the Agent Dashboard snapshot API.

    Mirrors :class:`extensions.capabilities.dashboard_entry.DashboardEntry`
    as a Pydantic model so FastAPI can use it as a ``response_model``.
    """

    id: str
    source: str
    title: str
    status: str = "pending"
    detail: str = ""
    source_session_id: str | None = None
    progress_pct: float | None = None
    parent_id: str | None = None
    order: int = 0
    tags: list[str] = Field(default_factory=list)
    owner: str | None = None
    updated_at_ms: int = 0
