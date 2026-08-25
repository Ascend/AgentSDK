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

"""Discover and summarize local ClawCodex sessions."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from extensions.visualizer.models.viz_models import SessionVizData
from extensions.visualizer.parsers.transcript_parser import coerce_timestamp, load_transcript_records

logger = logging.getLogger(__name__)

_RUNNING_RECENCY_SECONDS = 300


def _string(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _timestamp(data: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = coerce_timestamp(data.get(key))
        if value:
            return value
    return 0.0


class SessionMetadataParser:
    """Build the light session record used by index and detail APIs.

    ``session.json`` wins whenever it exists.  Only when it is absent do we
    read ``transcript.jsonl``; ``metadata.json`` remains a field fallback for
    either layout.
    """

    def __init__(
        self,
        sessions_dir: Path | None = None,
        transcripts_dir: Path | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self.sessions_dir = sessions_dir or (Path.home() / ".clawcodex" / "sessions")
        self.transcripts_dir = transcripts_dir or (Path.home() / ".clawcodex" / "transcripts")
        self.reports_dir = reports_dir or (Path.home() / ".clawcodex" / "reports")

    def parse(self, session_id: str) -> SessionVizData | None:
        sessions_root = self.sessions_dir.resolve()
        session_dir = (sessions_root / session_id).resolve()
        try:
            session_dir.relative_to(sessions_root)
        except ValueError:
            return None
        if not session_dir.is_dir():
            return None

        session_path = session_dir / "session.json"
        transcript_path = session_dir / "transcript.jsonl"
        metadata_path = session_dir / "metadata.json"
        source_path = session_path if session_path.exists() else transcript_path

        meta, meta_warning = self._read_object(metadata_path)
        records: list[dict[str, Any]] = []
        source_warnings: list[str] = []
        session_doc: dict[str, Any] = {}
        if source_path.exists():
            records, source_warnings, session_doc = load_transcript_records(source_path)

        # session.json fields are authoritative; metadata fills holes.
        primary = session_doc if session_path.exists() else meta
        secondary = meta if session_path.exists() else {}
        viz = SessionVizData(session_id=session_id)
        viz.parse_warnings.extend(source_warnings)
        if meta_warning:
            viz.parse_warnings.append(meta_warning)

        record_summary = self._summarize_records(records)
        viz.model = _string(primary, "model") or _string(secondary, "model") or record_summary["model"]
        viz.provider = _string(primary, "provider") or _string(secondary, "provider") or record_summary["provider"]
        viz.title = (
            _string(primary, "title", "name") or _string(secondary, "title", "name") or viz.model or session_id[:8]
        )
        viz.workspace = _string(primary, "cwd", "workspace", "working_directory") or _string(
            secondary, "cwd", "workspace", "working_directory"
        )
        viz.agent_name = _string(primary, "agent_name", "agent") or _string(secondary, "agent_name", "agent")
        raw_tags = primary.get("tags") if isinstance(primary.get("tags"), list) else secondary.get("tags")
        if isinstance(raw_tags, list):
            viz.tags = [str(tag) for tag in raw_tags]

        declared_start = _timestamp(primary, "created_at", "start_time", "started_at") or _timestamp(
            secondary, "created_at", "start_time", "started_at"
        )
        declared_end = _timestamp(primary, "updated_at", "last_updated", "end_time", "completed_at") or _timestamp(
            secondary, "updated_at", "last_updated", "end_time", "completed_at"
        )
        viz.start_time = record_summary["start_time"] or declared_start
        viz.end_time = record_summary["end_time"] or declared_end or viz.start_time
        if declared_start and (not record_summary["start_time"] or declared_start < viz.start_time):
            viz.start_time = declared_start
        if declared_end and declared_end > (viz.end_time or 0):
            viz.end_time = declared_end
        viz.duration_ms = max(0, int(round(((viz.end_time or 0) - viz.start_time) * 1000)))
        viz.turn_count = record_summary["turn_count"]
        viz.tool_count = record_summary["tool_count"]
        viz.stats.context_tokens = record_summary["context_tokens"]
        viz.stats.cost_usd = record_summary["cost_usd"]
        viz.status = self._infer_status(primary, secondary, source_path)
        viz.transcripts_dir = str(self.transcripts_dir)
        if source_path.exists():
            viz.transcript_path = str(source_path)

        report_path = session_dir / "report.md"
        tool_events_path = session_dir / "events.ndjson"
        debug_log_path = session_dir / "debug.ndjson"
        if report_path.is_file():
            viz.report_path = str(report_path)
        if tool_events_path.is_file():
            viz.tool_events_path = str(tool_events_path)
        if debug_log_path.is_file():
            viz.debug_log_path = str(debug_log_path)

        self._enrich_from_state_journal(viz)
        return viz

    def list_sessions(self, limit: int = 100) -> list[SessionVizData]:
        if not self.sessions_dir.is_dir():
            return []
        ranked = [
            (entry.stat().st_mtime, parsed)
            for entry in self.sessions_dir.iterdir()
            if entry.is_dir() and (parsed := self.parse(entry.name)) is not None
        ]
        ranked.sort(
            key=lambda item: max(item[0], item[1].end_time or 0, item[1].start_time or 0),
            reverse=True,
        )
        return [item for _, item in ranked[:limit]]

    @staticmethod
    def _read_object(path: Path) -> tuple[dict[str, Any], str | None]:
        if not path.exists():
            return {}, None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, f"{path.name}: {exc}"
        if not isinstance(value, dict):
            return {}, f"{path.name}: top-level JSON must be an object"
        return value, None

    @staticmethod
    def _summarize_records(  # pylint: disable=too-many-nested-blocks
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        timestamps: list[float] = []
        model = ""
        provider = ""
        turn_count = 0
        tool_count = 0
        context_tokens = 0
        cost_usd = 0.0
        cumulative_tokens: int | None = None

        for record in records:
            if record.get("type") == "cost_block":
                cost = record.get("cost") if isinstance(record.get("cost"), dict) else {}
                try:
                    cost_usd = float(cost.get("total_cost_usd", cost_usd) or cost_usd)
                except (TypeError, ValueError):
                    pass
                usage_by_model = cost.get("model_usage")
                if isinstance(usage_by_model, dict):
                    total = 0
                    for usage in usage_by_model.values():
                        if not isinstance(usage, dict):
                            continue
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        ):
                            value = usage.get(key)
                            if isinstance(value, (int, float)):
                                total += int(value)
                    if total:
                        cumulative_tokens = total
                continue
            if record.get("type") == "progress" or record.get("isMeta") or record.get("isVirtual"):
                continue
            if record.get("isCompactSummary"):
                continue
            ts = coerce_timestamp(record.get("timestamp"))
            if ts:
                timestamps.append(ts)
            role = record.get("role")
            if role in {"user", "assistant"}:
                turn_count += 1
            if role == "assistant":
                if not model:
                    model = _string(record, "model")
                if not provider:
                    provider = _string(record, "provider")
                usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ):
                    value = usage.get(key)
                    if isinstance(value, (int, float)):
                        context_tokens += int(value)
                for block in record.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_count += 1

        return {
            "start_time": min(timestamps) if timestamps else 0.0,
            "end_time": max(timestamps) if timestamps else 0.0,
            "model": model,
            "provider": provider,
            "turn_count": turn_count,
            "tool_count": tool_count,
            "context_tokens": cumulative_tokens if cumulative_tokens is not None else context_tokens,
            "cost_usd": cost_usd,
        }

    @staticmethod
    def _infer_status(primary: dict[str, Any], secondary: dict[str, Any], source_path: Path) -> str:
        status = _string(primary, "status") or _string(secondary, "status")
        if status:
            return status
        now = time.time()
        if source_path.exists():
            try:
                if now - source_path.stat().st_mtime < _RUNNING_RECENCY_SECONDS:
                    return "running"
            except OSError:
                pass
        updated = _timestamp(primary, "updated_at", "last_updated") or _timestamp(
            secondary, "updated_at", "last_updated"
        )
        if updated and now - updated < _RUNNING_RECENCY_SECONDS:
            return "running"
        return "completed" if source_path.exists() else "unknown"

    def _enrich_from_state_journal(self, viz: SessionVizData) -> None:
        """Read the existing Orchestrator journal without changing its protocol."""
        if not self.reports_dir.is_dir():
            return
        for run_dir in sorted(self.reports_dir.iterdir()):
            journal = run_dir / "state_journal.ndjson"
            if not run_dir.is_dir() or not run_dir.name.startswith("run_") or not journal.exists():
                continue
            try:
                lines = journal.read_text(encoding="utf-8").splitlines()
            except OSError:
                logger.warning("Unable to read state journal: %s", journal, exc_info=True)
                continue
            events: list[dict[str, Any]] = []
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed state journal line: file=%s line=%d",
                        journal,
                        line_number,
                    )
                    continue
                if isinstance(event, dict):
                    events.append(event)
            issue_id = next(
                (
                    str(event.get("issue_id", ""))
                    for event in events
                    if event.get("type") == "session_ref" and event.get("session_id") == viz.session_id
                ),
                "",
            )
            if not issue_id:
                continue
            viz.issue_id = issue_id
            for event in events:
                if event.get("type") == "verification" and str(event.get("issue_id", "")) == issue_id:
                    viz.verification_status = str(event.get("verification_status", ""))
                    break
            break


__all__ = ["SessionMetadataParser"]
