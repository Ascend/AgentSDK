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

"""Parse ClawCodex session records into one lane-ready event timeline.

The reader intentionally accepts both transcript shapes used in local data:
flat records (``role``/``content`` at the top level) and Claude-style nested
records (``message.role``/``message.content``).  A ``session.json`` file is
also accepted and reads ``conversation.messages``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from extensions.visualizer.models.viz_models import BarStatus, BarType, TimelineBar

logger = logging.getLogger(__name__)

SYSTEM_SUBTYPES = frozenset({"away_summary", "compact", "local_command"})

_PLAUSIBLE_TOOL_MS: dict[str, int] = {
    "read": 60_000,
    "glob": 60_000,
    "grep": 60_000,
    "webfetch": 120_000,
    "websearch": 120_000,
    "write": 120_000,
    "edit": 120_000,
    "apply_patch": 120_000,
    "bash": 5 * 60_000,
    "shell": 5 * 60_000,
}
_DEFAULT_TOOL_CAP_MS = 10 * 60_000


def coerce_timestamp(value: Any) -> float:
    """Return a Unix timestamp in seconds for ISO, second, or ms values."""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value) / 1000.0 if abs(value) >= 1_000_000_000_000 else float(value)
    if not isinstance(value, str) or not value.strip():
        return 0.0
    raw = value.strip()
    try:
        numeric = float(raw)
    except ValueError:
        numeric = None
    if numeric is not None and math.isfinite(numeric):
        return numeric / 1000.0 if abs(numeric) >= 1_000_000_000_000 else numeric
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def timestamp_iso(value: Any, fallback: float = 0.0) -> str | None:
    """Return the original ISO timestamp or a normalized UTC value."""
    if isinstance(value, str) and value.strip() and coerce_timestamp(value):
        return value.strip()
    ts = coerce_timestamp(value) or fallback
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested ``message`` envelope while preserving record metadata."""
    message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
    out = dict(raw)
    for key in ("role", "content", "model", "provider", "usage"):
        value = message.get(key)
        if value is not None:
            out[key] = value
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        out["content"] = []
    return out


def load_transcript_records(
    path: Path | str,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Load JSONL or ``session.json`` and return records, warnings, top-level data."""
    source = Path(path)
    warnings: list[str] = []
    top: dict[str, Any] = {}
    if not source.exists():
        return [], warnings, top

    if source.suffix.lower() == ".json":
        try:
            loaded = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [], [f"{source.name}: {exc}"], top
        if not isinstance(loaded, dict):
            return [], [f"{source.name}: top-level JSON must be an object"], top
        top = loaded
        conversation = loaded.get("conversation")
        if isinstance(conversation, dict):
            values = conversation.get("messages")
        else:
            values = loaded.get("messages") or loaded.get("records") or loaded.get("transcript")
        if not isinstance(values, list):
            return [], [f"{source.name}: conversation.messages is missing"], top
        records = [normalize_record(item) for item in values if isinstance(item, dict)]
        ignored = len(values) - len(records)
        if ignored:
            warnings.append(f"{source.name}: ignored {ignored} non-object message(s)")
        return records, warnings, top

    records: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(f"{source.name}:{line_number}: {exc.msg}")
                    continue
                if not isinstance(item, dict):
                    warnings.append(f"{source.name}:{line_number}: record is not an object")
                    continue
                records.append(normalize_record(item))
    except OSError as exc:
        warnings.append(f"{source.name}: {exc}")
    return records, warnings, top


def _text_from_blocks(blocks: Iterable[Any], *, include_thinking: bool = True) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif include_thinking and kind == "thinking":
            thinking = block.get("thinking") or block.get("text")
            if isinstance(thinking, str):
                parts.append(thinking)
    return "\n\n".join(part for part in parts if part.strip()).strip()


def _result_text(block: dict[str, Any]) -> str:
    value = block.get("content", "")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _text_from_blocks(value)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _finite_duration(value: Any) -> int | None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return int(round(value))


class TranscriptParser:
    """Convert one main or sub-agent transcript into ``TimelineBar`` events."""

    def __init__(self) -> None:
        from extensions.visualizer.builders.operation_categorizer import OperationCategorizer

        self.warnings: list[str] = []
        self._categorizer = OperationCategorizer()

    def parse_file(
        self,
        path: Path | str,
        *,
        agent_id: str | None = None,
        llm_duration_strategy: str | None = None,
    ) -> list[TimelineBar]:
        source = Path(path)
        records, warnings, _ = load_transcript_records(source)
        self.warnings = warnings
        strategy = llm_duration_strategy or ("trigger-gap" if source.name.lower() == "session.json" else "unrecorded")
        return self._parse_records(records, agent_id=agent_id, llm_duration_strategy=strategy)

    def parse_incremental(
        self,
        path: Path | str,
        last_offset: int = 0,
    ) -> tuple[list[TimelineBar], int]:
        source = Path(path)
        if not source.exists():
            return [], 0
        records: list[dict[str, Any]] = []
        warnings: list[str] = []
        with source.open("r", encoding="utf-8") as handle:
            handle.seek(last_offset)
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(f"{source.name}:{line_number}: {exc.msg}")
                    continue
                if isinstance(raw, dict):
                    records.append(normalize_record(raw))
                else:
                    warnings.append(f"{source.name}:{line_number}: record is not an object")
            new_offset = handle.tell()
        self.warnings = warnings
        return self._parse_records(records, agent_id=None, llm_duration_strategy="unrecorded"), new_offset

    def _parse_records(
        self,
        records: list[dict[str, Any]],
        *,
        agent_id: str | None,
        llm_duration_strategy: str,
    ) -> list[TimelineBar]:
        indexed = list(enumerate(records))
        indexed.sort(key=lambda pair: (coerce_timestamp(pair[1].get("timestamp")) or float("inf"), pair[0]))

        by_uuid = {str(record["uuid"]): record for _, record in indexed if record.get("uuid") is not None}
        results: dict[str, dict[str, Any]] = {}
        tool_names: dict[str, str] = {}
        for _, record in indexed:
            record_ts = coerce_timestamp(record.get("timestamp"))
            for block in record.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("id"):
                    tool_names[str(block["id"])] = str(block.get("name") or "unknown")
                if block.get("type") != "tool_result" or not block.get("tool_use_id"):
                    continue
                tool_use_id = str(block["tool_use_id"])
                results[tool_use_id] = {
                    "timestamp": record_ts,
                    "output": _result_text(block),
                    "is_error": bool(block.get("is_error")),
                    "duration_ms": _finite_duration(block.get("duration_ms"))
                    if block.get("duration_ms") is not None
                    else _finite_duration(block.get("durationMs")),
                }

        bars: list[TimelineBar] = []
        last_user_text: str | None = None
        last_user_ts = 0.0
        last_result_text: str | None = None
        last_result_ts = 0.0
        last_real_ts = 0.0
        prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", agent_id or "main")

        for order, record in indexed:
            if self._skip_record(record):
                continue
            role = str(record.get("role") or "")
            raw_ts = coerce_timestamp(record.get("timestamp"))
            ts_unrecorded = raw_ts <= 0
            ts = raw_ts or last_real_ts
            if raw_ts:
                last_real_ts = raw_ts
            absolute_time = timestamp_iso(record.get("timestamp"), ts)
            content = record.get("content", [])
            lane_id = agent_id or self._record_agent_id(record)

            if role == "system":
                subtype = str(record.get("subtype") or "")
                if subtype in SYSTEM_SUBTYPES:
                    text = _text_from_blocks(content)
                    bars.append(
                        TimelineBar(
                            id=f"{prefix}-system-{order}",
                            type=BarType.SYSTEM,
                            label=subtype,
                            start_time=ts,
                            end_time=ts,
                            duration_ms=0,
                            agent_id=lane_id,
                            status=BarStatus.SUCCESS,
                            detail={
                                "subtype": subtype,
                                "text": text,
                                "absolute_time": absolute_time,
                            },
                            user_role="system",
                            system_text=text or None,
                            absolute_time=absolute_time,
                            duration_unrecorded=True,
                            ts_unrecorded=ts_unrecorded,
                        )
                    )
                continue

            if role == "user":
                user_text = _text_from_blocks(content, include_thinking=False)
                if user_text:
                    last_user_text = user_text
                    last_user_ts = ts
                    bars.append(
                        TimelineBar(
                            id=f"{prefix}-user-{order}",
                            type=BarType.USER,
                            label="User prompt",
                            start_time=ts,
                            end_time=ts,
                            duration_ms=0,
                            agent_id=lane_id,
                            status=BarStatus.SUCCESS,
                            detail={"text": user_text, "absolute_time": absolute_time},
                            user_role="user",
                            user_text=user_text,
                            absolute_time=absolute_time,
                            duration_unrecorded=True,
                            ts_unrecorded=ts_unrecorded,
                        )
                    )
                for block_index, block in enumerate(content):
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_use_id = str(block.get("tool_use_id") or "")
                    if not tool_use_id:
                        continue
                    output = _result_text(block)
                    last_result_text = output
                    last_result_ts = ts
                    tool_name = tool_names.get(tool_use_id, "unknown")
                    marker = TimelineBar(
                        id=f"{prefix}-result-{order}-{block_index}",
                        type=BarType.TOOL_RESULT,
                        label=f"{tool_name} result",
                        start_time=ts,
                        end_time=ts,
                        duration_ms=0,
                        agent_id=lane_id,
                        status=BarStatus.ERROR if block.get("is_error") else BarStatus.SUCCESS,
                        detail={
                            "tool_use_id": tool_use_id,
                            "tool_name": tool_name,
                            "tool_output": output,
                            "absolute_time": absolute_time,
                        },
                        absolute_time=absolute_time,
                        duration_unrecorded=True,
                        ts_unrecorded=ts_unrecorded,
                    )
                    marker.category = self._categorizer.categorize(marker)
                    bars.append(marker)
                continue

            if role != "assistant":
                continue

            text = _text_from_blocks(content)
            usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
            input_text = self._find_input(record, by_uuid) or last_result_text or last_user_text
            trigger_ts = max(last_user_ts, last_result_ts)
            real_llm_duration = _finite_duration(usage.get("duration_ms"))
            llm_start = ts
            llm_duration = real_llm_duration or 0
            duration_unrecorded = real_llm_duration is None
            duration_heuristic = False
            duration_source = "recorded" if real_llm_duration is not None else "unrecorded"
            if real_llm_duration is not None and ts:
                llm_start = max(0.0, ts - real_llm_duration / 1000.0)
            elif llm_duration_strategy == "trigger-gap" and ts and trigger_ts and ts >= trigger_ts:
                llm_start = trigger_ts
                llm_duration = int(round((ts - trigger_ts) * 1000))
                duration_unrecorded = False
                duration_heuristic = True
                duration_source = "estimated"

            if text or any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
                llm_bar = TimelineBar(
                    id=f"{prefix}-llm-{order}",
                    type=BarType.LLM_CALL,
                    label="LLM text",
                    start_time=llm_start,
                    end_time=llm_start + llm_duration / 1000.0,
                    duration_ms=llm_duration,
                    agent_id=lane_id,
                    status=BarStatus.SUCCESS,
                    detail={
                        "text": text,
                        "input_text": input_text,
                        "absolute_time": absolute_time,
                        "duration_source": duration_source,
                        "usage": usage,
                    },
                    model=record.get("model") if isinstance(record.get("model"), str) else None,
                    input_text=input_text,
                    token_in=self._token(usage, "input_tokens"),
                    token_out=self._token(usage, "output_tokens"),
                    absolute_time=absolute_time,
                    duration_unrecorded=duration_unrecorded,
                    duration_heuristic=duration_heuristic,
                    ts_unrecorded=ts_unrecorded,
                )
                llm_bar.category = self._categorizer.categorize(llm_bar)
                bars.append(llm_bar)

            for block_index, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_name = str(block.get("name") or "unknown")
                tool_use_id = str(block.get("id") or f"missing-{order}-{block_index}")
                result = results.get(tool_use_id)
                duration_ms: int | None = None
                reported_duration_ms: int | None = None
                duration_source = "unrecorded"
                if result and result.get("duration_ms") is not None:
                    duration_ms = int(result["duration_ms"])
                    duration_source = "recorded"
                elif result and result.get("timestamp") and ts and result["timestamp"] >= ts:
                    duration_ms = int(round((result["timestamp"] - ts) * 1000))
                    duration_source = "record-gap"
                cap = _PLAUSIBLE_TOOL_MS.get(tool_name.lower(), _DEFAULT_TOOL_CAP_MS)
                if duration_ms is not None and duration_ms > cap:
                    reported_duration_ms = duration_ms
                    duration_ms = None
                    duration_source = "unrecorded"
                params = block.get("input") if isinstance(block.get("input"), dict) else block.get("input")
                detail: dict[str, Any] = {
                    "tool_use_id": tool_use_id,
                    "tool_name": tool_name,
                    "params": params,
                    "tool_input": params,
                    "tool_output": result.get("output") if result else None,
                    "absolute_time": absolute_time,
                    "duration_source": duration_source,
                }
                if reported_duration_ms is not None:
                    detail["reported_duration_ms"] = reported_duration_ms
                if isinstance(params, dict):
                    for key in (
                        "agent_id",
                        "subagent_id",
                        "subagentId",
                        "agentId",
                        "subagent_type",
                        "agent_type",
                        "description",
                    ):
                        if key in params:
                            detail[key] = params[key]
                    if tool_name.lower() in {"agent", "task", "subagent"}:
                        detail["is_agent_invocation"] = True
                        detail["subagent_type"] = params.get("subagent_type") or params.get("agent_type")
                        detail["subagent_description"] = params.get("description")
                tool_bar = TimelineBar(
                    id=f"{prefix}-tool-{order}-{block_index}",
                    type=BarType.TOOL_CALL,
                    label=tool_name,
                    start_time=ts,
                    end_time=ts + (duration_ms or 0) / 1000.0,
                    duration_ms=duration_ms or 0,
                    agent_id=lane_id,
                    status=(
                        BarStatus.ERROR
                        if result and result.get("is_error")
                        else BarStatus.SUCCESS
                        if result
                        else BarStatus.RUNNING
                    ),
                    detail=detail,
                    absolute_time=absolute_time,
                    duration_unrecorded=duration_ms is None,
                    ts_unrecorded=ts_unrecorded,
                )
                tool_bar.category = self._categorizer.categorize(tool_bar)
                bars.append(tool_bar)

        return sorted(bars, key=lambda bar: (bar.start_time, bar.id))

    @staticmethod
    def _skip_record(record: dict[str, Any]) -> bool:
        if record.get("type") in {"cost_block", "progress"}:
            return True
        return bool(
            record.get("isMeta")
            or record.get("isVirtual")
            or record.get("isCompactSummary")
            or record.get("isApiErrorMessage")
        )

    @staticmethod
    def _record_agent_id(record: dict[str, Any]) -> str | None:
        for key in ("agent_id", "subagent_id", "agentId", "subagentId"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _find_input(record: dict[str, Any], by_uuid: dict[str, dict[str, Any]]) -> str | None:
        parent = record.get("parentUuid") or record.get("parent_uuid")
        seen: set[str] = set()
        while parent and str(parent) not in seen:
            key = str(parent)
            seen.add(key)
            candidate = by_uuid.get(key)
            if not candidate:
                break
            text = _text_from_blocks(candidate.get("content", []), include_thinking=False)
            if text:
                return text
            for block in candidate.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    output = _result_text(block)
                    if output:
                        return output
            parent = candidate.get("parentUuid") or candidate.get("parent_uuid")
        return None

    @staticmethod
    def _token(usage: dict[str, Any], key: str) -> int | None:
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) and value >= 0 else None


__all__ = [
    "SYSTEM_SUBTYPES",
    "TranscriptParser",
    "coerce_timestamp",
    "load_transcript_records",
    "normalize_record",
    "timestamp_iso",
]
