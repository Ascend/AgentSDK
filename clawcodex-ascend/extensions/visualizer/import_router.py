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

"""Conditional, SSRF-guarded session import router.

Only mounted when ``--allow-import`` flag is set.
Provides endpoints for importing external session data via URL,
with SSRF protection.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .models.viz_models import ImportStatus  # pylint: disable=relative-beyond-top-level

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------


def _is_private_host(hostname: str) -> bool:
    """Return whether a literal or resolved address is unsafe to fetch.

    Import targets must resolve exclusively to globally routable unicast
    addresses. This rejects loopback, private, link-local, shared, reserved,
    unspecified and multicast ranges for both IPv4 and IPv6.
    """
    import socket

    def _is_unsafe(address: str) -> bool:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return True
        return not parsed.is_global or parsed.is_multicast

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        return _is_unsafe(str(literal))

    try:
        addrinfo = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return True
    return not addrinfo or any(_is_unsafe(sockaddr[0]) for *_, sockaddr in addrinfo)


def _validate_import_url(url: str) -> str:
    """Validate a URL for import, raising on SSRF risk."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs allowed, got: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("URL must have a hostname")
    if _is_private_host(parsed.hostname):
        raise ValueError(f"Import from private/local network not allowed: {parsed.hostname}")
    return url


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ImportRequest(BaseModel):
    """Request body for session import."""

    url: str = Field(description="URL to import session data from")
    workspace_id: str | None = Field(default=None, description="Target workspace ID")
    format: str = Field(
        default="json",
        pattern="^(json|jsonl)$",
        description="Source format (json, jsonl)",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def create_import_router() -> APIRouter:
    """Create the conditional import router."""
    router = APIRouter(prefix="/import", tags=["import"])

    @router.post("", status_code=202)
    async def start_import(request: Request, body: ImportRequest) -> dict[str, str]:
        """Start an async import job (conditional — only when allow_import=True)."""
        from .server import _AppState  # pylint: disable=relative-beyond-top-level

        state: _AppState = request.app.state.viz

        if not state.allow_import:
            raise HTTPException(
                status_code=403,
                detail="Import is not enabled. Use --allow-import flag.",
            )

        # SSRF check
        try:
            safe_url = _validate_import_url(body.url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create async task
        task_id = uuid.uuid4().hex[:12]
        status = ImportStatus(task_id=task_id, status="pending")
        state.import_tasks[task_id] = status

        # Run import in background
        asyncio.create_task(_do_import(state, task_id, safe_url, body))

        return {"task_id": task_id, "status": "pending"}

    @router.get("/status/{task_id}")
    async def get_import_status(request: Request, task_id: str) -> ImportStatus:
        """Check the status of an import job."""
        from .server import _AppState  # pylint: disable=relative-beyond-top-level

        state: _AppState = request.app.state.viz
        status = state.import_tasks.get(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Import task not found")
        return status

    return router


# ---------------------------------------------------------------------------
# Background import logic
# ---------------------------------------------------------------------------


async def _do_import(state: Any, task_id: str, url: str, body: ImportRequest) -> None:
    """Execute the actual import in the background."""
    import_status = state.import_tasks[task_id]
    import_status.status = "running"

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.text

        import_status.progress = 50

        # Save to sessions dir
        session_id = f"imported-{task_id}"
        session_dir = state.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        if body.format == "jsonl":
            # Re-wrap each imported line into the new ClawCodeX transcript
            # envelope: ``{role, content (list of typed blocks), type, uuid,
            # timestamp (ISO 8601), isMeta, isVirtual, isCompactSummary,
            # origin}``. Upstream payload lines are not assumed to already
            # be in this shape; we coerce on write so the visualizer's
            # parsers (transcript_parser / session_parser) can read them
            # uniformly with locally-produced sessions.
            (session_dir / "transcript.jsonl").write_text(_rewrite_imported_jsonl(data), encoding="utf-8")
        else:
            # Save as metadata.json
            (session_dir / "metadata.json").write_text(data, encoding="utf-8")

        import_status.status = "completed"
        import_status.progress = 100
        import_status.result = {"session_id": session_id, "path": str(session_dir)}

    except Exception as e:
        logger.error("Import failed for task %s: %s", task_id, e)
        import_status.status = "failed"
        import_status.error = str(e)


# ---------------------------------------------------------------------------
# Envelope rewriter
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    """Return the current wall-clock time as ISO 8601 (UTC, no microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_iso_timestamp(value: Any) -> str:
    """Coerce an imported timestamp into an ISO 8601 string.

    Accepts float / int (Unix epoch), ISO 8601 string, or ``None`` —
    mirrors the parser-side coercion but always writes ISO output.
    """
    if value is None or value == "":
        return _iso_now()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            # Accept both naive and 'Z'-suffixed ISO strings; emit UTC.
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            return _iso_now()
    return _iso_now()


def _envelope_block_text(text: str) -> dict[str, Any]:
    """Wrap a raw string into a typed ``text`` content block."""
    return {"type": "text", "text": text}


def _coerce_content_blocks(content: Any) -> list[dict[str, Any]]:
    """Coerce ``content`` into a list of typed content blocks.

    The new envelope requires ``content`` to be a list of typed blocks
    (``text`` / ``tool_use`` / ``tool_result`` / ``thinking`` / ...).
    Imported legacy payloads sometimes carry a bare string or a
    ``tool_calls`` list on the envelope — normalize those into blocks so
    the visualizer can render them without special-casing.
    """
    if isinstance(content, list):
        out: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict):
                # Already a typed block; pass through but enforce type key.
                if "type" not in block:
                    out.append({**block, "type": "text"})
                else:
                    out.append(block)
            elif isinstance(block, str):
                out.append(_envelope_block_text(block))
            else:
                out.append(_envelope_block_text(str(block)))
        return out
    if isinstance(content, str):
        return [_envelope_block_text(content)]
    if content is None:
        return []
    return [_envelope_block_text(str(content))]


def _coerce_tool_use_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool_use blocks from either the modern ``content`` list or
    the legacy ``tool_calls`` field on an assistant envelope.

    Returns a list of ``tool_use`` blocks (possibly empty) — caller is
    expected to merge these into the envelope's ``content`` list.
    """
    blocks: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for idx, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            tool_use_id = call.get("id") or call.get("tool_use_id") or f"imported-tu-{idx}"
            func = call.get("function") or {}
            tool_name = func.get("name") if isinstance(func, dict) else call.get("name") or "unknown"
            arguments = func.get("arguments") if isinstance(func, dict) else call.get("arguments") or call.get("input")
            if isinstance(arguments, str):
                # Best-effort: keep the raw string; parsers will JSON-decode
                # when reading. Avoid silent loss.
                input_payload: Any = arguments
            else:
                input_payload = arguments or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "tool_use_id": tool_use_id,
                    "name": tool_name,
                    "input": input_payload,
                }
            )
    return blocks


def _coerce_tool_result_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Lift a top-level ``tool`` envelope (with ``tool_call_id`` / ``content``)
    into a single ``tool_result`` content block.
    """
    if message.get("role") != "tool":
        return []
    tool_use_id = message.get("tool_call_id") or message.get("tool_use_id") or ""
    is_error = bool(message.get("is_error"))
    raw = message.get("content")
    if isinstance(raw, list):
        result_content = raw
    elif isinstance(raw, str):
        result_content = [_envelope_block_text(raw)]
    elif raw is None:
        result_content = []
    else:
        result_content = [_envelope_block_text(str(raw))]
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result_content,
    }
    if is_error:
        block["is_error"] = True
    return [block]


def _envelope_message(
    raw: dict[str, Any],
    *,
    is_first_user: bool,
    fallback_ts: str,
) -> dict[str, Any] | None:
    """Coerce one raw imported record into a new-envelope message dict.

    Returns ``None`` for records that don't carry a useful role (e.g.
    bare metadata lines) — caller is expected to skip them.
    """
    role = raw.get("role")
    if not role:
        return None
    if role not in ("user", "assistant", "system", "tool"):
        return None

    if role == "tool":
        default_origin = "tool_result"
    elif role == "user":
        default_origin = "import" if is_first_user else "user"
    else:
        default_origin = "agent"

    message: dict[str, Any] = {
        "role": role if role != "tool" else "user",
        "type": "message",
        "uuid": raw.get("uuid") or raw.get("id") or uuid.uuid4().hex,
        "timestamp": _to_iso_timestamp(raw.get("timestamp") or raw.get("_timestamp")),
        "isMeta": bool(raw.get("isMeta", False)),
        "isVirtual": bool(raw.get("isVirtual", False)),
        "isCompactSummary": bool(raw.get("isCompactSummary", False)),
        "origin": raw.get("origin") or default_origin,
    }

    # tool_result: synthesize content blocks and stash tool_use_id on detail.
    if role == "tool":
        blocks = _coerce_tool_result_blocks(raw)
        message["content"] = blocks or _coerce_content_blocks(raw.get("content"))
        tool_use_id = raw.get("tool_call_id") or raw.get("tool_use_id") or ""
        if tool_use_id:
            message["toolUseID"] = tool_use_id
        return message

    content_blocks = _coerce_content_blocks(raw.get("content"))
    if role == "assistant":
        content_blocks = content_blocks + _coerce_tool_use_blocks(raw)
        # Carry model / usage / stop_reason forward if present.
        for key in (
            "model",
            "stop_reason",
            "usage",
            "requestId",
            "duration_ms",
            "apiError",
            "error",
            "errorDetails",
        ):
            if raw.get(key) is not None:
                message[key] = raw[key]
        if raw.get("isApiErrorMessage"):
            message["isApiErrorMessage"] = True
    elif role == "system":
        # System subtype may carry level / subtype / data — forward verbatim.
        for key in ("subtype", "level", "preventContinuation", "data"):
            if raw.get(key) is not None:
                message[key] = raw[key]
        if raw.get("toolUseID"):
            message["toolUseID"] = raw["toolUseID"]

    message["content"] = content_blocks
    return message


def _rewrite_imported_jsonl(data: str) -> str:
    """Rewrite an imported JSONL payload to the new ClawCodeX envelope.

    Tolerant of legacy formats (string content + ``tool_calls`` /
    ``tool_call_id``) and bare partial records: anything we can't map
    cleanly is dropped, not crashed on, so an import never bricks the
    visualizer.
    """
    out_lines: list[str] = []
    first_user_seen = False
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON line in imported transcript")
            continue
        if not isinstance(raw, dict):
            continue
        msg = _envelope_message(raw, is_first_user=not first_user_seen, fallback_ts=_iso_now())
        if msg is None:
            continue
        if msg.get("role") == "user" and not first_user_seen:
            first_user_seen = True
        out_lines.append(json.dumps(msg, default=str))
    return "\n".join(out_lines) + ("\n" if out_lines else "")
