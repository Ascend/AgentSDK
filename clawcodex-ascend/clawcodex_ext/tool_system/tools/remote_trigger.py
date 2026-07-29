"""RemoteTriggerTool — manage scheduled remote agent triggers.

Mirrors ``claude-code-best/packages/builtin-tools/.../RemoteTriggerTool.ts``.
This is a thin HTTP client for a remote "triggers" REST API. The TS
original targets the claude.ai CCR API and injects an OAuth bearer token
in-process; clawcodex does not ship that OAuth stack, so this port reads
the endpoint base URL and bearer token from configuration / environment
and otherwise preserves the action set (list/get/create/update/run) and
the audit-log behaviour.

Configuration sources (first non-empty wins):
- ``CLAWCODEX_TRIGGERS_API_URL``  — base URL for the triggers API.
- ``CLAWCODEX_TRIGGERS_TOKEN``    — bearer token sent as Authorization.
- ``CLAWCODEX_TRIGGERS_ORG``      — optional ``x-organization-uuid`` header.

When either the base URL or token is unset the tool reports itself as
disabled (``is_enabled == False``) so the model does not waste a turn
trying to call an unconfigured endpoint.

Read-only for ``list`` / ``get``; mutating for ``create`` / ``update`` /
``run``. Concurrency-safe (stateless HTTP client). Uses ``urllib`` to
avoid an ``httpx`` dependency.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult


_TRIGGER_ID_RE = re.compile(r"^[\w-]+$")
_MAX_BODY_CHARS = 100_000
_REQUEST_TIMEOUT = 20.0

# Process-local audit log (in-memory; the TS original persists to disk).
# Capped to avoid unbounded growth in long-running sessions.
_AUDIT_LOG: list[dict[str, Any]] = []
_AUDIT_CAP = 500
_audit_counter = 0


def _api_base() -> str:
    return os.environ.get("CLAWCODEX_TRIGGERS_API_URL", "").rstrip("/")


def _bearer_token() -> str:
    return os.environ.get("CLAWCODEX_TRIGGERS_TOKEN", "")


def _org_uuid() -> str:
    return os.environ.get("CLAWCODEX_TRIGGERS_ORG", "")


def _is_configured() -> bool:
    return bool(_api_base()) and bool(_bearer_token())


def _append_audit(record: dict[str, Any]) -> str:
    global _audit_counter
    _audit_counter += 1
    audit_id = f"audit-{_audit_counter}"
    entry = {"audit_id": audit_id, "ts": time.time(), **record}
    _AUDIT_LOG.append(entry)
    if len(_AUDIT_LOG) > _AUDIT_CAP:
        del _AUDIT_LOG[: len(_AUDIT_LOG) - _AUDIT_CAP]
    return audit_id


def _validate_trigger_id(trigger_id: str | None) -> str | None:
    if trigger_id is None:
        return None
    if not isinstance(trigger_id, str) or not _TRIGGER_ID_RE.match(trigger_id):
        raise ToolInputError(r"trigger_id must match /^[\w-]+$/")
    return trigger_id


def _request(
    method: str,
    url: str,
    token: str,
    body: Any | None,
) -> tuple[int, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    org = _org_uuid()
    if org:
        headers["x-organization-uuid"] = org

    data: bytes | None = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:  # noqa: S310 — outbound to configured API
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace")) if raw else None
    except (ValueError, json.JSONDecodeError):
        parsed = raw.decode("utf-8", errors="replace")
    return status, parsed


def _remote_trigger_call(tool_input: dict[str, Any], _context: ToolContext) -> ToolResult:
    action = tool_input.get("action")
    if action not in ("list", "get", "create", "update", "run"):
        raise ToolInputError(f"unknown action: {action!r}")

    trigger_id = _validate_trigger_id(tool_input.get("trigger_id"))
    body = tool_input.get("body")
    if body is not None and not isinstance(body, dict):
        raise ToolInputError("body must be an object")

    # Action-specific required-field checks (mirrors the TS switch).
    if action in ("get", "update", "run") and not trigger_id:
        raise ToolInputError(f"{action} requires trigger_id")
    if action in ("create", "update") and body is None:
        raise ToolInputError(f"{action} requires body")

    base = _api_base()
    token = _bearer_token()
    audit_base: dict[str, Any] = {"action": action}
    if trigger_id:
        audit_base["trigger_id"] = trigger_id

    if not base or not token:
        msg = "RemoteTrigger is not configured. Set CLAWCODEX_TRIGGERS_API_URL and CLAWCODEX_TRIGGERS_TOKEN."
        audit_id = _append_audit({**audit_base, "ok": False, "error": msg})
        return ToolResult(
            name="RemoteTrigger",
            output={"status": 0, "json": "", "audit_id": audit_id, "error": msg},
            is_error=True,
        )

    method, url, payload = {
        "list": ("GET", f"{base}/v1/code/triggers", None),
        "get": ("GET", f"{base}/v1/code/triggers/{trigger_id}", None),
        "create": ("POST", f"{base}/v1/code/triggers", body),
        "update": ("POST", f"{base}/v1/code/triggers/{trigger_id}", body),
        "run": ("POST", f"{base}/v1/code/triggers/{trigger_id}/run", {}),
    }[action]

    try:
        status, parsed = _request(method, url, token, payload)
    except Exception as exc:  # noqa: BLE001 — surface transport errors + audit
        audit_id = _append_audit({**audit_base, "ok": False, "error": str(exc)})
        return ToolResult(
            name="RemoteTrigger",
            output={"status": 0, "json": "", "audit_id": audit_id, "error": str(exc)},
            is_error=True,
        )

    ok = 200 <= status < 300
    audit_id = _append_audit({**audit_base, "ok": ok, "status": status})
    json_str = json.dumps(parsed, ensure_ascii=False) if parsed is not None else ""
    if len(json_str) > _MAX_BODY_CHARS:
        json_str = json_str[:_MAX_BODY_CHARS] + "\n[truncated]"

    return ToolResult(
        name="RemoteTrigger",
        output={"status": status, "json": json_str, "audit_id": audit_id},
        is_error=not ok,
    )


def _map_result_to_api(output: Any, tool_use_id: str) -> dict[str, Any]:
    if isinstance(output, dict):
        lines = [f"HTTP {output.get('status', '')}"]
        if output.get("error"):
            lines.append(str(output["error"]))
        if output.get("json"):
            lines.append(str(output["json"]))
        content: str | list[dict[str, Any]] = "\n".join(lines)
    else:
        content = str(output)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


RemoteTriggerTool: Tool = build_tool(
    name="RemoteTrigger",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "get", "create", "update", "run"],
                "description": "CRUD/run action against the triggers API.",
            },
            "trigger_id": {
                "type": "string",
                "pattern": r"^[\w-]+$",
                "description": "Required for get, update, and run.",
            },
            "body": {
                "type": "object",
                "additionalProperties": True,
                "description": "JSON body for create and update.",
            },
        },
        "required": ["action"],
    },
    call=_remote_trigger_call,
    prompt=(
        "RemoteTrigger: manage scheduled remote agent triggers via the "
        "triggers REST API. Use this instead of curl — the bearer token is "
        "read from configuration in-process and never exposed to the shell.\n\n"
        "Actions:\n"
        "- list: GET /v1/code/triggers\n"
        "- get: GET /v1/code/triggers/{trigger_id}\n"
        "- create: POST /v1/code/triggers (requires body)\n"
        "- update: POST /v1/code/triggers/{trigger_id} (requires body)\n"
        "- run: POST /v1/code/triggers/{trigger_id}/run\n\n"
        "Configuration: CLAWCODEX_TRIGGERS_API_URL, CLAWCODEX_TRIGGERS_TOKEN, "
        "CLAWCODEX_TRIGGERS_ORG (optional). When unset the tool is disabled. "
        "Every call is recorded to an in-memory audit log (capped)."
    ),
    description=(
        "Manage scheduled remote agent triggers (triggers) via the triggers REST API. Auth handled in-process."
    ),
    search_hint="manage scheduled remote agent triggers",
    max_result_size_chars=100_000,
    should_defer=True,
    is_enabled=_is_configured,
    is_concurrency_safe=lambda _input: True,
    is_read_only=lambda input_data: input_data.get("action") in ("list", "get"),
    is_destructive=lambda input_data: input_data.get("action") == "run",
    map_result_to_api=_map_result_to_api,
    to_auto_classifier_input=lambda input_data: (
        f"RemoteTrigger {input_data.get('action', '')}"
        + (f" {input_data['trigger_id']}" if input_data.get("trigger_id") else "")
    ),
)
