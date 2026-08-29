#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=cell-var-from-loop
from __future__ import annotations

import inspect
import json
from typing import Any, Protocol

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from .mcp import _run_async, _run_on_loop


class _McpResourceClient(Protocol):
    def list_resources(self) -> list[dict[str, Any]]: ...
    def read_resource(self, uri: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Prompts (ported from TS ListMcpResourcesTool/prompt.ts and
# ReadMcpResourceTool/prompt.ts)
# ---------------------------------------------------------------------------

LIST_MCP_RESOURCES_DESCRIPTION = """\
Lists available resources from configured MCP servers.
Each resource object includes a 'server' field indicating which server it's from.

Usage examples:
- List all resources from all servers: `listMcpResources`
- List resources from a specific server: `listMcpResources({ server: "myserver" })`\
"""

LIST_MCP_RESOURCES_PROMPT = """\
List available resources from configured MCP servers.
Each returned resource will include all standard MCP resource fields plus a 'server' field
indicating which server the resource belongs to.

Parameters:
- server (optional): The name of a specific MCP server to get resources from. If not provided,
  resources from all servers will be returned.\
"""

READ_MCP_RESOURCE_DESCRIPTION = """\
Reads a specific resource from an MCP server.
- server: The name of the MCP server to read from
- uri: The URI of the resource to read

Usage examples:
- Read a resource from a server: `readMcpResource({ server: "myserver", uri: "my-resource-uri" })`\
"""

READ_MCP_RESOURCE_PROMPT = """\
Reads a specific resource from an MCP server, identified by server name and resource URI.

Parameters:
- server (required): The name of the MCP server from which to read the resource
- uri (required): The URI of the resource to read\
"""


# ---------------------------------------------------------------------------
# mapResultToApi for ListMcpResourcesTool
# (ported from TS ListMcpResourcesTool.ts mapToolResultToToolResultBlockParam)
# ---------------------------------------------------------------------------


def _list_resources_map_result_to_api(output: Any, tool_use_id: str) -> dict[str, Any]:
    """Format ListMcpResources result for the API.

    Returns a helpful message for empty results, JSON for non-empty.
    """
    if not output or (isinstance(output, list) and len(output) == 0):
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": "No resources found. MCP servers may still provide tools even if they have no resources.",
        }
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(output),
    }


# ---------------------------------------------------------------------------
# mapResultToApi for ReadMcpResourceTool
# (ported from TS ReadMcpResourceTool.ts mapToolResultToToolResultBlockParam)
# ---------------------------------------------------------------------------


def _read_resource_map_result_to_api(output: Any, tool_use_id: str) -> dict[str, Any]:
    """Format ReadMcpResource result for the API.

    Returns JSON-stringified content.
    """
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(output),
    }


# ---------------------------------------------------------------------------
# Call implementations
# ---------------------------------------------------------------------------


def _list_mcp_resources_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    server = tool_input.get("server")
    if server is not None and (not isinstance(server, str) or not server.strip()):
        raise ToolInputError("server must be a non-empty string when provided")

    clients: list[tuple[str, Any]]
    if server:
        client = context.mcp_clients.get(server)
        if client is None:
            return ToolResult(
                name="ListMcpResourcesTool",
                output={"error": f"mcp server not connected: {server}"},
                is_error=True,
            )
        clients = [(server, client)]
    else:
        clients = list(context.mcp_clients.items())

    resources: list[dict[str, Any]] = []
    manager_loop = getattr(context, "mcp_manager_loop", None)
    for name, client in clients:
        if hasattr(client, "list_resources"):
            try:

                async def _async_list() -> Any:
                    raw = client.list_resources()
                    if inspect.isawaitable(raw):
                        raw = await raw
                    return raw

                coro = _async_list()
                if manager_loop is not None and not manager_loop.is_closed():
                    items = _run_on_loop(coro, manager_loop)
                else:
                    items = _run_async(coro)
            except Exception as e:
                resources.append({"server": name, "uri": "", "name": "", "description": str(e)})
                continue
            if isinstance(items, list):
                for r in items:
                    if isinstance(r, dict):
                        resources.append(
                            {
                                "uri": str(r.get("uri", "")),
                                "name": str(r.get("name", "")),
                                "mimeType": r.get("mimeType"),
                                "description": r.get("description"),
                                "server": name,
                            }
                        )
    return ToolResult(name="ListMcpResourcesTool", output=resources)


ListMcpResourcesTool: Tool = build_tool(
    name="ListMcpResourcesTool",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "server": {
                "type": "string",
                "description": "Optional server name to filter resources by",
            },
        },
    },
    call=_list_mcp_resources_call,
    prompt=LIST_MCP_RESOURCES_PROMPT,
    description=LIST_MCP_RESOURCES_DESCRIPTION,
    map_result_to_api=_list_resources_map_result_to_api,
    max_result_size_chars=100_000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    is_mcp=True,
    search_hint="list resources from connected MCP servers",
    # Mirrors TS ListMcpResourcesTool.toAutoClassifierInput.
    to_auto_classifier_input=lambda input_data: ((input_data or {}).get("server", "") or ""),
)


def _read_mcp_resource_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    server = tool_input.get("server")
    uri = tool_input.get("uri")
    if not isinstance(server, str) or not server.strip():
        raise ToolInputError("server must be a non-empty string")
    if not isinstance(uri, str) or not uri.strip():
        raise ToolInputError("uri must be a non-empty string")

    client = context.mcp_clients.get(server)
    if client is None:
        return ToolResult(
            name="ReadMcpResourceTool",
            output={"error": f"mcp server not connected: {server}"},
            is_error=True,
        )
    if not hasattr(client, "read_resource"):
        return ToolResult(
            name="ReadMcpResourceTool",
            output={"error": f"mcp server does not support resources: {server}"},
            is_error=True,
        )
    try:

        async def _async_read() -> Any:
            raw = client.read_resource(uri)
            if inspect.isawaitable(raw):
                raw = await raw
            return raw

        coro = _async_read()
        manager_loop = getattr(context, "mcp_manager_loop", None)
        if manager_loop is not None and not manager_loop.is_closed():
            out = _run_on_loop(coro, manager_loop)
        else:
            out = _run_async(coro)
    except Exception as e:
        return ToolResult(
            name="ReadMcpResourceTool",
            output={"error": f"failed to read resource from {server}: {e}"},
            is_error=True,
        )
    if isinstance(out, dict) and "contents" in out:
        return ToolResult(name="ReadMcpResourceTool", output=out)
    return ToolResult(
        name="ReadMcpResourceTool",
        output={"contents": [{"uri": uri, **(out if isinstance(out, dict) else {"text": str(out)})}]},
    )


ReadMcpResourceTool: Tool = build_tool(
    name="ReadMcpResourceTool",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "server": {
                "type": "string",
                "description": "The MCP server name",
            },
            "uri": {
                "type": "string",
                "description": "The resource URI to read",
            },
        },
        "required": ["server", "uri"],
    },
    call=_read_mcp_resource_call,
    prompt=READ_MCP_RESOURCE_PROMPT,
    description=READ_MCP_RESOURCE_DESCRIPTION,
    map_result_to_api=_read_resource_map_result_to_api,
    max_result_size_chars=100_000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
    is_mcp=True,
    search_hint="read a specific MCP resource by URI",
    # Mirrors TS ReadMcpResourceTool.toAutoClassifierInput.
    to_auto_classifier_input=lambda input_data: (
        f"{(input_data or {}).get('server', '')} {(input_data or {}).get('uri', '')}"
    ),
)
