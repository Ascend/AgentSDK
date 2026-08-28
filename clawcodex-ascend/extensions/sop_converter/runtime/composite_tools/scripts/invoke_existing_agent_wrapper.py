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

"""Executable macro for invoking an agent from a catalog record."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    """Locate the repository root that contains ``extensions/sop_converter``."""
    cursor = Path(_HERE).resolve()
    for parent in cursor.parents:
        if (parent / "extensions" / "sop_converter").is_dir():
            return str(parent)
    return ""


def _bundle_path_from_argv(argv: list[str]) -> str:
    for index, item in enumerate(argv):
        if item == "--bundle-path" and index + 1 < len(argv):
            return argv[index + 1]
    return os.environ.get("CLAWCODEX_BUNDLE_PATH", "").strip()


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _trace_payload(trace: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "step_id": step.step_id,
            "kind": step.kind,
            "status": step.status,
            "error_code": step.error_code,
            "error": step.error,
        }
        for step in trace
    ]


def invoke_existing_agent(
    agent_ref: str = "",
    query: str = "",
    inputs: Any = None,
    bundle_path: str | None = None,
    resource_type: str = "",
    agent_id: str = "",
) -> dict[str, Any]:
    """Execute: load record, materialize agent, invoke, return output.

    ``resource_type`` is accepted only for compatibility with
    older generated fallback wrappers. ``agent_ref`` may be the stable ID or the
    persisted agent name; ``agent_id`` remains a backwards-compatible alias.
    """
    del resource_type
    reference = str(agent_ref or agent_id or "")
    from extensions.sop_converter.composite_runtime import CompositeWorkflowRunner
    from extensions.sop_converter.composite_workflows import invoke_existing_agent_workflow
    from extensions.sop_converter.resource_catalog import CatalogExecutionContext

    result = CompositeWorkflowRunner().run(
        invoke_existing_agent_workflow(),
        {
            "agent_ref": reference,
            "agent_id": str(agent_id or ""),
            "query": query,
            "inputs": inputs,
        },
        resources={
            "catalog": CatalogExecutionContext(
                bundle_path=Path(bundle_path).expanduser().resolve() if bundle_path else None,
                bundle_id=Path(bundle_path).name if bundle_path else "default",
            )
        },
    )
    trace = _trace_payload(result.trace)
    if result.is_error:
        failed_step = trace[-1]["step_id"] if trace else ""
        return {
            "error": result.error,
            "error_code": result.error_code or "workflow_step_failed",
            "step_id": failed_step,
            "agent_ref": reference,
            "agent_id": agent_id or reference,
            "trace": trace,
        }

    output = result.output
    return {
        "agent_ref": reference,
        "agent_id": output.get("agent_id", agent_id or reference),
        "output": output.get("output"),
        "raw": output.get("raw"),
        "text": output.get("text", ""),
        "method": output.get("method", ""),
        "trace": trace,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        _emit(
            {
                "error": "usage: invoke_existing_agent '<json_args>' [--bundle-path <path>]",
                "error_code": "usage",
            }
        )
        return 2
    if argv[1] != "invoke_existing_agent":
        _emit({"error": f"unknown method: {argv[1]}", "error_code": "unknown_method"})
        return 1
    try:
        args = json.loads(argv[2])
    except json.JSONDecodeError as exc:
        _emit({"error": f"invalid JSON args: {exc}", "error_code": "invalid_json"})
        return 1
    if not isinstance(args, dict):
        _emit({"error": "tool arguments must be a JSON object", "error_code": "invalid_input"})
        return 1
    # Standalone execution needs the repo root importable; do it only after
    # validating CLI input so error-only calls do not affect import resolution.
    repo_root = _repo_root()
    if not repo_root:
        _emit(
            {
                "error": (f"repository root containing extensions/sop_converter not found from {_HERE}"),
                "error_code": "repo_root_not_found",
            }
        )
        return 1
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    bundle_path = _bundle_path_from_argv(argv)
    try:
        payload = invoke_existing_agent(
            agent_ref=str(args.get("agent_ref", "") or args.get("name", "")),
            agent_id=str(args.get("agent_id", "")),
            query=str(args.get("query", "")),
            inputs=args.get("inputs"),
            bundle_path=bundle_path or None,
        )
    except Exception as exc:  # noqa: BLE001
        # Always emit structured JSON so callers never parse a traceback.
        _emit(
            {
                "error": f"invoke_existing_agent failed: {exc}",
                "error_code": getattr(exc, "error_code", "invoke_failed"),
                "agent_ref": str(args.get("agent_ref", "") or args.get("name", "")),
                "agent_id": str(args.get("agent_id", "")),
            }
        )
        return 1
    _emit(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
