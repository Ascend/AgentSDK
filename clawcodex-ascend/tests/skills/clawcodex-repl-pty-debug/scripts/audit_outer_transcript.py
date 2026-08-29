#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Audit outer ClawCodex A transcripts for forbidden no-discovery tool calls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TOOL_CALL_RE = re.compile(r"●\s*(Bash|Read|Glob|Grep)\s*(?:\(|\n)", re.MULTILINE)
FORBIDDEN_TARGET_RE = re.compile(
    r"(\.agents/skills|\.claude/skills|tests/skills|"
    r"clawcodex_ext/debug/repl_pty_session\.py|scripts/debug/repl_pty_session\.py)"
)
DISCOVERY_SHELL_RE = re.compile(r"\b(ls|find|grep|rg|cat|sed|head|tail|bash|sh|python3\s+-c|python\s+-c)\b")
HELPER_EXEC_RE = re.compile(
    r"\bpython\b\s+[^;&|]*pty_(?:adaptive|jsonl)_driver\.py"
    r"|\b(?:bash|sh)\b\s+[^;&|]*pty_(?:adaptive|jsonl)_driver\.sh"
)
ADAPTIVE_RESPONSE_RE = re.compile(
    r"controller response|artifact read|result\.json|clean\.txt|raw\.log|"
    r'"event"\s*:\s*"(?:ready|observed|error|stopped)"|'
    r"\b(?:delta|kind|state|error_kind|signals)\b",
    re.IGNORECASE,
)
ADAPTIVE_DECISION_RE = re.compile(
    r"\b(?:decision|decide|decided|basis|because|next based on|so ask)\b",
    re.IGNORECASE,
)
ADAPTIVE_NEXT_OP_RE = re.compile(
    r'next controller op|"op"\s*:\s*"(?:send|key|observe|raw)"|'
    r"\bnext\s+(?:send|key|observe|raw)\b",
    re.IGNORECASE,
)
CONTROLLER_RESPONSE_EVENTS = {"ready", "observed", "error", "stopped"}
ADAPTIVE_NEXT_OPS = {"send", "key", "observe", "raw"}


def _line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _compact_window(text: str, start: int, length: int = 900) -> str:
    next_tool = TOOL_CALL_RE.search(text, start + 1)
    end = min(start + length, next_tool.start() if next_tool is not None else len(text))
    return " ".join(text[start:end].split())


def _command_window(text: str, start: int) -> str:
    next_tool = TOOL_CALL_RE.search(text, start + 1)
    output_marker = text.find("⎿", start)
    end = next_tool.start() if next_tool is not None else len(text)
    if output_marker != -1:
        end = min(end, output_marker)
    return " ".join(text[start:end].split())


def _allowed_helper_execution(kind: str, window: str) -> bool:
    if kind != "Bash":
        return False
    if "pty_adaptive_driver.py" not in window and "pty_jsonl_driver.py" not in window:
        if "pty_adaptive_driver.sh" not in window and "pty_jsonl_driver.sh" not in window:
            return False
    if DISCOVERY_SHELL_RE.search(window):
        # The helper may be executed with python, but shell discovery commands
        # around a skill/helper path are still forbidden.
        normalized = window.replace("python3", "python")
        if HELPER_EXEC_RE.search(normalized):
            without_helper_exec = HELPER_EXEC_RE.sub("", normalized, count=1)
            return DISCOVERY_SHELL_RE.search(without_helper_exec) is None
        return False
    return True


def audit_text(text: str, *, source: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        kind = match.group(1)
        command = _command_window(text, match.start())
        target = FORBIDDEN_TARGET_RE.search(command)
        if target is None:
            continue
        if _allowed_helper_execution(kind, command):
            continue
        if kind == "Bash" and not DISCOVERY_SHELL_RE.search(command):
            continue
        findings.append(
            {
                "source": source,
                "line": _line_no(text, match.start()),
                "tool": kind,
                "target": target.group(1),
                "snippet": _compact_window(text, match.start())[:500],
            }
        )
    return findings


def _jsonl_payloads(text: str) -> list[tuple[int, dict[str, Any]]]:
    payloads: list[tuple[int, dict[str, Any]]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append((line_no, payload))
    return payloads


def _audit_adaptive_driver_jsonl(source: str, text: str) -> list[dict[str, Any]] | None:
    payloads = _jsonl_payloads(text)
    if not any(payload.get("event") == "decider_request" for _, payload in payloads):
        return None

    records: list[dict[str, Any]] = []
    last_response_line: int | None = None
    for line_no, payload in payloads:
        event = payload.get("event")
        if event in CONTROLLER_RESPONSE_EVENTS:
            last_response_line = line_no
            continue
        if event != "decider_request" or payload.get("source") != "decide_next(response)":
            continue
        request = payload.get("request")
        basis = payload.get("basis")
        op = request.get("op") if isinstance(request, dict) else None
        records.append(
            {
                "source": source,
                "ok": (last_response_line is not None and isinstance(basis, dict) and op in ADAPTIVE_NEXT_OPS),
                "response_line": last_response_line,
                "decision_line": line_no,
                "next_op_line": line_no if op in ADAPTIVE_NEXT_OPS else None,
                "next_op": op,
                "structured": True,
            }
        )

    if records:
        return records
    return [
        {
            "source": source,
            "ok": False,
            "response_line": None,
            "decision_line": None,
            "next_op_line": None,
            "structured": True,
        }
    ]


def audit_adaptive_order(texts: list[tuple[str, str]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for source, text in texts:
        structured_records = _audit_adaptive_driver_jsonl(source, text)
        if structured_records is not None:
            records.extend(structured_records)
            continue
        response = ADAPTIVE_RESPONSE_RE.search(text)
        decision = ADAPTIVE_DECISION_RE.search(text, response.end()) if response else None
        next_op = ADAPTIVE_NEXT_OP_RE.search(text, decision.end()) if decision else None
        records.append(
            {
                "source": source,
                "ok": bool(response and decision and next_op),
                "response_line": _line_no(text, response.start()) if response else None,
                "decision_line": _line_no(text, decision.start()) if decision else None,
                "next_op_line": _line_no(text, next_op.start()) if next_op else None,
                "structured": False,
            }
        )
    return {
        "ok": any(record["ok"] for record in records),
        "records": records,
    }


def audit_paths(paths: list[Path], *, require_adaptive_order: bool = False) -> dict[str, Any]:
    all_findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    missing: list[str] = []
    texts: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        scanned.append(str(path))
        text = path.read_text(encoding="utf-8", errors="replace")
        texts.append((str(path), text))
        all_findings.extend(audit_text(text, source=str(path)))
    adaptive_order = None
    if require_adaptive_order:
        adaptive_order = audit_adaptive_order(texts)
        if not adaptive_order["ok"]:
            all_findings.append(
                {
                    "type": "adaptive_order",
                    "message": (
                        "missing response/read -> decision -> next send/key/observe "
                        "evidence in scanned transcript files"
                    ),
                }
            )
    result = {
        "ok": not all_findings,
        "scanned": scanned,
        "missing": missing,
        "findings": all_findings,
    }
    if adaptive_order is not None:
        result["adaptive_order"] = adaptive_order
    return {
        **result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument(
        "--require-adaptive-order",
        action="store_true",
        help="Require response/read -> decision -> next send/key/observe evidence",
    )
    args = parser.parse_args(argv)

    result = audit_paths(args.paths, require_adaptive_order=args.require_adaptive_order)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if result["ok"]:
            print("ok: no forbidden outer no-discovery tool calls found")
        else:
            print("forbidden outer no-discovery tool calls found:")
            for finding in result["findings"]:
                print(
                    f"- {finding['source']}:{finding['line']} "
                    f"{finding['tool']} {finding['target']}: {finding['snippet']}"
                )
        if result["missing"]:
            print("missing files:")
            for path in result["missing"]:
                print(f"- {path}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
