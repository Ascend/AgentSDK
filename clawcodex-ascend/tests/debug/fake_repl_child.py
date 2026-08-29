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

from __future__ import annotations

import json
import os
import sys
import termios
import time
import tty
from pathlib import Path


def _read_single_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> int:
    if os.environ.get("FAKE_REPL_READY_MARKER", "1") != "0":
        print(
            "CLAWCODEX_AGENT_DEBUG::repl.ready::"
            + json.dumps({"session_id": "fake-session", "surface": "repl", "stream": True}),
            flush=True,
        )
    else:
        print("╭──────────────── CLAWCODEX ────────────────╮", flush=True)
        print("❯ ", flush=True)
        print("WARNING: your terminal doesn't support cursor position requests (CPR).", flush=True)
    while True:
        try:
            line = input("> ")
        except EOFError:
            print("Goodbye!", flush=True)
            return 0

        if line == "/exit":
            print("Goodbye!", flush=True)
            return 0
        if line.startswith("/goal clear"):
            print("Goal cleared", flush=True)
            continue
        if line.startswith("/goal "):
            print(f"Goal set: {line.removeprefix('/goal ')}", flush=True)
            continue
        if line == "/goal":
            print("Status: active\nTokens: 82/inf", flush=True)
            continue
        if line == "token-status":
            print("Tokens: 0/inf\nTurns executed: 1", flush=True)
            continue
        if line.startswith("write-adaptive-file "):
            path = Path(line.removeprefix("write-adaptive-file ").strip())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "<!doctype html>\n<html>\n<body>\n<h1>Adaptive PTY</h1>\n<p>round1</p>\n</body>\n</html>\n",
                encoding="utf-8",
            )
            print(f"ADAPTIVE-FILE-WROTE {path}", flush=True)
            continue
        if line.startswith("append-adaptive-footer "):
            path = Path(line.removeprefix("append-adaptive-footer ").strip())
            text = path.read_text(encoding="utf-8")
            footer = '<footer data-adaptive="round2">verified</footer>'
            if footer not in text:
                text = text.replace("</body>", f"{footer}\n</body>")
            path.write_text(text, encoding="utf-8")
            print(f"ADAPTIVE-FOOTER-WROTE {path}", flush=True)
            continue
        if line.startswith("write-adaptive-decision "):
            path = Path(line.removeprefix("write-adaptive-decision ").strip())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"basis": "footer verified", "next": "done"}),
                encoding="utf-8",
            )
            print(f"ADAPTIVE-DECISION-WROTE {path}", flush=True)
            continue
        if line == "interleaved-token":
            print("GOAL-PTY", flush=True)
            print("Thinking status redraw", flush=True)
            print("❯", flush=True)
            print("-OK", flush=True)
            continue
        if line == "interleaved-token-with-prompt":
            print("Assistant", flush=True)
            print("PTY-S", flush=True)
            print("⠋ Thinking…  (esc to interrupt · ctrl+b background · enter to queue)", flush=True)
            print("❯", flush=True)
            print("MOKE-OK", flush=True)
            print("❯", flush=True)
            continue
        if line == "delayed-output":
            time.sleep(0.4)
            print("late-output", flush=True)
            continue
        if line == "provider-error":
            print("ProviderError: invalid_api_key from fake provider", flush=True)
            continue
        if line == "network-error":
            print("NetworkError: DNS lookup failed for fake provider", flush=True)
            continue
        if line == "rendered-connection-error":
            print("Assistant", flush=True)
            print("Query error: Connection error.", flush=True)
            print("Connection error.", flush=True)
            continue
        if line == "/tool Skill doc-error-text":
            print("Tool result:", flush=True)
            print(
                '{"success": true, "prompt": "Failure signature: local slash commands '
                "work, but natural-language messages fail with `httpcore.ConnectError`, "
                "`nodename nor servname provided`, `APIConnectionError`, or "
                '`Connection error`."}',
                flush=True,
            )
            print("❯ ", flush=True)
            continue
        if line == "noisy-assistant":
            print("Assistant", flush=True)
            print(
                "⠋ Thinking…  (esc to interrupt · ctrl+b background · enter to queue · 0s)",
                flush=True,
            )
            print(
                "deepseek · deepseek-v4-flash · /tmp/workspace · mode: Default · turns: 0 · tokens: 0 in / 0 out",
                flush=True,
            )
            print("WARNING: your terminal doesn't support cursor position requests (CPR).", flush=True)
            print("NOISY-ASSISTANT-OK", flush=True)
            print("❯ ", flush=True)
            continue
        if line == "permission-prompt":
            print(
                "Permission Required\n\n"
                "  ▸   1. [y] Yes, allow this action\n"
                "      2. [n] No, deny this action\n\n"
                "  ↑↓ navigate · Enter select · 1-9 quick select · Esc cancel\n"
                "⚠ Permission Required\n"
                "  Claude wants to use Bash. Allow?",
                flush=True,
            )
            continue
        if line == "permission-resolved":
            print(
                "Permission Required\n\n"
                "  ▸   1. [y] Yes, allow this action\n"
                "      2. [n] No, deny this action\n\n"
                "  ↑↓ navigate · Enter select · 1-9 quick select · Esc cancel\n\n"
                "Tool result:\n"
                '{"stdout": "SC5-PERMISSION-OK"}\n\n'
                "❯ ",
                flush=True,
            )
            continue
        if line == "permission-resolved-rendered":
            print(
                "Permission Required\n\n"
                "  ▸   1. [y] Yes, allow this action\n"
                "      2. [n] No, deny this action\n\n"
                "  ↑↓ navigate · Enter select · 1-9 quick select · Esc cancel\n\n"
                "● Write (./.reports/pty-debug-loop/tmp-run/permission-file.txt)\n"
                "  ⎿  Wrote 2 lines to ./.reports/pty-debug-loop/tmp-run/permission-file.txt\n"
                "            1  PERMISSION_TOKEN\n"
                "            2  allowed-by-enter-key\n\n"
                "❯ ",
                flush=True,
            )
            continue
        if line == "permission-interactive":
            print(
                "Permission Required\n\n"
                "  ▸   1. [y] Yes, allow this action\n"
                "      2. [n] No, deny this action\n\n"
                "  ↑↓ navigate · Enter select · 1-9 quick select · Esc cancel\n"
                "⚠ Permission Required\n"
                "  Claude wants to use Bash. Allow?",
                flush=True,
            )
            key = _read_single_key()
            if key in {"1", "y", "Y"}:
                print('\nTool result:\n{"stdout": "SC5-PERMISSION-KEY-OK"}\n\n❯ ', flush=True)
            else:
                print("\nTool error:\npermission denied by fake child\n\n❯ ", flush=True)
            continue
        if line.startswith("silent "):
            time.sleep(0.4)
            continue
        if "goal pty ok" in line.lower():
            print("GOAL-PTY-OK", flush=True)
            continue
        print(f"echo:{line}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
