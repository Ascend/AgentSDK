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

"""``clawcodex api`` CLI subcommand for the Hermes-compatible remote API."""

from __future__ import annotations

# Relative imports bind local code; target subpackages are exposed dynamically.
# pylint: disable=relative-beyond-top-level,no-name-in-module

import argparse
import logging
import sys
from pathlib import Path


def register_api_subcommand() -> None:
    """Register the ``api`` subcommand with the CLI registry."""

    from clawcodex_ext.cli.subcommand_registry import register

    @register("api")
    def _api_handler(args: list[str]) -> int:
        return run_api(args)


def run_api(args: list[str] | None = None) -> int:
    """Entry point for ``clawcodex api``."""

    parser = argparse.ArgumentParser(
        prog="clawcodex api",
        description="Run the ClawCodex Hermes-compatible Remote Agent API",
    )
    subparsers = parser.add_subparsers(dest="command")
    serve = subparsers.add_parser("serve", help="Start the HTTP API server")
    serve.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8642, help="Port to listen on (default: 8642)")
    serve.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace exposed to agent runs (default: current directory)",
    )
    serve.add_argument("--model", default=None, help="Default model for agent runs")
    serve.add_argument(
        "--provider",
        default=None,
        help="Provider for agent runs (default: current ClawCodex config)",
    )
    serve.add_argument("--max-turns", type=int, default=20, help="Maximum agent tool turns")
    serve.add_argument(
        "--permission-mode",
        choices=["bypassPermissions", "dontAsk"],
        default="bypassPermissions",
        help=(
            "Tool approval policy: bypass interactive approvals or deny tools "
            "that require approval (default: bypassPermissions)"
        ),
    )
    serve.add_argument(
        "--state-limit",
        type=int,
        default=128,
        help="Maximum in-memory Responses API entries (default: 128)",
    )
    serve.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Request timeout in seconds (default: 600)",
    )
    serve.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    parsed = parser.parse_args(args)
    if parsed.command != "serve":
        parser.print_help()
        return 2

    logging.basicConfig(
        level=getattr(logging, parsed.log_level.upper()),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    from .core import RemoteAPIConfig
    from .stdlib_server import serve

    config = RemoteAPIConfig(
        workspace=parsed.workspace.expanduser().resolve(),
        host=parsed.host,
        port=parsed.port,
        provider=parsed.provider,
        model=parsed.model,
        max_turns=parsed.max_turns,
        permission_mode=parsed.permission_mode,
        timeout_seconds=parsed.timeout,
        state_limit=parsed.state_limit,
    )
    print(f"ClawCodex Remote Agent API starting at http://{parsed.host}:{parsed.port}")
    print(f"Workspace: {config.workspace}")
    print(f"Permission mode: {config.permission_mode}")
    print("Press Ctrl+C to stop\n")

    try:
        serve(config)
    except KeyboardInterrupt:
        print("\nRemote Agent API stopped.")
    except Exception as exc:
        print(f"Error starting remote API: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run_api())
