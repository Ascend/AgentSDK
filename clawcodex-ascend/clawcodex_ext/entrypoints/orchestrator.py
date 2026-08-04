#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Orchestrator CLI subcommand router — noun-verb structure.

Usage:
  clawcodex orchestrator server status|stop|start [--workspace PATH]   # daemon-level
  clawcodex orchestrator issue list|show|stop|... --id <id>            # issue-level
  clawcodex orchestrator workflow init [--kind TRACKER] ...            # scaffold workflow.md
  clawcodex orchestrator dashboard [--port PORT]                       # dashboard

Design:
  - noun-verb structure: ``<noun> <verb> [--param value]``
  - Self-describing named parameters (``--id``, ``--workspace``, etc.)
  - All operations are idempotent where possible

Each subparser declares its own arguments — there are no global options
extracted before the subcommand token.
"""

from __future__ import annotations

import argparse
import os


def run_orchestrator_subcommand(rest: list[str]) -> int:
    """Handle ``clawcodex orchestrator <subcommand>``.

    Supports the noun-verb structure (``server|issue|dashboard``).

    Returns the process exit code.
    """
    # Find subcommand token position (everything else is passed through)
    # Pass through everything verbatim — subparsers own their args
    filtered_rest = list(rest)

    # Build the main parser with subparsers
    parser = argparse.ArgumentParser(
        prog="clawcodex orchestrator",
        description="Autonomous issue processing orchestration",
        epilog="""
Usage (noun-verb):
  server status|stop|start    Manage the orchestrator daemon
  issue list|show|tail|...    Manage individual issues (use --id <id>)
  workflow init [OPTIONS]     Scaffold workflow.md from packaged template
  dashboard [--port PORT]     Standalone LiveView UI
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Import CLI modules to register their subparsers
    from extensions.orchestrator.cli.dashboard import add_dashboard_parser
    from extensions.orchestrator.cli.issue import add_issue_parser
    from extensions.orchestrator.cli.rules import add_rules_parser
    from extensions.orchestrator.cli.server import add_server_parser
    from extensions.orchestrator.cli.workspace import add_workspace_parser
    from extensions.orchestrator.cli.workflow import add_workflow_parser

    # Register noun-verb subparsers
    add_server_parser(subparsers)  # server status|stop|start
    add_issue_parser(subparsers)  # issue list|show|tail|stop|pause|resume|...
    add_rules_parser(subparsers)  # rules list|review|delete|refresh|stats
    add_workflow_parser(subparsers)  # workflow init|list-templates
    add_workspace_parser(subparsers)  # workspace list|show|cd|cleanup|verify
    add_dashboard_parser(subparsers)  # dashboard [--port PORT]

    # Parse all arguments
    if os.environ.get("_ARGCOMPLETE") == "1":
        import argcomplete

        argcomplete.autocomplete(parser)
    args = parser.parse_args(filtered_rest)

    # Dispatch — noun-verb dispatch
    if args.subcommand == "server":
        from extensions.orchestrator.cli.server import run as run_server

        return run_server(args)
    elif args.subcommand == "issue":
        from extensions.orchestrator.cli.issue import run as run_issue

        return run_issue(args)
    elif args.subcommand == "workflow":
        from extensions.orchestrator.cli.workflow import run as run_workflow

        return run_workflow(args)
    elif args.subcommand == "workspace":
        from extensions.orchestrator.cli.workspace import run as run_workspace

        return run_workspace(args)
    elif args.subcommand == "rules":
        from extensions.orchestrator.cli.rules import run as run_rules

        return run_rules(args)
    elif args.subcommand == "dashboard":
        from extensions.orchestrator.cli.dashboard import run as run_dashboard

        return run_dashboard(args)
    else:
        parser.print_help()
        return 2
