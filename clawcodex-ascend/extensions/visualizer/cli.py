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

"""The ``clawcodex viz`` CLI subcommand.

Usage::

    clawcodex viz                       # Start visualizer server (port 8765)
    clawcodex viz --port 9000           # Custom port
    clawcodex viz --allow-import        # Enable session import endpoint
    clawcodex viz --sessions-dir /path  # Custom sessions directory
    clawcodex viz --transcripts-dir /path  # Custom sub-agent transcripts dir
    clawcodex viz --no-open             # Don't open browser

Layout in the new ClawCodeX format:

- ``~/.clawcodex/sessions/<sid>/transcript.jsonl`` — main session transcript
- ``~/.clawcodex/transcripts/<agent_id>.jsonl`` — sub-agent transcript
  (flat fallback, used when the nested resolver is not registered)
- ``~/.clawcodex/sessions/<sid>/subagents/agent-<agent_id>.jsonl`` —
  sub-agent transcript nested under the parent session
- ``~/.clawcodex/tool-events/<run_id>/events.ndjson`` — tool-event audit log
- ``~/.clawcodex/reports/run_*/state_journal.ndjson`` — orchestrator
  state journals
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)


def register_viz_subcommand() -> None:
    """Register the ``viz`` subcommand with the CLI registry.

    This helper is self-contained. It tries to import the
    CLI subcommand registry lazily and silently no-ops if the registry
    is unavailable (e.g. a minimal environment where
    ``clawcodex_ext.cli.subcommand_registry`` has not been installed,
    or tests that mock the CLI surface). The visualizer package no
    longer has a hard import-time dependency on
    ``clawcodex_ext.cli.subcommand_registry`` — the dependency is
    optional and explicit, suitable for an entry-points style plugin
    loader.
    """
    try:
        from clawcodex_ext.cli.subcommand_registry import register  # pylint: disable=no-name-in-module
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "Could not register viz subcommand (registry unavailable): %s",
            exc,
        )
        return

    @register("viz")
    def _viz_handler(args: list[str]) -> int:
        return run_viz(args)


def run_viz(args: list[str] | None = None) -> int:
    """Entry point for the ``clawcodex viz`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="clawcodex viz",
        description="Start the local Session Visualizer web server",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1; use 0.0.0.0 explicitly for LAN access)",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help="Main sessions directory (default: ~/.clawcodex/sessions)",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=None,
        help="Sub-agent transcripts directory (default: ~/.clawcodex/transcripts)",
    )
    parser.add_argument(
        "--workspaces-file",
        type=Path,
        default=None,
        help="Path to workspaces.json (default: ~/.clawcodex/workspaces.json)",
    )
    parser.add_argument(
        "--allow-import",
        action="store_true",
        default=False,
        help="Enable the session import endpoint (SSRF-protected)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        default=False,
        help="Don't open the browser automatically",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable uvicorn auto-reload (dev mode)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    parsed = parser.parse_args(args)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, parsed.log_level.upper()),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    # Lazy import to avoid pulling in FastAPI on every CLI invocation
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is required for the visualizer. "
            "Ensure the approved AgentSDK runtime provides uvicorn[standard].",
            file=sys.stderr,
        )
        return 1

    try:
        from .server import create_app
    except ImportError as exc:
        missing = exc.name or "visualizer web dependency"
        print(
            f"Error: Visualizer web dependency is unavailable: {missing}",
            file=sys.stderr,
        )
        return 1

    app = create_app(
        sessions_dir=parsed.sessions_dir,
        transcripts_dir=parsed.transcripts_dir,
        workspaces_file=parsed.workspaces_file,
        allow_import=parsed.allow_import,
        host=parsed.host,
        port=parsed.port,
    )

    url = f"http://localhost:{parsed.port}"

    if not parsed.no_open:
        # Open browser after a short delay
        import threading

        def _open_browser() -> None:
            import time

            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"ClawCodex Visualizer starting at {url}")
    print(f"   Sessions dir: {parsed.sessions_dir or '~/.clawcodex/sessions'}")
    print(f"   Transcripts dir: {parsed.transcripts_dir or '~/.clawcodex/transcripts'}")
    print(f"   Import: {'enabled' if parsed.allow_import else 'disabled'}")
    print("   Press Ctrl+C to stop\n")

    try:
        uvicorn.run(
            app,
            host=parsed.host,
            port=parsed.port,
            log_level=parsed.log_level,
            reload=parsed.reload,
        )
    except KeyboardInterrupt:
        print("\nVisualizer stopped.")
    except Exception as e:
        print(f"Error starting visualizer: {e}", file=sys.stderr)
        return 1

    return 0
