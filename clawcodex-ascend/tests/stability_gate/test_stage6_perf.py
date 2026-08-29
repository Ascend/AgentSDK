#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Tests for stage6 perf."""

from __future__ import annotations

import os
import time

_THRESHOLD_MULT = float(os.environ.get("CLAWCODEX_CI_THRESHOLD_MULT", "1"))


class TestStage6Perf:
    """Tests for TestStage6Perf."""

    # ── CLI budgets ──────────────────────────────────────────────

    def test_cli_help_import_time(self):
        """Verify cli help import time."""
        import subprocess
        import sys

        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.argv = ['clawcodex', '--help']; "
                    "from src.cli import _build_parser; p = _build_parser(); "
                    "p.parse_args(['--help'])"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        budget = 3.0 * _THRESHOLD_MULT
        assert elapsed < budget, (
            f"CLI --help import took {elapsed:.2f}s, expected < {budget:.2f}s (threshold multiplier={_THRESHOLD_MULT})"
        )

    def test_cli_subprocess_startup_time(self):
        """Verify cli subprocess startup time."""
        import subprocess
        import sys

        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "src.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0
        budget = 5.0 * _THRESHOLD_MULT
        assert elapsed < budget, (
            f"CLI --help subprocess took {elapsed:.2f}s, expected < {budget:.2f}s "
            f"(threshold multiplier={_THRESHOLD_MULT})"
        )

    # ── Conversation budget ──────────────────────────────────────

    def test_conversation_import_time(self):
        """Verify conversation import time."""
        import subprocess
        import sys

        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-c", "from src.agent.conversation import Conversation"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        budget = 2.0 * _THRESHOLD_MULT
        assert elapsed < budget, (
            f"Conversation import took {elapsed:.2f}s, expected < {budget:.2f}s "
            f"(threshold multiplier={_THRESHOLD_MULT})"
        )

    # ── Agent loop budget ────────────────────────────────────────

    def test_agent_loop_warm_start(self):
        """Verify agent loop warm start."""
        import subprocess
        import sys

        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [
                sys.executable,
                "-c",
                ("from src.query import query, QueryParams, QueryEngine, QueryEngineConfig, StreamEvent"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        budget = 3.0 * _THRESHOLD_MULT
        assert elapsed < budget, (
            f"Agent loop warm-start took {elapsed:.2f}s, expected < {budget:.2f}s "
            f"(threshold multiplier={_THRESHOLD_MULT})"
        )

    # ── Tool execution budget ────────────────────────────────────

    def test_tool_execution_path_latency(self):
        """Verify tool execution path latency."""
        import subprocess
        import sys

        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [
                sys.executable,
                "-c",
                (
                    "from src.tool_system.registry import ToolRegistry, get_all_base_tools; "
                    "from src.tool_system.build_tool import find_tool_by_name, Tool"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        budget = 2.0 * _THRESHOLD_MULT
        assert elapsed < budget, (
            f"Tool execution path took {elapsed:.2f}s, expected < {budget:.2f}s "
            f"(threshold multiplier={_THRESHOLD_MULT})"
        )

    # ── REPL / Headless budget ───────────────────────────────────

    def test_repl_input_pipeline_cold_start(self):
        """Verify repl input pipeline cold start."""
        import subprocess
        import sys

        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-c", "from src.repl import ClawcodexREPL"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, f"stderr={proc.stderr!r}"
        budget = 5.0 * _THRESHOLD_MULT
        assert elapsed < budget, (
            f"REPL cold start took {elapsed:.2f}s, expected < {budget:.2f}s (threshold multiplier={_THRESHOLD_MULT})"
        )

    def test_repl_heavy_runtime_cold_start(self):
        """Verify repl heavy runtime cold start."""
        import subprocess
        import sys

        pytest = __import__("pytest")
        pytest.importorskip("httpx")

        # Probe isolates import timing from test runner state.
        # Runs in fresh subprocess so sys.modules cache starts empty.
        probe = (
            "import sys, os, time;"
            "os.environ.setdefault('HOME', '/tmp');"
            "sys.path.insert(0, '.');"
            "from clawcodex_ext.repl.core import _load_heavy_runtime;"
            "_load_heavy_runtime();"
            "print(int(time.monotonic() * 1000))"
        )

        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, (
            f"_load_heavy_runtime() failed (rc={proc.returncode}): stderr={proc.stderr[-400:]!r}"
        )
        try:
            heavy_ms = int(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            heavy_ms = -1
        budget = 6.5 * _THRESHOLD_MULT
        assert elapsed < budget, (
            f"REPL _load_heavy_runtime() cold start took {elapsed:.2f}s "
            f"(inner={heavy_ms}ms), expected < {budget:.2f}s "
            f"(threshold multiplier={_THRESHOLD_MULT})"
        )

    def test_repl_first_prompt_end_to_end(self):
        """Verify repl first prompt end to end."""
        import subprocess
        import sys

        pytest = __import__("pytest")
        pytest.importorskip("httpx")

        # Probe is the canonical REPL constructor entry point, minus the
        # interactive prompt loop. Stops just before prompt_toolkit to
        # avoid TTY-dependent timing variance.
        probe = (
            "import sys, os, time;"
            "os.environ.setdefault('HOME', '/tmp');"
            "os.environ.setdefault('CLAWCODEX_API_KEY', 'sk-perf-test');"
            "os.environ.setdefault('ANTHROPIC_API_KEY', 'sk-perf-test');"
            "sys.path.insert(0, '.');"
            "from src.repl import ClawcodexREPL;"
            "repl = ClawcodexREPL(provider_name='anthropic', stream=False);"
            "repl._print_startup_header();"
            "print(int(time.monotonic() * 1000))"
        )

        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, (
            f"REPL end-to-end init failed (rc={proc.returncode}): stderr={proc.stderr[-400:]!r}"
        )
        try:
            inner_ms = int(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            inner_ms = -1
        budget = 7.0 * _THRESHOLD_MULT
        assert elapsed < budget, (
            f"REPL end-to-end cold start took {elapsed:.2f}s "
            f"(inner={inner_ms}ms), expected < {budget:.2f}s "
            f"(threshold multiplier={_THRESHOLD_MULT})"
        )

    def test_build_default_registry_defer_fast(self):
        """Verify build default registry defer fast."""
        import subprocess
        import sys

        pytest = __import__("pytest")
        pytest.importorskip("httpx")

        probe = """
import sys, os, time
os.environ.setdefault('HOME', '/tmp')
sys.path.insert(0, '.')
from src.tool_system.defaults import build_default_registry
t0 = time.monotonic()
r = build_default_registry(provider=object(), defer_extended_tools=True)
elapsed = (time.monotonic() - t0) * 1000
stage_a_count = len(r.list_tools())
print(f\"DEFER_MS={int(elapsed)}\")
print(f\"STAGE_A_COUNT={stage_a_count}\")
"""
        start = time.monotonic()
        proc = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=15,
        )
        time.monotonic() - start
        assert proc.returncode == 0, (
            f"build_default_registry probe failed (rc={proc.returncode}): stderr={proc.stderr[-400:]!r}"
        )
        lines = proc.stdout.strip().splitlines()
        try:
            defer_ms = int(lines[-2].split("=", 1)[1])
            stage_a_count = int(lines[-1].split("=", 1)[1])
        except (ValueError, IndexError):
            defer_ms = -1
            stage_a_count = -1
        # Stage A must register at least 50 tools (51 ALL_STATIC_TOOLS + Agent + ToolSearch).
        assert stage_a_count >= 50, f"Stage A only registered {stage_a_count} tools, expected >= 50"
        # Defer path must return quickly (Stage A is the only sync work).
        1.0 * _THRESHOLD_MULT
        assert defer_ms < 1000, (
            f"build_default_registry(defer_extended_tools=True) took {defer_ms}ms, "
            f"expected < 1000ms (threshold multiplier={_THRESHOLD_MULT}). "
            f"Stage A is supposed to be the only synchronous work."
        )
