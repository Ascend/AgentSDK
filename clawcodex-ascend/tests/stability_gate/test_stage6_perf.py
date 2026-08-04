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

"""Stage 6 — 性能守卫（budget 详见各测试 docstring）。

验证关键操作的响应时间在可接受范围内。
防止因意外引入重型 import 或阻塞操作导致的 CLI / Agent / REPL 启动缓慢。

CI 阈值放宽
-----------
本文件硬编码的本地基线阈值在 CI runner 上会假阳性（GitHub Actions
ubuntu-latest 2 核通常比开发者本地慢 2-3 倍）。通过环境变量
``CLAWCODEX_CI_THRESHOLD_MULT`` 可以在 CI 上按倍率放宽：

* 本地开发：默认 1（使用原始基线）
* CI PR job：设为 ``2``（保留 ~50% 余量）
* CI nightly job：不设（严格使用原始阈值，作为 perf 回归把关）

参见 ``.github/workflows/ci.yml`` 和 ``stage6-perf-nightly.yml``。
"""

from __future__ import annotations

import os
import time

# CI 上放宽阈值的倍率。读取环境变量是为了让 PR/nightly 用同一份代码。
# 本地默认 1.0；CI 在 workflow 里 export 为 2.0。
_THRESHOLD_MULT = float(os.environ.get("CLAWCODEX_CI_THRESHOLD_MULT", "1"))


class TestStage6Perf:
    """性能回归检测 — 冷启动 / 导入 / 管线响应预算。"""

    # ── CLI budgets ──────────────────────────────────────────────

    def test_cli_help_import_time(self):
        """--help 快速路径不应导入重型模块（budget: < 3s，CI x N）。"""
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
        """python -m src.cli --help 子进程启动时间（budget: < 5s，CI x N）。"""
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
        """Conversation 模块导入不应拉入重型依赖（budget: < 2s，CI x N）。"""
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
        """Agent loop 全套模块导入（query + engine + transitions）不应超过 3s。

        测量从零冷启动导入完整 agent loop 路径的时间。
        这是用户第一条消息进入 LLM 调用前必须等待的模块解析成本。
        """
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
        """工具执行路径导入（registry + build_tool + dispatch）不应超过 2s。

        测量从零冷启动导入工具注册表、工具构建器和查找函数的
        时间。这是工具执行管线的最小冷启动成本。
        """
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
        """REPL 类导入（不实例化）不应超过 5s，CI x N。

        测量从零冷启动 ``from src.repl import ClawcodexREPL`` 的时间。
        重型依赖（Session、providers、tools）在首次实例化时通过
        ``_load_heavy_runtime()`` 加载，因此本测试只覆盖类导入成本。
        本地 budget 设在 prompt_toolkit + rich + completer 基线 ~4.5s 之上
        留 0.5s 余量。
        """
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
        """REPL 重型运行时栈冷启动不应超过 6.5s，CI x N。

        ``_load_heavy_runtime()`` 在 ``ClawcodexREPL.__init__`` 首次调用时
        触发，集中 import Session、provider、tool registry、CommandSystem
        的 hook 关联 + CostTracker、HistoryLog 等 ~22 项重型依赖（QueryEngine
        与 command_system 已延后到首条 prompt，见 B+C 优化后）。这是用
        户从命令行启动到 REPL 第一个 prompt 之前的最大单点开销。

        本地基线 ~5.3s（实测，B+C 优化后），留 1.2s 余量设 6.5s budget。
        CI 上 ×2 阈值。失败说明有模块被无谓地拉到 ``_load_heavy_runtime``
        顶部 import 了。

        跳过条件：依赖 ``httpx``（Codex OAuth 的传递依赖）缺失时跳过，
        避免在没有安装完整依赖的环境下产生 false negative。
        """
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
        """REPL 端到端冷启动（import + _load_heavy_runtime + 构造 + 渲染）不应超过 7s。

        模拟用户从命令行启动 clawcodex 到 REPL 第一个 prompt 出现的
        完整路径：包括 ``ClawcodexREPL`` 类导入、``_load_heavy_runtime``
        全栈加载、``ClawcodexREPL.__init__`` 实例化（provider、Session、
        tool registry Stage A）以及 ``_print_startup_header`` 渲染。

        不实际进入 prompt_toolkit 主循环（那受 terminal 状态影响不可控），
        但 ``__init__`` 已完成 REPL 启动期全部同步重型工作。

        Stage B（extension tools + persisted agent tools + workflow tool）
        runs on a daemon thread (see ``build_default_registry`` in
        ``clawcodex_ext/tool_system/defaults.py``); it does not block this
        measurement. ``test_build_default_registry_defer_fast`` below
        guards the Stage A / Stage B split by ensuring defer mode returns
        within the synchronous budget.

        B+C 优化后：CommandSystem 延后到首次 ``/``，QueryEngine 延后到
        首条非 slash 提示；本地基线 ~6.0s（实测），留 1.0s 余量设 7s budget。
        CI 上 ×2 阈值。

        跳过条件：依赖 ``httpx``（Codex OAuth 的传递依赖）缺失时跳过。
        """
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
        """``build_default_registry(defer_extended_tools=True)`` 应立即返回。

        Stage A（51 个静态工具 + Agent + ToolSearch）必须同步注册。
        Stage B（extension tools + persisted agent tools + workflow tool）
        推到 daemon 线程，所以 ``build_default_registry`` 调用必须 < 1000ms
        返回（budget）。否则说明 Stage B 又被错误地阻塞到 sync path。

        这个测试和 ``test_repl_first_prompt_end_to_end`` 一起验证 deferred
        Stage B 方案的两个不变量：
        1. ``build_default_registry(defer_extended_tools=True)`` 快速返回
        2. ``ClawcodexREPL.__init__`` 端到端时间 < 9s

        如果 Stage A 的 ``ALL_STATIC_TOOLS`` 列表未来被改大（例如新增重型
        工具类），本测试会触发 Stage A 的回归 —— 因为 Stage A 是同步的，
        defer 路径只对 Stage B 起作用。
        """
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
