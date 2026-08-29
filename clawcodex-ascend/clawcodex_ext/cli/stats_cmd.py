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

# pylint: disable=no-name-in-module
"""tool/skill stats CLI subcommand.

Usage::

 clawcodex stats # Aggregate summary
 clawcodex stats --kind tool # Tool calls only
 clawcodex stats --kind skill # Skill calls only
 clawcodex stats --limit 10 # Ten most recent details
 clawcodex stats --agent orchestrator # One agent's statistics
 clawcodex stats --json # JSON output
"""

from __future__ import annotations

import argparse
import json
import sys

from clawcodex_ext.cli.subcommand_registry import register
from clawcodex_ext.tool_stats import get_stats, get_summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clawcodex stats",
        description="工具/Skill 调用统计",
    )
    p.add_argument(
        "--kind",
        choices=["tool", "skill"],
        default=None,
        help='筛选类型："tool" 或 "skill"',
    )
    p.add_argument("--agent", default=None, help="按 agent_id 筛选")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="返回最近 N 条明细（0=全部）",
    )
    p.add_argument("--json", action="store_true", help="JSON 格式输出")
    return p


@register("stats")
def run_stats_command(args: list[str]) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(args)

    kind = parsed.kind
    agent = parsed.agent

    if parsed.limit > 0:
        # Detailed records
        rows = get_stats(kind=kind, agent_id=agent, limit=parsed.limit)
        if parsed.json:
            json.dump({"kind": kind, "agent_id": agent, "rows": rows}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            if not rows:
                print("(no records)")
                return 0
            print(f"{'#':>4}  {'Type':<6}  {'Name':<24}  {'Dur(ms)':>8}  {'OK':<4}  {'Error':<20}  {'Agent':<16}")
            print("-" * 90)
            for i, r in enumerate(rows, 1):
                name = r.get("tool") or r.get("skill") or "?"
                err = (r.get("error") or "")[:20]
                print(
                    f"{i:>4}  {r.get('kind', '?'):<6}  {name:<24}  "
                    f"{r.get('dur_ms', 0):>8.1f}  "
                    f"{'✅' if r.get('ok') else '❌':<4}  {err:<20}  "
                    f"{r.get('agent_id', '?'):<16}"
                )
    else:
        # Aggregate summary
        summary = get_summary(kind=kind, agent_id=agent)
        if parsed.json:
            json.dump(summary, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            title = "工具" if kind == "tool" else "Skill" if kind == "skill" else "工具/Skill"
            agent_label = f" (agent={agent})" if agent else ""
            print(f"\n📊 {title} 调用统计{agent_label}:")
            print(f"   总调用: {summary['total_calls']}")
            print(f"   平均耗时: {summary['avg_duration_ms']} ms")
            print(f"   错误率: {summary['error_rate']:.1%}")
            if summary["by_name"]:
                print("\n   按名称分布:")
                for name, count in summary["by_name"].items():
                    ok_count = summary["by_name_ok"].get(name, 0)
                    rate = ok_count / count if count else 0
                    print(f"     {name:<24}  {count:>4} 次  成功率 {rate:.0%}")

    return 0
