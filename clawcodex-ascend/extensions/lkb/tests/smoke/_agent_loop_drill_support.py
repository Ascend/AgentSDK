#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=E0611,W1510

"""SMOKE-LKB-001: "The Agent Loop" release drill with a real AgentLoop and scripted LLM.

End-to-end smoke test (spec §13.7) driving the LKB Plan Graph through the
REAL agent loop (``run_query_as_agent_loop``) with a scripted provider that
mocks the LLM's returns.  Each drill phase is one or more *scenes*: a user
message enters the loop, the scripted provider (the mocked LLM) emits
TaskCreate/TaskUpdate/TaskList tool calls, and the test asserts the
resulting Board state through internal readers (Store snapshot / read model
/ audit events).

Design notes:

* Task ids are server-generated (``T-<hex8>``); the script never hardcodes
  them.  Tool-input factories are evaluated lazily when the provider pops
  the action, resolving ids from the current session projection
  (``ctx.tasks``) by subject prefix (``"T0"`` .. ``"T8"``).
* Multiple executors use independent ``ToolContext`` objects. The Claim
  race uses two OS processes, each with its own Repository and real loop;
  the final recovery read uses a third process with a fresh Session.
* Recovery uses the public ``/lkb revalidate`` command, never the domain
  application service.
* T4's "demo script fails first, then passes" is modelled faithfully at
  the file level (two real ``subprocess`` runs, both outputs kept).
* The six phases run once in a single serial smoke test.  Assertions stay
  adjacent to each phase so failures remain local, while expensive earlier
  phases (especially the multi-process claim race) are not replayed for
  every later checkpoint.
* The smoke does NOT reproduce the drill's "2000+ words per chapter"
  content requirement; it writes real deliverable files and binds their
  sha256 into task metadata instead.

Run:  pytest extensions/lkb/tests/smoke/test_agent_loop_drill.py -q -s
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import queue
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clawcodex_ext.command_system.lkb_command import _lkb_call
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.query.agent_loop_compat import run_query_as_agent_loop
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.types.messages import UserMessage

from lkb.ascii_board import render_board
from lkb.graph_types import NodeRef
from lkb.read_model import build_board_view
from lkb.repository import JsonFileLkbRepository

# ── drill script data ─────────────────────────────────────────────────

# prefix, subject, description
_TASKS = [
    (
        "T0",
        "收集参考资料并撰写术语表",
        "收集 Agent Loop 参考资料，产出 docs/drill/glossary.md",
    ),
    ("T1", "制定文章大纲与章节契约", "产出 docs/drill/outline.md：各章主题、边界、术语用法"),
    ("T2", "撰写第一章「什么是 Agent 的 Loop」", "写入 docs/drill/chapter-1.md"),
    ("T3", "撰写第二章「Loop 中的状态与上下文」", "写入 docs/drill/chapter-2.md"),
    (
        "T4",
        "编写并实际运行 Agent Loop 演示脚本",
        "docs/drill/demo_loop.py；首跑允许失败，修复后通过",
    ),
    ("T5", "全文交叉引用与术语一致性校对", "产出 docs/drill/cross-ref-report.md"),
    ("T6", "撰写第三章「多 Agent 协作与任务调度」", "写入 docs/drill/chapter-3.md"),
    ("T7", "发布前整体审校", "产出 docs/drill/review.md 审校批准"),
    (
        "T8",
        "汇总终稿并撰写发布说明",
        "产出 docs/drill/final.md 与 docs/drill/release-notes.md",
    ),
]

# (dependent, prerequisite)
_DEPS = [
    ("T2", "T1"),
    ("T3", "T1"),
    ("T4", "T2"),
    ("T5", "T2"),
    ("T5", "T3"),
    ("T5", "T0"),
    ("T6", "T3"),
    ("T7", "T4"),
    ("T7", "T5"),
    ("T7", "T6"),
    ("T8", "T7"),
]

_OUTLINE_V1 = """# 《Agent 的 Loop》大纲 v1

## 章节契约
- 第一章：什么是 Agent 的 Loop —— 定义感知-决策-行动循环。
- 第二章：Loop 中的状态与上下文 —— 状态载体与上下文窗口管理。
- 第三章：多 Agent 协作与任务调度 —— 领取、依赖与发布节奏。

## 术语用法
- Loop：单次「观察 → 决策 → 工具调用 → 观察结果」迭代。
- Context：模型每轮可见的消息序列。
"""

_OUTLINE_V2 = """# 《Agent 的 Loop》大纲 v2（迟到的变更）

## 章节契约
- 第一章：什么是 Agent 的 Loop —— 定义感知-决策-行动循环（不变）。
- 第二章：Loop 中的工具调用与权限 —— 由「状态与上下文」改写：工具
  调用的门禁、权限模式与拒绝处理。
- 第三章：多 Agent 协作与任务调度 —— 领取、依赖与发布节奏（不变）。

## 术语用法
- Loop：单次「观察 → 决策 → 工具调用 → 观察结果」迭代。
- Tool Gate：工具执行前的统一校验点。
"""

_GLOSSARY = """# 术语表

- Agent：在 Loop 中调用工具完成目标的执行体。
- Loop：观察-决策-行动-再观察的迭代。
- Claim：对任务的原子领取。
- Revalidate：上游变更后按依赖顺序确认任务仍然有效。
"""

_CROSS_REF = """# 交叉引用与术语一致性校对报告

- 三章对「Loop」的定义一致（见术语表）。
- 第一章引用的演示脚本与 docs/drill/demo_loop.py 行为一致（run2 通过）。
- 结论：无冲突术语，交叉引用闭环。
"""

_REVIEW_V1 = """# 发布前整体审校（v1）

- 大纲契约：符合。
- 章节完整性：符合。
- 演示脚本：run2 通过。
- 结论：APPROVED，允许发布。
"""

_REVIEW_V2 = """# 发布前整体审校（v2，变更后重审）

- 大纲契约 v2：符合（第二章已改写为工具调用与权限）。
- 演示脚本：run3 复跑通过。
- 结论：APPROVED，允许恢复发布。
"""

_FINAL_V1 = """# 《Agent 的 Loop》终稿 v1

（首轮终稿按 outline v1 汇总三章正文，见 chapter-1/2/3。）
"""

_RELEASE_NOTES_V1 = """# 发布说明 v1

- 首次发布：T0~T8 全部完成。
"""

_CHAPTER_2_V2 = """# 第二章 Loop 中的工具调用与权限

Agent Loop 的每次工具调用都要经过统一门禁。门禁依据工具 schema、当前权限模式、
任务所有权和依赖状态决定调用能否执行；被拒绝的调用必须原样返回给模型，供下一轮
观察和修正。权限拒绝不能被文字声明绕过，也不能通过重复调用掩盖。

LKB 把领取、依赖、完成与失效传播纳入同一状态模型，使工具调用结果、任务状态
和发布条件保持一致。本章是迟到变更后的真实 v2 交付物。
"""

_CROSS_REF_V2 = """# 交叉引用与术语一致性校对报告 v2

- 第二章已改为「Loop 中的工具调用与权限」，与 outline v2 一致。
- 三章对「Loop」和「Tool Gate」的用法与术语表一致。
- 演示脚本 run3 通过，全文交叉引用已按 v2 复核。
- 结论：v2 交叉引用闭环。
"""

_FINAL_V2 = """# 《Agent 的 Loop》终稿 v2

终稿按 outline v2 汇总三章正文；第二章现为「Loop 中的工具调用与权限」。
"""

_RELEASE_NOTES_V2 = """# 发布说明 v2

- 迟到变更：第二章由状态与上下文改为工具调用与权限。
- T1 重开后按拓扑序恢复，所有受影响交付物均已重写或重新确认。
- 独立任务 T0 未受影响。
"""

_DEMO_V1 = '''"""Agent Loop 演示（v1，故意失败版）：状态收敛检查。"""

import sys

state = {"goal": "write-chapter", "done": False}
for step in range(3):
    # BUG: The completion branch forgets to set done, so the loop cannot converge.
    if step == 2 and state["goal"] == "write-chapter":
        pass
print("loop finished without convergence")
sys.exit("demo failed: loop did not converge")
'''

_DEMO_V2 = '''"""Agent Loop 演示（v2，修复版）：状态收敛检查。"""

import sys

state = {"goal": "write-chapter", "done": False}
for step in range(3):
    if step == 2 and state["goal"] == "write-chapter":
        state["done"] = True
        break
if not state["done"]:
    sys.exit("demo failed: loop did not converge")
print(f"LOOP OK: converged after {step + 1} steps")
'''


def _chapter(title: str) -> str:
    return (
        f"# {title}\n\n"
        "Agent 的 Loop 是「观察 → 决策 → 行动 → 再观察」的迭代过程：\n"
        "每一轮迭代都把上一步的工具结果重新纳入上下文，使下一步决策\n"
        "始终基于最新状态。任务的领取、依赖与完成状态通过逻辑看板\n"
        "（LKB）统一管理，保证多 Agent 协作时语义一致、拒绝可解释。\n"
        "本章正文在冒烟中以真实写入的文件代替完整长文。\n"
    )


# ── scripted provider (the mocked LLM) ────────────────────────────────


def _chat_response(content: str, finish: str, tool_uses=None) -> ChatResponse:
    return ChatResponse(
        content=content,
        model="drill-mock-llm",
        usage={"input_tokens": 1, "output_tokens": 1},
        finish_reason=finish,
        tool_uses=tool_uses,
    )


def _messages_have_tool_result(messages, expected_tool_use_id: str) -> bool:
    for message in messages or ():
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        blocks = content if isinstance(content, list) else (content,)
        for block in blocks:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            tool_use_id = block.get("tool_use_id") if isinstance(block, dict) else getattr(block, "tool_use_id", None)
            if block_type == "tool_result" and tool_use_id == expected_tool_use_id:
                return True
    return False


def _advertised_tool_names(tools) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        if isinstance(tool, dict):
            name = tool.get("name")
        else:
            name = getattr(tool, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


class DrillProvider:
    """Queue-based scripted LLM — the mock of the model's returns.

    Each ``chat`` pops one action:

    * ``("tool", name, input_or_factory)`` → ``finish_reason="tool_use"``.
      Factories are evaluated lazily at pop time so task ids resolve from
      the *current* session projection, never hardcoded.
    * ``("say", text)`` → ``finish_reason="stop"`` (ends the loop run).

    ``before_tool`` (optional hook, e.g. a process ``Barrier``) fires
    right before a tool_use response is returned — used to synchronize
    concurrent claims from two independent loop runs.
    """

    def __init__(self, label, actions, transcript, before_tool=None):
        self.label = label
        self.actions = list(actions)
        self.transcript = transcript
        self.before_tool = before_tool
        self.calls = 0
        self.awaiting_tool_use_id: str | None = None

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.awaiting_tool_use_id is not None:
            assert _messages_have_tool_result(messages, self.awaiting_tool_use_id), (
                f"{self.label}: AgentLoop did not feed tool_result {self.awaiting_tool_use_id!r} back to the provider"
            )
            self.awaiting_tool_use_id = None
        if not self.actions:
            raise AssertionError(f"{self.label}: scripted action queue exhausted without an explicit stop")
        action = self.actions.pop(0)
        if action[0] == "say":
            self.transcript.append({"actor": self.label, "kind": "say", "text": action[1]})
            return _chat_response(action[1], "stop")
        _, name, spec = action
        advertised = _advertised_tool_names(tools)
        assert name in advertised, f"{self.label}: {name} was not advertised to the provider: {advertised}"
        if callable(spec):
            spec = spec()
        if self.before_tool is not None:
            self.before_tool()
        self.transcript.append({"actor": self.label, "kind": "tool_call", "tool": name, "input": spec})
        tool_use_id = f"drill-{self.label}-{self.calls}"
        self.awaiting_tool_use_id = tool_use_id
        return _chat_response(
            "",
            "tool_use",
            tool_uses=[{"id": tool_use_id, "name": name, "input": spec}],
        )

    def chat_stream(self, messages, tools=None, **kwargs):
        return iter(())

    def chat_stream_response(self, messages, tools=None, **kwargs):
        raise NotImplementedError


# ── fixture + helpers ─────────────────────────────────────────────────


@pytest.fixture
def drill(tmp_path: Path, tmp_home: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fresh Board + 4 executor contexts + shared transcript."""
    from clawcodex_ext.feature_gate import get_registry
    import lkb.repository as repository_module

    monkeypatch.setitem(get_registry()._overrides, "LKB_PLAN_GRAPH", True)

    ws = tmp_path / "ws"
    (ws / "docs" / "drill").mkdir(parents=True)
    repo = JsonFileLkbRepository(home=tmp_home)
    monkeypatch.setattr(repository_module, "_repository_singleton", repo)
    board_id = repo.resolve_board(ws, session_id="drill-session").board_id

    def _ctx(agent_id: str) -> ToolContext:
        ctx = ToolContext(workspace_root=ws)
        ctx.agent_id = agent_id
        ctx.session_id = "drill-session"
        return ctx

    d = {
        "repo": repo,
        "board_id": board_id,
        "home": tmp_home,
        "ws": ws,
        "ctxs": {a: _ctx(a) for a in ("agent-a", "agent-b", "agent-c", "operator")},
        "names": {},
        "transcript": [],
    }
    # One shared registry; the provider only feeds the (unused) Agent tool.
    d["registry"] = build_default_registry(
        provider=DrillProvider("bootstrap", [], d["transcript"]), load_agent_tools=False
    )
    return d


def run_scene(d, actor: str, actions: list, user_msg: str, *, before_tool=None, max_turns=None):
    """One drill scene: a user message enters the real agent loop; the
    scripted provider (mocked LLM) drives the queued tool calls.
    """
    ctx = d["ctxs"][actor]
    provider = DrillProvider(actor, actions, d["transcript"], before_tool=before_tool)
    events: list = []
    result = asyncio.run(
        run_query_as_agent_loop(
            initial_messages=[UserMessage(content=user_msg)],
            provider=provider,
            tool_registry=d["registry"],
            tool_context=ctx,
            system_prompt="你是 LKB 发布演练的多 Agent 团队成员，全程使用任务工具管理计划与执行状态。",
            max_turns=max_turns or (len(actions) + 2),
            on_event=events.append,
        )
    )
    assert not provider.actions, f"{actor}: unconsumed scripted actions: {provider.actions}"
    assert provider.awaiting_tool_use_id is None, f"{actor}: final tool result was never observed"
    expected_tool_results = sum(1 for action in actions if action[0] == "tool")
    actual_tool_results = sum(1 for event in events if getattr(event, "kind", "") == "tool_result")
    assert actual_tool_results == expected_tool_results, (
        f"{actor}: expected {expected_tool_results} tool results, got {actual_tool_results}"
    )
    d["transcript"].append(
        {
            "kind": "scene_end",
            "actor": actor,
            "user_msg": user_msg,
            "turns": result.num_turns,
            "tool_results": [
                {
                    # tool_result events carry an empty tool_name; recover it
                    # from the matching tool_use event via tool_use_id.
                    "tool": _tool_names_by_use_id(events).get(getattr(e, "tool_use_id", None), "?"),
                    "is_error": bool(getattr(e, "is_error", False)),
                    "output": str(getattr(e, "tool_output", "") or getattr(e, "error", "") or "")[:300],
                }
                for e in events
                if getattr(e, "kind", "") == "tool_result"
            ],
        }
    )
    return result, events


def _tool_names_by_use_id(events) -> dict:
    return {
        getattr(e, "tool_use_id", None): getattr(e, "tool_name", "?")
        for e in events
        if getattr(e, "kind", "") == "tool_use"
    }


def _tool_outputs(events, tool: str) -> list[dict]:
    names = _tool_names_by_use_id(events)
    outputs: list[dict] = []
    for event in events:
        if getattr(event, "kind", "") != "tool_result":
            continue
        if names.get(getattr(event, "tool_use_id", None)) != tool:
            continue
        raw = getattr(event, "tool_output", None)
        if isinstance(raw, dict):
            outputs.append(raw)
            continue
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                outputs.append(parsed)
    return outputs


def _child_repository(home: str) -> JsonFileLkbRepository:
    """Build a process-local repository and feature registry."""
    os.environ["HOME"] = home
    os.environ["USERPROFILE"] = home
    os.environ["CLAWCODEX_HOME"] = home
    os.environ["CLAWCODEX_FEATURE_LKB_PLAN_GRAPH"] = "1"

    import lkb.repository as repository_module
    from clawcodex_ext.feature_gate import get_registry

    get_registry()._overrides["LKB_PLAN_GRAPH"] = True
    repository = JsonFileLkbRepository(home=Path(home))
    repository_module._repository_singleton = repository
    return repository


def _claim_loop_process(args: tuple, result_queue) -> None:
    """Run one real AgentLoop claim in an isolated process."""
    home, workspace, session_id, plan_id, task_id, actor, barrier = args
    try:
        _child_repository(home)
        context = ToolContext(workspace_root=Path(workspace))
        context.agent_id = actor
        context.session_id = session_id
        context.lkb_plan_id = plan_id
        transcript: list[dict] = []
        provider = DrillProvider(
            actor,
            [
                ("tool", "TaskUpdate", {"taskId": task_id, "owner": actor}),
                ("say", "已如实记录系统返回"),
            ],
            transcript,
            before_tool=lambda: barrier.wait(timeout=120),
        )
        registry = build_default_registry(provider=provider, load_agent_tools=False)
        events: list = []
        asyncio.run(
            run_query_as_agent_loop(
                initial_messages=[UserMessage(content="两个子代理同时领取 T1")],
                provider=provider,
                tool_registry=registry,
                tool_context=context,
                system_prompt="你是 LKB 发布演练的子代理。",
                max_turns=4,
                on_event=events.append,
            )
        )
        denied = _scene_denied(events)
        outputs = _tool_outputs(events, "TaskUpdate")
        result_queue.put(
            {
                "actor": actor,
                "decision": "denied" if denied else "committed",
                "outputs": outputs,
                "transcript": transcript,
                "actions_remaining": len(provider.actions),
                "awaiting_tool_result": provider.awaiting_tool_use_id is not None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - marshal child failure to parent
        result_queue.put({"actor": actor, "error": f"{type(exc).__name__}: {exc}"})


def _restart_reader_process(args: tuple, result_queue) -> None:
    """Open the completed Board from a new process, session and ToolContext."""
    home, workspace, session_id, expected_board_id, plan_id = args
    try:
        repository = _child_repository(home)
        context = ToolContext(workspace_root=Path(workspace))
        context.agent_id = "restart-reader"
        context.session_id = session_id
        context.lkb_plan_id = plan_id
        transcript: list[dict] = []
        provider = DrillProvider(
            "restart-reader",
            [("tool", "TaskList", {}), ("say", "重启读取完成")],
            transcript,
        )
        registry = build_default_registry(provider=provider, load_agent_tools=False)
        events: list = []
        asyncio.run(
            run_query_as_agent_loop(
                initial_messages=[UserMessage(content="新会话读取当前 LKB Board")],
                provider=provider,
                tool_registry=registry,
                tool_context=context,
                system_prompt="只通过公开工具读取恢复后的 LKB Board。",
                max_turns=4,
                on_event=events.append,
            )
        )
        task_list = _tool_outputs(events, "TaskList")
        board_text = str(_lkb_call("board", SimpleNamespace(tool_context=context)).value)
        resolved_board_id = repository.resolve_board(
            Path(workspace),
            session_id=session_id,
        ).board_id
        envelope = repository._get_store(expected_board_id).load()
        snapshot = repository.load_snapshot(expected_board_id)
        result_queue.put(
            {
                "resolved_board_id": resolved_board_id,
                "plan_id": context.lkb_plan_id,
                "task_list": task_list,
                "board_text": board_text,
                "states": {
                    ref.id: node.state
                    for ref, node in snapshot.nodes.items()
                    if ref.graph == plan_id and ref.kind == "task"
                },
                "derived": {
                    ref.id: (node.payload or {}).get("derived_status")
                    for ref, node in snapshot.nodes.items()
                    if ref.graph == plan_id and ref.kind == "task"
                },
                "claim_count": len(envelope.claims),
                "store_revision": envelope.store_revision,
                "plan_revision": snapshot.graphs[plan_id].revision,
                "session_binding": envelope.board.get("session_plan_bindings", {}).get(session_id),
                "plan_session_ids": envelope.graphs[plan_id].get("plan", {}).get("session_ids", []),
                "actions_remaining": len(provider.actions),
                "awaiting_tool_result": provider.awaiting_tool_use_id is not None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - marshal child failure to parent
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})


def _multiprocessing_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context("spawn")


def _collect_process_results(processes, result_queue, expected: int) -> list[dict]:
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=180)
        assert not process.is_alive(), f"child process {process.pid} did not exit"
        assert process.exitcode == 0, f"child process {process.pid} exited {process.exitcode}"
    results: list[dict] = []
    for _ in range(expected):
        try:
            results.append(result_queue.get(timeout=10))
        except queue.Empty as exc:
            raise AssertionError(f"only {len(results)}/{expected} child results arrived") from exc
    return results


def _tid(d, prefix: str) -> str:
    """Resolve the server-generated task id from the session projection."""
    if prefix not in d["names"]:
        ctx = d["ctxs"]["agent-a"]
        for tid, task in ctx.tasks.items():
            if str(task.get("subject", "")).startswith(f"{prefix} "):
                d["names"][prefix] = tid
                break
        else:
            raise AssertionError(f"{prefix} not in projection: {sorted(ctx.tasks)}")
    return d["names"][prefix]


def _plan_id(d) -> str:
    for ctx in d["ctxs"].values():
        pid = getattr(ctx, "lkb_plan_id", None)
        if pid:
            return pid
    raise AssertionError("plan id not resolved yet (run phase 1 first)")


def _env(d):
    return d["repo"]._get_store(d["board_id"]).load()


def _view(d):
    return build_board_view(_env(d), plan_id=_plan_id(d))


def _revision_pair(d) -> tuple[int, int]:
    snapshot = d["repo"].load_snapshot(d["board_id"])
    return snapshot.store_revision, snapshot.graphs[_plan_id(d)].revision


def _badge(d, prefix: str) -> str:
    tid = _tid(d, prefix)
    for row in _view(d).rows:
        if row.task_id == tid:
            return row.badge
    raise AssertionError(f"{prefix} ({tid}) not in board view")


def _node(d, prefix: str):
    snap = d["repo"].load_snapshot(d["board_id"])
    return snap.nodes[NodeRef(_plan_id(d), "task", _tid(d, prefix))]


def _audit(d) -> list:
    return list(_env(d).events)


def _denials(d, code: str, prefix: str | None = None) -> list:
    """Audit denial events carrying `code` (optionally scoped to a task)."""
    out = []
    for e in _audit(d):
        if e.get("decision") != "denied" or code not in str(e.get("reason", "")):
            continue
        if prefix is not None and not str(e.get("subject_ref", "")).endswith(f":task:{_tid(d, prefix)}"):
            continue
        out.append(e)
    return out


def _scene_denied(events, tool: str = "TaskUpdate") -> bool:
    names = _tool_names_by_use_id(events)
    return any(
        getattr(e, "kind", "") == "tool_result"
        and getattr(e, "is_error", False)
        and names.get(getattr(e, "tool_use_id", None)) == tool
        for e in events
    )


def _write(d, rel: str, content: str) -> Path:
    path = d["ws"] / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _lkb_mutation(d, command: str, *, actor: str):
    """Execute a mutating public ``/lkb`` command and record its result."""
    text = str(_lkb_call(command, SimpleNamespace(tool_context=d["ctxs"][actor])).value)
    committed = text.startswith("Revalidated ")
    operation = command.split(maxsplit=1)[0]
    d["transcript"].append(
        {
            "kind": "lkb-command",
            "op": operation,
            "command": command,
            "actor": actor,
            "decision": "committed" if committed else "denied",
            "reason": text,
        }
    )
    return SimpleNamespace(decision="committed" if committed else "denied", reason=text)


def _revalidate(d, prefix: str, *, actor: str = "agent-a"):
    return _lkb_mutation(d, f"revalidate {_tid(d, prefix)}", actor=actor)


def _run_demo(d, out_name: str) -> subprocess.CompletedProcess:
    """Really run the demo script and keep the full output on disk."""
    proc = subprocess.run(
        [sys.executable, "demo_loop.py"],
        cwd=d["ws"] / "docs" / "drill",
        capture_output=True,
        text=True,
        timeout=60,
    )
    (d["ws"] / "docs" / "drill" / out_name).write_text(
        f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
        encoding="utf-8",
    )
    return proc


def _lkb_board_text(d, actor: str = "agent-a") -> str:
    return str(_lkb_call("board", SimpleNamespace(tool_context=d["ctxs"][actor])).value)


def _print_board(d, title: str, capsys) -> None:
    if capsys is not None:
        capsys.readouterr()
        print(f"\n=== {title} ===")
        print(render_board(_view(d), width=110))


# Scene action helpers (closures capture the drill dict at scene build time)


def _act_claim(d, prefix: str, actor: str):
    return ("tool", "TaskUpdate", lambda: {"taskId": _tid(d, prefix), "owner": actor})


def _act_start(d, prefix: str):
    return ("tool", "TaskUpdate", lambda: {"taskId": _tid(d, prefix), "status": "in_progress"})


def _act_complete(d, prefix: str, metadata: dict | None = None):
    def _input():
        inp = {"taskId": _tid(d, prefix), "status": "completed"}
        if metadata is not None:
            inp["metadata"] = metadata
        return inp

    return ("tool", "TaskUpdate", _input)


def _act_dep(d, dep: str, prereq: str):
    return (
        "tool",
        "TaskUpdate",
        lambda: {"taskId": _tid(d, dep), "addBlockedBy": [_tid(d, prereq)]},
    )


def _act_meta(d, prefix: str, metadata: dict):
    return ("tool", "TaskUpdate", lambda: {"taskId": _tid(d, prefix), "metadata": metadata})


# ── phase 1: build the task graph ─────────────────────────────────────


__all__ = [name for name in globals() if not name.startswith("__")]
