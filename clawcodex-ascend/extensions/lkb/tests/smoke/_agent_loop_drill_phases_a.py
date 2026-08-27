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

# Split drill phases intentionally share the support module's private scenario namespace.
# ruff: noqa: F403, F405
# pylint: disable=C0207,C3001,E0602,W0401

"""Phases one through four of the Agent Loop release drill."""

from tests.smoke._agent_loop_drill_support import *


def _phase1(d) -> None:
    actions = []
    for prefix, subject, desc in _TASKS:

        def _create(s=f"{prefix} {subject}", de=desc):
            return {"subject": s, "description": de}

        actions.append(("tool", "TaskCreate", _create))
    for dep, prereq in _DEPS:
        actions.append(_act_dep(d, dep, prereq))
    actions.append(("tool", "TaskList", {}))
    actions.append(("say", "任务图已建立：T0/T1 就绪，T2~T8 阻塞"))
    _, events = run_scene(
        d,
        "agent-a",
        actions,
        "阶段一·规划建图：建立 T0~T8 及依赖",
        max_turns=len(actions) + 2,
    )
    task_list_outputs = _tool_outputs(events, "TaskList")
    assert len(task_list_outputs) == 1
    d["phase1_task_list"] = task_list_outputs[0]


def _assert_phase1(d, capsys) -> None:
    assert _badge(d, "T0") == "ready"
    assert _badge(d, "T1") == "ready"
    for p in ("T2", "T3", "T4", "T5", "T6", "T7", "T8"):
        assert _badge(d, p) == "blocked", f"{p}: expected blocked, got {_badge(d, p)}"
    # TaskList projection carries the compact lkb summary.
    tasks = d["ctxs"]["agent-a"].tasks
    assert tasks[_tid(d, "T5")]["lkb"]["derivedStatus"] == "blocked"
    assert set(tasks[_tid(d, "T5")]["lkb"]["activeBlockers"]) == {
        _tid(d, "T2"),
        _tid(d, "T3"),
        _tid(d, "T0"),
    }
    board_summary = d["phase1_task_list"]["lkbBoard"]
    assert board_summary["boardId"] == d["board_id"]
    assert board_summary["planId"] == _plan_id(d)
    assert board_summary["counts"] == {
        "ready": 2,
        "running": 0,
        "blocked": 7,
        "needsRecheck": 0,
    }
    # /lkb board command renders the same board.
    board_text = _lkb_board_text(d)
    assert f"LKB BOARD: {d['ws'].name} /" in board_text
    assert "Ready 2 | Running 0 | Blocked 7 | Recheck 0 | Issues 7" in board_text
    assert all(len(line) <= 110 for line in board_text.splitlines())
    _print_board(d, "Phase 1: initial blocked graph", capsys)


# ── phase 2: protection-mechanism probes ──────────────────────────────


def _phase2(d) -> None:
    plan = lambda: _plan_id(d)  # noqa: E731 - terse closure

    # (a) cycle probe: T1 depends_on T8 must be denied.
    _, events = run_scene(
        d,
        "agent-a",
        [
            (
                "tool",
                "TaskUpdate",
                lambda: {"taskId": _tid(d, "T1"), "addBlockedBy": [_tid(d, "T8")]},
            ),
            ("say", "已如实记录系统返回"),
        ],
        "探针(a)：尝试让 T1 依赖 T8",
    )
    assert _scene_denied(events), "cycle probe must surface a tool-level denial"
    assert _denials(d, "dependency_cycle", "T1"), "audit must record dependency_cycle"
    env = _env(d)
    assert not any(
        e.get("type") == "depends_on"
        and str(e.get("source", "")) == f"{plan()}:task:{_tid(d, 'T1')}"
        and str(e.get("target", "")) == f"{plan()}:task:{_tid(d, 'T8')}"
        for e in env.edges.values()
    ), "denied edge must not exist in the graph"

    # (b) concurrent claim of T1 from two independent processes, each
    # running its own real AgentLoop + ToolContext + Repository.
    mp_ctx = _multiprocessing_context()
    barrier = mp_ctx.Barrier(2, timeout=120)
    result_queue = mp_ctx.Queue()
    actors = ("agent-a", "agent-b")
    processes = [
        mp_ctx.Process(
            target=_claim_loop_process,
            args=(
                (
                    str(d["home"]),
                    str(d["ws"]),
                    f"claim-process-{actor}",
                    _plan_id(d),
                    _tid(d, "T1"),
                    actor,
                    barrier,
                ),
                result_queue,
            ),
        )
        for actor in actors
    ]
    results = _collect_process_results(processes, result_queue, len(processes))
    assert all("error" not in result for result in results), results
    assert all(result["actions_remaining"] == 0 for result in results)
    assert all(not result["awaiting_tool_result"] for result in results)
    for result in results:
        d["transcript"].extend(result["transcript"])
        d["transcript"].append(
            {
                "kind": "scene_end",
                "actor": result["actor"],
                "user_msg": "探针(b)：两个独立进程同时领取 T1",
                "turns": 2,
                "tool_results": [
                    {
                        "tool": "TaskUpdate",
                        "is_error": result["decision"] == "denied",
                        "output": json.dumps(output, ensure_ascii=False)[:300],
                    }
                    for output in result["outputs"]
                ],
            }
        )
    winners = [result["actor"] for result in results if result["decision"] == "committed"]
    losers = [result["actor"] for result in results if result["decision"] == "denied"]
    assert len(winners) == 1 and len(losers) == 1, (
        f"expected exactly one claim winner, got winners={winners} losers={losers}"
    )
    d["t1_owner"] = winners[0]
    assert _denials(d, "already_claimed", "T1")
    active = [
        c
        for c in _env(d).claims.values()
        if c.get("status") == "active" and str(c.get("task_ref", "")) == f"{plan()}:task:{_tid(d, 'T1')}"
    ]
    assert len(active) == 1

    # (c) claim a currently blocked task.
    _, events = run_scene(
        d,
        "agent-c",
        [
            ("tool", "TaskUpdate", lambda: {"taskId": _tid(d, "T2"), "owner": "agent-c"}),
            ("say", "已如实记录系统返回"),
        ],
        "探针(c)：领取仍被阻塞的 T2",
    )
    assert _scene_denied(events)
    assert _denials(d, "blocked", "T2")
    assert d["ctxs"]["agent-c"].tasks[_tid(d, "T2")]["lkb"]["activeBlockers"] == [_tid(d, "T1")]

    # (d) start a task that was never claimed.
    _, events = run_scene(
        d,
        "agent-c",
        [
            ("tool", "TaskUpdate", lambda: {"taskId": _tid(d, "T0"), "status": "in_progress"}),
            ("say", "已如实记录系统返回"),
        ],
        "探针(d)：直接开始未领取的 T0",
    )
    assert _scene_denied(events)
    assert _denials(d, "owner_required", "T0")


def _assert_phase2(d, capsys) -> None:
    assert d["t1_owner"] in ("agent-a", "agent-b")
    _print_board(d, "Phase 2: probes done (T1 claimed, denials audited)", capsys)


# ── phase 3: parallel execution, round 1 ──────────────────────────────


def _phase3(d) -> None:
    owner = d["t1_owner"]

    # T1: start → write outline → complete with deliverable metadata.
    outline = _write(d, "docs/drill/outline.md", _OUTLINE_V1)
    _, events = run_scene(
        d,
        owner,
        [
            _act_start(d, "T1"),
            _act_complete(
                d,
                "T1",
                metadata={
                    "deliverable": {
                        "path": "docs/drill/outline.md",
                        "sha256": _sha256(outline),
                    }
                },
            ),
            ("say", "T1 完成"),
        ],
        "阶段三：T1 启动、写入大纲并完成",
    )
    assert not _scene_denied(events)
    assert _badge(d, "T2") == "ready"
    assert _badge(d, "T3") == "ready"

    # T2 (agent-b): claim → start → write chapter-1 → complete via a single
    # atomic patch (status + deliverable metadata in one TaskUpdate).
    run_scene(
        d,
        "agent-b",
        [_act_claim(d, "T2", "agent-b"), _act_start(d, "T2"), ("say", "T2 进行中")],
        "阶段三：agent-b 领取并启动 T2",
    )
    ch1 = _write(d, "docs/drill/chapter-1.md", _chapter("第一章 什么是 Agent 的 Loop"))
    before_patch = _revision_pair(d)
    _, events = run_scene(
        d,
        "agent-b",
        [
            _act_complete(
                d,
                "T2",
                metadata={"deliverable": {"path": "docs/drill/chapter-1.md", "sha256": _sha256(ch1)}},
            ),
            ("say", "T2 完成"),
        ],
        "阶段三：agent-b 完成 T2（patch_task 原子提交）",
    )
    assert not _scene_denied(events)
    assert _node(d, "T2").state == "completed"
    after_patch = _revision_pair(d)
    assert after_patch == (before_patch[0] + 1, before_patch[1] + 1), (
        f"atomic TaskUpdate must publish one Store/Plan revision: {before_patch} -> {after_patch}"
    )

    # T3 (agent-c): claim → start → write chapter-2 → metadata → complete.
    run_scene(
        d,
        "agent-c",
        [_act_claim(d, "T3", "agent-c"), _act_start(d, "T3"), ("say", "T3 进行中")],
        "阶段三：agent-c 领取并启动 T3",
    )
    ch2 = _write(d, "docs/drill/chapter-2.md", _chapter("第二章 Loop 中的状态与上下文"))
    _, events = run_scene(
        d,
        "agent-c",
        [
            _act_meta(
                d,
                "T3",
                {"deliverable": {"path": "docs/drill/chapter-2.md", "sha256": _sha256(ch2)}},
            ),
            _act_complete(d, "T3"),
            ("say", "T3 完成"),
        ],
        "阶段三：agent-c 完成 T3",
    )
    assert not _scene_denied(events)
    assert _node(d, "T3").state == "completed"

    # T4 (agent-c): claim → start, keep a real failed run, fix the demo,
    # run it successfully, then complete with both run hashes recorded.
    run_scene(
        d,
        "agent-c",
        [
            _act_claim(d, "T4", "agent-c"),
            _act_start(d, "T4"),
            ("say", "T4 进行中"),
        ],
        "阶段三：agent-c 领取并启动 T4",
    )
    _write(d, "docs/drill/demo_loop.py", _DEMO_V1)
    run1 = _run_demo(d, "run1.txt")
    assert run1.returncode != 0, "first demo run must fail (drill allows it)"
    _write(d, "docs/drill/demo_loop.py", _DEMO_V2)
    run2 = _run_demo(d, "run2.txt")
    assert run2.returncode == 0, f"fixed demo must pass: {run2.stderr}"
    assert "LOOP OK" in run2.stdout
    _, events = run_scene(
        d,
        "agent-c",
        [
            _act_complete(
                d,
                "T4",
                metadata={
                    "deliverable": {
                        "path": "docs/drill/demo_loop.py",
                        "sha256": _sha256(d["ws"] / "docs" / "drill" / "demo_loop.py"),
                        "run1_sha256": _sha256(d["ws"] / "docs" / "drill" / "run1.txt"),
                        "run2_sha256": _sha256(d["ws"] / "docs" / "drill" / "run2.txt"),
                    }
                },
            ),
            ("say", "T4 完成"),
        ],
        "阶段三：修复后完成 T4",
    )
    assert not _scene_denied(events)
    assert _node(d, "T4").state == "completed"


def _assert_phase3(d, capsys) -> None:
    for rel in ("run1.txt", "run2.txt"):
        assert (d["ws"] / "docs" / "drill" / rel).is_file(), rel
    run1_text = (d["ws"] / "docs" / "drill" / "run1.txt").read_text(encoding="utf-8")
    assert "did not converge" in run1_text
    _print_board(d, "Phase 3: T1~T4 completed (T4 after fail→fix→pass)", capsys)


# ── phase 4: first release ────────────────────────────────────────────


def _phase4(d) -> None:
    # T0 (independent, agent-a).
    run_scene(
        d,
        "agent-a",
        [_act_claim(d, "T0", "agent-a"), _act_start(d, "T0"), ("say", "T0 进行中")],
        "阶段四：agent-a 领取并启动 T0",
    )
    glos = _write(d, "docs/drill/glossary.md", _GLOSSARY)
    run_scene(
        d,
        "agent-a",
        [
            _act_meta(
                d,
                "T0",
                {"deliverable": {"path": "docs/drill/glossary.md", "sha256": _sha256(glos)}},
            ),
            _act_complete(d, "T0"),
            ("say", "T0 完成"),
        ],
        "阶段四：agent-a 完成 T0",
    )

    # T5 (agent-b) — depends on T2, T3, T0 (all completed by now).
    run_scene(
        d,
        "agent-b",
        [_act_claim(d, "T5", "agent-b"), _act_start(d, "T5"), ("say", "T5 进行中")],
        "阶段四：agent-b 领取并启动 T5",
    )
    xref = _write(d, "docs/drill/cross-ref-report.md", _CROSS_REF)
    _, events = run_scene(
        d,
        "agent-b",
        [
            _act_complete(
                d,
                "T5",
                metadata={
                    "deliverable": {
                        "path": "docs/drill/cross-ref-report.md",
                        "sha256": _sha256(xref),
                    }
                },
            ),
            ("say", "T5 完成"),
        ],
        "阶段四：agent-b 完成 T5（patch_task 原子提交）",
    )
    assert not _scene_denied(events)

    # T6 (agent-c).
    run_scene(
        d,
        "agent-c",
        [_act_claim(d, "T6", "agent-c"), _act_start(d, "T6"), ("say", "T6 进行中")],
        "阶段四：agent-c 领取并启动 T6",
    )
    ch3 = _write(d, "docs/drill/chapter-3.md", _chapter("第三章 多 Agent 协作与任务调度"))
    run_scene(
        d,
        "agent-c",
        [
            _act_meta(
                d,
                "T6",
                {"deliverable": {"path": "docs/drill/chapter-3.md", "sha256": _sha256(ch3)}},
            ),
            _act_complete(d, "T6"),
            ("say", "T6 完成"),
        ],
        "阶段四：agent-c 完成 T6",
    )

    # T7 (agent-a): write the review, then complete with its artifact hash.
    review = _write(d, "docs/drill/review.md", _REVIEW_V1)
    _, events = run_scene(
        d,
        "agent-a",
        [
            _act_claim(d, "T7", "agent-a"),
            _act_start(d, "T7"),
            _act_complete(
                d,
                "T7",
                metadata={
                    "deliverable": {
                        "path": "docs/drill/review.md",
                        "sha256": _sha256(review),
                    }
                },
            ),
            ("say", "T7 完成"),
        ],
        "阶段四：完成发布前整体审校 T7",
    )
    assert not _scene_denied(events)

    # T8 (agent-a) — depends on T7.
    run_scene(
        d,
        "agent-a",
        [_act_claim(d, "T8", "agent-a"), _act_start(d, "T8"), ("say", "T8 进行中")],
        "阶段四：agent-a 领取并启动 T8",
    )
    final = _write(d, "docs/drill/final.md", _FINAL_V1)
    notes = _write(d, "docs/drill/release-notes.md", _RELEASE_NOTES_V1)
    _, events = run_scene(
        d,
        "agent-a",
        [
            _act_complete(
                d,
                "T8",
                metadata={
                    "deliverable": {
                        "path": "docs/drill/final.md",
                        "sha256": _sha256(final),
                        "release_notes_sha256": _sha256(notes),
                    }
                },
            ),
            ("say", "T8 完成，第一轮发布就绪"),
        ],
        "阶段四：agent-a 完成 T8（第一轮发布）",
    )
    assert not _scene_denied(events)


def _assert_phase4(d, capsys) -> None:
    view = _view(d)
    assert all(r.badge == "verified" for r in view.rows), [(r.task_id, r.badge) for r in view.rows]
    assert view.summary.issues == 0
    # ≥3 distinct executors claimed tasks (drill rule: multi-agent).
    owners = {str(c.get("owner_ref", "")).split(":")[-1] for c in _env(d).claims.values()}
    assert {"agent-a", "agent-b", "agent-c"} <= owners
    assert "outline v1" in (d["ws"] / "docs" / "drill" / "final.md").read_text(encoding="utf-8")
    assert "变更发布" not in (d["ws"] / "docs" / "drill" / "release-notes.md").read_text(encoding="utf-8")
    _print_board(d, "Phase 4: first release — all verified", capsys)


# ── phase 5: the late change ──────────────────────────────────────────
