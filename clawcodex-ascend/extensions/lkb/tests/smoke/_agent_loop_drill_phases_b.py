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
# pylint: disable=E0602,W0401,W0614

"""Phases five and six of the Agent Loop release drill."""

from ._agent_loop_drill_support import *


def _phase5(d) -> None:
    # operator reopens T1 (TaskUpdate status=pending → reopen_task), then
    # records the change request in task metadata.  The transition reason is
    # also supplied explicitly so the invalidation audit is self-explanatory.
    _, events = run_scene(
        d,
        "operator",
        [
            (
                "tool",
                "TaskUpdate",
                lambda: {
                    "taskId": _tid(d, "T1"),
                    "status": "pending",
                    "reason": "第二章需求发生变更",
                },
            ),
            _act_meta(
                d,
                "T1",
                {
                    "change_request": "第二章由「Loop 中的状态与上下文」改写为「Loop 中的工具调用与权限」",
                    "requested_by": "operator",
                },
            ),
            ("say", "变更已下达：重开 T1"),
        ],
        "阶段五：迟到的变更 — operator 重开 T1",
    )
    assert not _scene_denied(events)
    # Apply the change by updating the outline artifact.
    _write(d, "docs/drill/outline.md", _OUTLINE_V2)


def _assert_phase5(d, capsys) -> None:
    t1 = _node(d, "T1")
    assert t1.state == "pending"
    assert t1.owner is None, "reopen must clear the owner (commit 08a624ee contract)"
    assert "工具调用与权限" in str(t1.payload or {})
    # Downstream completed tasks keep base=completed, gain needs_recheck.
    for p in ("T2", "T3", "T4", "T5", "T6", "T7", "T8"):
        node = _node(d, p)
        assert node.state == "completed", f"{p} base status changed: {node.state}"
        assert (node.payload or {}).get("derived_status") == "needs_recheck", (
            f"{p} expected needs_recheck, got {(node.payload or {}).get('derived_status')!r}"
        )
    # Independent T0 is untouched.
    t0 = _node(d, "T0")
    assert t0.state == "completed"
    assert not (t0.payload or {}).get("derived_status")
    # The changed outline invalidates the old chapter/final; recovery must
    # rewrite them rather than merely clearing needs_recheck.
    assert "工具调用与权限" in (d["ws"] / "docs" / "drill" / "outline.md").read_text(encoding="utf-8")
    assert "状态与上下文" in (d["ws"] / "docs" / "drill" / "chapter-2.md").read_text(encoding="utf-8")
    assert "outline v1" in (d["ws"] / "docs" / "drill" / "final.md").read_text(encoding="utf-8")
    # Invalidation propagation is audited.
    assert any(e.get("type") == "invalidation_propagation" for e in _audit(d))
    _print_board(d, "Phase 5: T1 reopened — downstream needs_recheck, T0 intact", capsys)


# ── phase 6: recover the release ──────────────────────────────────────


def _phase6(d) -> None:
    # Probe: skipping un-revalidated upstream must be denied (topo gate).
    r = _revalidate(d, "T8")
    assert r.decision == "denied"
    assert "upstream not verified" in (r.reason or "")

    # Re-claim T1 (reopen cleared the owner) → start → complete against
    # outline v2, recording the new artifact revision in metadata.
    outline_v2 = d["ws"] / "docs" / "drill" / "outline.md"
    _, events = run_scene(
        d,
        "agent-a",
        [
            _act_claim(d, "T1", "agent-a"),
            _act_start(d, "T1"),
            _act_complete(
                d,
                "T1",
                metadata={
                    "deliverable": {
                        "path": "docs/drill/outline.md",
                        "sha256": _sha256(outline_v2),
                    }
                },
            ),
            ("say", "T1 重新完成"),
        ],
        "阶段六：重领 T1 并按 outline v2 重新完成",
    )
    assert not _scene_denied(events)

    # Recover every downstream task in topological order. Files that must
    # change are rewritten before revalidation; unchanged deliverables are
    # explicitly revalidated against the new upstream state.
    r = _revalidate(d, "T2", actor="agent-b")
    assert r.decision == "committed", f"revalidate T2: {r.reason}"

    chapter2 = _write(d, "docs/drill/chapter-2.md", _CHAPTER_2_V2)
    r = _revalidate(d, "T3", actor="agent-c")
    assert r.decision == "committed", f"revalidate T3: {r.reason}"
    _, events = run_scene(
        d,
        "agent-c",
        [
            _act_meta(
                d,
                "T3",
                {"deliverable": {"path": "docs/drill/chapter-2.md", "sha256": _sha256(chapter2)}},
            ),
            ("say", "T3 v2 交付物已更新"),
        ],
        "阶段六：重写第二章并更新交付物哈希",
    )
    assert not _scene_denied(events)

    run3 = _run_demo(d, "run3.txt")
    assert run3.returncode == 0
    r = _revalidate(d, "T4", actor="agent-c")
    assert r.decision == "committed", f"revalidate T4: {r.reason}"
    _, events = run_scene(
        d,
        "agent-c",
        [
            _act_meta(
                d,
                "T4",
                {
                    "deliverable": {
                        "path": "docs/drill/demo_loop.py",
                        "sha256": _sha256(d["ws"] / "docs" / "drill" / "demo_loop.py"),
                        "run3_sha256": _sha256(d["ws"] / "docs" / "drill" / "run3.txt"),
                    }
                },
            ),
            ("say", "T4 演示脚本已复跑"),
        ],
        "阶段六：复跑演示脚本并更新运行哈希",
    )
    assert not _scene_denied(events)

    cross_ref = _write(d, "docs/drill/cross-ref-report.md", _CROSS_REF_V2)
    r = _revalidate(d, "T5", actor="agent-b")
    assert r.decision == "committed", f"revalidate T5: {r.reason}"
    _, events = run_scene(
        d,
        "agent-b",
        [
            _act_meta(
                d,
                "T5",
                {
                    "deliverable": {
                        "path": "docs/drill/cross-ref-report.md",
                        "sha256": _sha256(cross_ref),
                    }
                },
            ),
            ("say", "T5 v2 校对报告已更新"),
        ],
        "阶段六：按新第二章重做全文交叉引用校对",
    )
    assert not _scene_denied(events)

    r = _revalidate(d, "T6", actor="agent-c")
    assert r.decision == "committed", f"revalidate T6: {r.reason}"

    review2 = _write(d, "docs/drill/review-v2.md", _REVIEW_V2)
    r = _revalidate(d, "T7")
    assert r.decision == "committed", f"revalidate T7: {r.reason}"
    _, events = run_scene(
        d,
        "agent-a",
        [
            _act_meta(
                d,
                "T7",
                {
                    "deliverable": {
                        "path": "docs/drill/review-v2.md",
                        "sha256": _sha256(review2),
                    }
                },
            ),
            ("say", "T7 v2 审校完成"),
        ],
        "阶段六：完成 v2 发布前审校",
    )
    assert not _scene_denied(events)

    final = _write(d, "docs/drill/final.md", _FINAL_V2)
    notes = _write(d, "docs/drill/release-notes.md", _RELEASE_NOTES_V2)
    r = _revalidate(d, "T8")
    assert r.decision == "committed", f"revalidate T8: {r.reason}"
    _, events = run_scene(
        d,
        "agent-a",
        [
            _act_meta(
                d,
                "T8",
                {
                    "deliverable": {
                        "path": "docs/drill/final.md",
                        "sha256": _sha256(final),
                        "release_notes_sha256": _sha256(notes),
                    }
                },
            ),
            ("say", "T8 v2 终稿与发布说明已更新"),
        ],
        "阶段六：汇总 v2 终稿并更新发布说明",
    )
    assert not _scene_denied(events)


def _assert_phase6(d, capsys) -> None:
    view = _view(d)
    assert all(r.badge == "verified" for r in view.rows), [(r.task_id, r.badge) for r in view.rows]
    assert view.summary.issues == 0

    # Cross-session restart recovery: a new process owns a fresh Repository,
    # Session, ToolContext and AgentLoop. It resolves the Board by workspace,
    # calls TaskList, and renders /lkb board without any old Context cache.
    before_restart_envelope = _env(d)
    before_restart = d["repo"].load_snapshot(d["board_id"])
    mp_ctx = _multiprocessing_context()
    result_queue = mp_ctx.Queue()
    restart_process = mp_ctx.Process(
        target=_restart_reader_process,
        args=(
            (
                str(d["home"]),
                str(d["ws"]),
                "drill-restart-session",
                d["board_id"],
                _plan_id(d),
            ),
            result_queue,
        ),
    )
    restart = _collect_process_results([restart_process], result_queue, 1)[0]
    assert "error" not in restart, restart
    assert restart["actions_remaining"] == 0
    assert not restart["awaiting_tool_result"]
    assert restart["resolved_board_id"] == d["board_id"]
    assert restart["plan_id"] == _plan_id(d)
    assert restart["store_revision"] == before_restart_envelope.store_revision + 1
    assert restart["plan_revision"] == before_restart.graphs[_plan_id(d)].revision + 1
    assert restart["session_binding"] == _plan_id(d)
    assert "drill-restart-session" in restart["plan_session_ids"]
    assert restart["claim_count"] == len(before_restart_envelope.claims)
    assert set(restart["states"]) == {_tid(d, p) for p, *_rest in _TASKS}
    assert set(restart["states"].values()) == {"completed"}
    assert not any(restart["derived"].values())
    assert len(restart["task_list"]) == 1
    restart_list = restart["task_list"][0]
    assert restart_list["lkbBoard"]["boardId"] == d["board_id"]
    assert restart_list["lkbBoard"]["planId"] == _plan_id(d)
    assert len(restart_list["tasks"]) == len(_TASKS)
    assert f"LKB BOARD: {d['ws'].name} /" in restart["board_text"]
    assert "Ready 0 | Running 0 | Blocked 0 | Recheck 0 | Issues 0" in restart["board_text"]

    # Full audit chain: every probe denial + invalidation + revalidates.
    for code in (
        "dependency_cycle",
        "already_claimed",
        "blocked",
        "owner_required",
    ):
        assert _denials(d, code), f"audit missing denial code {code}"
    assert any(e.get("type") == "invalidation_propagation" for e in _audit(d))
    revalidated = [
        t
        for t in d["transcript"]
        if t.get("kind") == "lkb-command" and t.get("op") == "revalidate" and t.get("decision") == "committed"
    ]
    assert len(revalidated) == 7, f"expected 7 revalidates (T2..T8), got {len(revalidated)}"
    denied_revalidates = [
        t
        for t in d["transcript"]
        if t.get("kind") == "lkb-command" and t.get("op") == "revalidate" and t.get("decision") == "denied"
    ]
    assert len(denied_revalidates) == 1

    # All real deliverables (incl. both failing and passing demo outputs).
    for rel in (
        "outline.md",
        "glossary.md",
        "chapter-1.md",
        "chapter-2.md",
        "chapter-3.md",
        "cross-ref-report.md",
        "review.md",
        "review-v2.md",
        "final.md",
        "release-notes.md",
        "demo_loop.py",
        "run1.txt",
        "run2.txt",
        "run3.txt",
    ):
        assert (d["ws"] / "docs" / "drill" / rel).is_file(), f"missing deliverable {rel}"

    chapter2_text = (d["ws"] / "docs" / "drill" / "chapter-2.md").read_text(encoding="utf-8")
    final_text = (d["ws"] / "docs" / "drill" / "final.md").read_text(encoding="utf-8")
    notes_text = (d["ws"] / "docs" / "drill" / "release-notes.md").read_text(encoding="utf-8")
    cross_ref_text = (d["ws"] / "docs" / "drill" / "cross-ref-report.md").read_text(encoding="utf-8")
    assert "工具调用与权限" in chapter2_text
    assert "状态与上下文" not in chapter2_text.splitlines()[0]
    assert "终稿 v2" in final_text and "工具调用与权限" in final_text
    assert "发布说明 v2" in notes_text and "迟到变更" in notes_text
    assert "校对报告 v2" in cross_ref_text and "工具调用与权限" in cross_ref_text

    for prefix, rel in (
        ("T3", "chapter-2.md"),
        ("T5", "cross-ref-report.md"),
        ("T8", "final.md"),
    ):
        deliverable = (_node(d, prefix).payload or {})["metadata"]["deliverable"]
        assert deliverable["sha256"] == _sha256(d["ws"] / "docs" / "drill" / rel)
    t8_deliverable = (_node(d, "T8").payload or {})["metadata"]["deliverable"]
    assert t8_deliverable["release_notes_sha256"] == _sha256(d["ws"] / "docs" / "drill" / "release-notes.md")

    # Every tool-level protection probe surfaced its refusal to the loop.
    denied_tool_results = [
        r for s in d["transcript"] if s.get("kind") == "scene_end" for r in s["tool_results"] if r["is_error"]
    ]
    assert len(denied_tool_results) == 4, (
        f"expected exactly 4 recorded tool-level denials, got {len(denied_tool_results)}"
    )
    assert denied_revalidates[0]["command"] == f"revalidate {_tid(d, 'T8')}"

    final_board = _lkb_board_text(d)
    assert f"LKB BOARD: {d['ws'].name} /" in final_board
    assert "Ready 0 | Running 0 | Blocked 0 | Recheck 0 | Issues 0" in final_board
    assert "\x1b[" not in final_board
    assert all(len(line) <= 110 for line in final_board.splitlines())
    _print_board(d, "Phase 6: release recovered — all verified, issues 0", capsys)
    if capsys is not None:
        capsys.readouterr()
        print("\n=== Drill transcript (abbrev) ===")
        for entry in d["transcript"]:
            kind = entry.get("kind")
            if kind == "tool_call":
                print(f"[{entry['actor']}] {entry['tool']} {entry['input']}")
            elif kind == "lkb-command":
                print(f"[{entry['actor']}] /lkb {entry['command']} -> {entry['decision']} {entry.get('reason') or ''}")
            elif kind == "scene_end":
                errs = [r for r in entry["tool_results"] if r["is_error"]]
                if errs:
                    print(f"[{entry['actor']}] denials: {[e['output'][:80] for e in errs]}")
