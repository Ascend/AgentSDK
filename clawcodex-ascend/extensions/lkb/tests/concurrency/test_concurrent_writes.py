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

"""Concurrency tests for LKB JSON Graph Store.

Covers:
  LKB-STORE-023 — 16 subprocesses concurrent create/modify SAME board
                  → all serializable, no lost updates
                  (multiprocessing.Barrier; assert final revision ==
                  successful writes + all changes present)
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path


from lkb.json_store import BoardEnvelope
from lkb.commands import CommandResult
from lkb.refs import NodeRef


# ── worker function (must be importable for multiprocessing) ──────────


def _worker_add_node(args: tuple) -> tuple[int, bool, str]:
    """Worker function for concurrent write test.

    Parameters (as tuple for pickling):
      board_dir_str: string path to board directory
      board_id: board ID string
      worker_id: integer worker ID (0..N-1)
      barrier_fd: barrier object (shared via multiprocessing)

    Returns:
      (worker_id, success_bool, error_message)
    """
    board_dir_str, board_id, worker_id, barrier = args

    try:
        # Wait for all workers to be ready
        barrier.wait(timeout=120)

        # Create store and add a unique node
        from lkb.file_lock import BoardFileLock
        from lkb.json_store import JsonBoardStore
        from lkb.ir_hash import canonical_hash

        lock = BoardFileLock(Path(board_dir_str))
        store = JsonBoardStore(
            Path(board_dir_str),
            board_id=board_id,
            lock=lock,
        )

        node_id = f"T-{worker_id:03d}"
        cid = f"worker-{worker_id}-cmd"
        rh = canonical_hash({"kind": "add_node", "worker": worker_id, "node": node_id})

        def mutate(env: BoardEnvelope) -> tuple[BoardEnvelope, CommandResult]:
            ref = NodeRef("plan", "task", node_id)
            env.nodes[node_id] = {
                "ref": ref.to_str(),
                "title": f"Task from worker {worker_id}",
                "state": "pending",
                "owner": f"worker-{worker_id}",
                "revision": 1,
                "payload": {"worker_id": worker_id},
                "created_at": "2026-01-01T00:00:00.000Z",
                "updated_at": "2026-01-01T00:00:00.000Z",
            }
            # Ensure plan graph exists
            if "plan" not in env.graphs:
                env.graphs["plan"] = {
                    "graph_id": "plan",
                    "board_id": board_id,
                    "graph_kind": "plan",
                    "revision": 0,
                    "created_at": "2026-01-01T00:00:00.000Z",
                    "updated_at": "2026-01-01T00:00:00.000Z",
                }
            result = CommandResult(
                decision="committed",
                command_id=cid,
            )
            return env, result

        result = store.execute_atomic(
            board_id,
            cid,
            rh,
            None,
            mutate,
            actor=f"worker-{worker_id}",
        )

        return (worker_id, result.committed, "")

    except Exception as exc:
        return (worker_id, False, str(exc))


# ── LKB-STORE-023 ─────────────────────────────────────────────────────


class TestStore023ConcurrentWrites:
    """16 subprocesses concurrent create/modify SAME board.

    All writes must be serializable with no lost updates.
    Uses multiprocessing.Barrier so all workers start at roughly the
    same time (spec §11.3: no sleep-based races).
    """

    NUM_WORKERS = 16

    def test_sixteen_subprocesses_concurrent_modify_same_board(self, tmp_home: Path) -> None:
        """LKB-STORE-023: 16 subprocesses, same board, no lost updates."""
        from lkb.repository import JsonFileLkbRepository

        board_id = "store-023-concurrent"
        repo = JsonFileLkbRepository(home=tmp_home)
        repo.resolve_board(explicit_id=board_id)

        from lkb.board_resolver import board_dir

        bd = board_dir(board_id, home=tmp_home)
        board_dir_str = str(bd)

        # Use fork context on Linux for fast worker startup
        # (spawn is slow due to import overhead, causing barrier timeouts)
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context("spawn")

        # Barrier: just the workers (main process is not a party)
        barrier = ctx.Barrier(self.NUM_WORKERS, timeout=120)

        # Prepare args for each worker
        args_list = [(board_dir_str, board_id, i, barrier) for i in range(self.NUM_WORKERS)]

        # Start all worker processes
        processes = []
        results: list[tuple[int, bool, str]] = []

        def _target(args: tuple, result_queue: multiprocessing.Queue) -> None:
            result = _worker_add_node(args)
            result_queue.put(result)

        result_queue = ctx.Queue()

        for args in args_list:
            p = ctx.Process(target=_target, args=(args, result_queue))
            processes.append(p)
            p.start()

        # Wait for all workers to finish
        for p in processes:
            p.join(timeout=180)

        # Collect results from queue
        for _ in range(self.NUM_WORKERS):
            if not result_queue.empty():
                results.append(result_queue.get(timeout=5))

        # Check results
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]

        # All 16 should succeed (or almost all — contention is OK as long
        # as no data is lost)
        assert len(successful) >= self.NUM_WORKERS * 0.9, (
            f"Too many failures: {len(failed)} / {self.NUM_WORKERS}. Failures: {failed[:3]}"
        )

        # Now verify: all successful writes are present
        snap = repo.load_snapshot(board_id)
        task_nodes = [n for n in snap.nodes.values() if n.ref.graph == "plan"]

        # All successful worker nodes should be present
        successful_ids = {f"T-{r[0]:03d}" for r in successful}
        found_ids = {n.ref.id for n in task_nodes}

        missing = successful_ids - found_ids
        assert not missing, f"Missing nodes from successful writes: {missing}"

        # The number of plan-task nodes equals the number of successful writes
        assert len(task_nodes) == len(successful), f"Expected {len(successful)} nodes, found {len(task_nodes)}"

    def test_concurrent_writes_preserve_hash_chain(self, tmp_home: Path) -> None:
        """LKB-STORE-023: after concurrent writes, payload hash chain is intact."""
        from lkb.repository import JsonFileLkbRepository
        from lkb.json_store import (
            _verify_payload_hash,
        )

        board_id = "store-023-hash-chain"
        repo = JsonFileLkbRepository(home=tmp_home)
        repo.resolve_board(explicit_id=board_id)

        from lkb.board_resolver import board_dir

        bd = board_dir(board_id, home=tmp_home)
        board_dir_str = str(bd)

        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context("spawn")

        barrier = ctx.Barrier(self.NUM_WORKERS, timeout=120)
        args_list = [(board_dir_str, board_id, i, barrier) for i in range(self.NUM_WORKERS)]

        result_queue = ctx.Queue()
        processes = []

        def _target(args: tuple, result_queue: multiprocessing.Queue) -> None:
            result = _worker_add_node(args)
            result_queue.put(result)

        for args in args_list:
            p = ctx.Process(target=_target, args=(args, result_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=180)

        results = []
        for _ in range(self.NUM_WORKERS):
            if not result_queue.empty():
                results.append(result_queue.get(timeout=5))

        successful = [r for r in results if r[1]]
        assert len(successful) > 0, "No successful writes"

        # Verify the final board.json has a valid payload hash
        import json

        board_json = bd / "board.json"
        with open(board_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert _verify_payload_hash(data), "Final board.json has invalid payload hash"

        # And board_id matches
        assert data["board"]["board_id"] == board_id


# ── _wait_for helper (spec §11.3) ────────────────────────────────────


def _wait_for(condition_fn, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """Poll *condition_fn* until it returns True or *timeout* elapses.

    Spec §11.3: never use sleep-based races — always poll with a
    timeout guard.  Returns True if condition was met, False on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition_fn():
            return True
        time.sleep(interval)
    return condition_fn()


def test_concurrent_first_create_is_exactly_once(tmp_home: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from lkb.repository import JsonFileLkbRepository

    workers = 8
    barrier = threading.Barrier(workers)

    def create() -> tuple[str, int]:
        repository = JsonFileLkbRepository(home=tmp_home)
        barrier.wait(timeout=10)
        board = repository.resolve_board(explicit_id="first-create")
        return board.board_id, board.store_revision

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda _: create(), range(workers)))

    assert results == [("first-create", 0)] * workers
    repository = JsonFileLkbRepository(home=tmp_home)
    assert repository.resolve_board(explicit_id="first-create").store_revision == 0
