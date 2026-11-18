# -*- coding: utf-8 -*-
#
# This file is part of the AgentSDK project.
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
#
import asyncio
import random
import time
from typing import Any

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

SIGNAL_CANCEL_TIMEOUT = 10.0


class TrainRouter:
    """Routes training requests to executor instances managed by TrainManager."""

    _router = None
    _active_executors = {}
    _active_train_refs = {}
    _shutdown_hard_limit = None
    _session_drain_timeout = None
    _rollout_manager_enabled = False

    def __init__(self, train_manager) -> None:
        """
        Initialize TrainRouter with a TrainManager actor handle.

        Args:
            train_manager: Ray actor handle for the TrainManager.
        """
        self.train_manager = train_manager

    @classmethod
    async def create(cls, shutdown_hard_limit=None, session_drain_timeout=None,
                     rollout_manager_enabled=None) -> "TrainRouter":
        """
        Get or create the singleton TrainRouter instance.

        Args:
            shutdown_hard_limit: Hang-guard seconds for the drain wait, read from
                the train config; None keeps the module default.
            session_drain_timeout: Business budget (seconds) for waiting the
                rollout task to drain in-flight sessions; None keeps the default.
            rollout_manager_enabled: Whether the unified rollout entry is on; when
                off the driver never queries or waits the rollout task ref.

        Returns:
            The singleton TrainRouter instance.
        """
        if shutdown_hard_limit is not None:
            cls._shutdown_hard_limit = float(shutdown_hard_limit)
        if session_drain_timeout is not None:
            cls._session_drain_timeout = float(session_drain_timeout)
        if rollout_manager_enabled is not None:
            cls._rollout_manager_enabled = bool(rollout_manager_enabled)
        if cls._router is None:
            from aura.trainer.train_manager import get_or_create_train_manager

            train_manager = await get_or_create_train_manager()
            cls._router = TrainRouter(train_manager)
            logger.info("Train router created.")
        return cls._router

    async def train(self, name, *args, **kwargs) -> Any:
        infer_instance = await self.train_manager.get_instance.remote(name)
        executor = random.choice(infer_instance.executor_list)
        key = id(executor)
        self._active_executors[key] = executor
        train_ref = executor.execute_method.remote("fit")
        self._active_train_refs[key] = train_ref
        try:
            return await train_ref
        finally:
            self._active_train_refs.pop(key, None)
            self._active_executors.pop(key, None)

            if self._rollout_manager_enabled:
                try:
                    await asyncio.to_thread(self._wait_executor_rollout_drained, executor)
                except Exception as exp:
                    logger.warning(f"wait rollout shutdown failed: {type(exp).__name__}")

    @classmethod
    def _wait_executor_rollout_drained(cls, executor):
        """Drain, release and wait one executor's rollout resources (blocking)."""
        rollout_refs, rollout_workers = cls._collect_rollout_handles([executor])
        if cls._wait_rollout_drain(cls._start_rollout_drain(rollout_workers, "normal"), "normal"):
            cls._release_rollout_workers(rollout_workers, "normal")
        cls._wait_rollout_drained(rollout_refs, "normal")

    @classmethod
    def cancel_active(cls, signal_name="SIGTERM", timeout=SIGNAL_CANCEL_TIMEOUT):
        """Cancel every executor still running ``fit`` (best-effort, blocking)."""
        executors = list(cls._active_executors.values())
        rollout_refs, rollout_workers = cls._collect_rollout_handles(executors)
        drain_refs = cls._start_rollout_drain(rollout_workers, signal_name)
        refs = []
        for executor in executors:
            try:
                refs.append(executor.cancel.remote(signal_name))
            except Exception as exp:
                logger.warning(f"cancel executor failed: {type(exp).__name__}")
        if not refs:
            return 0
        import ray

        try:
            ray.get(refs, timeout=timeout)
        except Exception as exp:
            logger.warning(
                f"[AURA_ROLLOUT] router.signal.cancel.timeout signal={signal_name} executors={len(refs)} "
                f"error={type(exp).__name__}"
            )
        if cls._wait_rollout_drain(drain_refs, signal_name):
            cls._release_rollout_workers(rollout_workers, signal_name)
        cls._wait_train_drained(signal_name, rollout_refs)
        return len(refs)

    @classmethod
    def _collect_rollout_handles(cls, executors):
        """Fetch the rollout handles registered by the running trainers."""
        if not executors:
            return [], []
        import ray

        proxies = []
        for executor in executors:
            try:
                proxies.append(executor.get_rollout_future.remote())
            except Exception as exp:
                logger.warning(f"get rollout future failed: {type(exp).__name__}")
        rollout_refs = []
        rollout_workers = []
        for proxy in proxies:
            try:
                payload = ray.get(proxy, timeout=SIGNAL_CANCEL_TIMEOUT)
            except Exception as exp:
                logger.warning(f"resolve rollout future failed: {type(exp).__name__}")
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("ref") is not None:
                rollout_refs.append(payload["ref"])
            if payload.get("worker") is not None:
                rollout_workers.append(payload["worker"])
        return rollout_refs, rollout_workers

    @classmethod
    def _start_rollout_drain(cls, rollout_workers, signal_name="normal"):
        """Kick off the worker-side drain without blocking."""
        from aura.trainer.rollout.rollout_manager import DEFAULT_DRAIN_TIMEOUT

        if not rollout_workers:
            return []
        budget = float(cls._session_drain_timeout or DEFAULT_DRAIN_TIMEOUT)
        refs = []
        for worker in rollout_workers:
            try:
                refs.append(worker.drain_sessions.remote(budget))
            except Exception as exp:
                logger.warning(f"start drain sessions failed: {type(exp).__name__}")
        logger.warning(
            f"[AURA_ROLLOUT] router.signal.cancel.drain.begin signal={signal_name} "
            f"workers={len(rollout_workers)} submitted={len(refs)} budget={budget}s"
        )
        return refs

    @classmethod
    def _wait_rollout_drain(cls, drain_refs, signal_name="normal") -> bool:
        """Wait for the driver-initiated session drain to finish."""
        import ray

        from aura.trainer.rollout.rollout_manager import (
            DEFAULT_DRAIN_TIMEOUT,
            SHUTDOWN_WAIT_MARGIN,
        )

        if not drain_refs:
            return False
        budget = float(cls._session_drain_timeout or DEFAULT_DRAIN_TIMEOUT)
        started = time.time()
        try:
            aborted = ray.get(drain_refs, timeout=budget + SHUTDOWN_WAIT_MARGIN)
        except Exception as exp:
            if type(exp).__name__ in ("ActorDiedError", "RayActorError"):
                logger.info(
                    f"[AURA_ROLLOUT] router.signal.cancel.drain.skipped signal={signal_name} "
                    f"reason=worker_released elapsed={time.time() - started:.3f}s"
                )

                return signal_name != "normal"
            logger.warning(
                f"[AURA_ROLLOUT] router.signal.cancel.drain.failed signal={signal_name} "
                f"error={type(exp).__name__} elapsed={time.time() - started:.3f}s"
            )
            return False
        logger.warning(
            f"[AURA_ROLLOUT] router.signal.cancel.drain.done signal={signal_name} aborted={aborted} "
            f"elapsed={time.time() - started:.3f}s"
        )
        return True

    @classmethod
    def _release_rollout_workers(cls, rollout_workers, signal_name="normal"):
        """Release worker-owned resources after the drain finished."""
        import ray

        from aura.trainer.rollout.resource_ledger import STATUS_RELEASED, STATUS_SKIPPED
        from aura.trainer.rollout.rollout_manager import SHUTDOWN_WAIT_MARGIN

        if signal_name != "normal":

            cls._release_controller_actors(signal_name)
        if not rollout_workers:
            return
        refs = []
        for worker in rollout_workers:
            try:
                refs.append(worker.release.remote())
            except Exception as exp:
                logger.warning(f"release rollout worker failed: {type(exp).__name__}")
        if not refs:
            return
        started = time.time()
        released = None
        try:
            released = ray.get(refs, timeout=SHUTDOWN_WAIT_MARGIN)
        except Exception as exp:
            logger.warning(
                f"[AURA_ROLLOUT] manager.shutdown.release.failed owner=driver signal={signal_name} "
                f"error={type(exp).__name__} elapsed={time.time() - started:.3f}s"
            )
        finally:
            for worker in rollout_workers:
                try:
                    ray.kill(worker)
                except Exception as exp:
                    logger.warning(
                        f"[AURA_ROLLOUT] manager.release.worker.kill.failed owner=driver "
                        f"signal={signal_name} error={type(exp).__name__}"
                    )
        if released is None:
            return
        if not any(released):
            logger.info(f"[AURA_ROLLOUT] manager.shutdown.release.skip owner=driver signal={signal_name} "
                        "reason=already_released_by_task")
            return
        elapsed = time.time() - started

        for value in released:
            status = STATUS_RELEASED if value else STATUS_SKIPPED
            logger.info(
                f"[AURA_ROLLOUT] manager.release.item kind=actor name=RolloutWorker "
                f"status={status} elapsed={elapsed:.3f}"
            )
        logger.warning(
            f"[AURA_ROLLOUT] manager.release.done owned={len(released)} "
            f"released={sum(1 for value in released if value)} failed=0 elapsed={elapsed:.3f} "
            f"owner=driver signal={signal_name}"
        )
        logger.warning(
            f"[AURA_ROLLOUT] manager.shutdown.release.done owner=driver signal={signal_name} "
            f"workers={len(refs)} elapsed={elapsed:.3f}s"
        )

    @classmethod
    def _release_controller_actors(cls, signal_name="normal"):
        """Release controller named actors on behalf of a cancelled rollout task. """
        try:
            from aura.controllers.rollout_controller.rollout_controller import RolloutController
        except ImportError as exp:
            logger.warning(
                f"[AURA_ROLLOUT] manager.release.controller.skip signal={signal_name} "
                f"error={type(exp).__name__}"
            )
            return
        for name in RolloutController.release_named_actors():
            logger.warning(
                f"[AURA_ROLLOUT] manager.release.item kind=named_actor name={name} "
                f"status=released owner=driver signal={signal_name}"
            )

    @classmethod
    def _wait_refs(cls, refs, timeout) -> bool:
        """Block until every ref finishes or the budget elapses. """
        import ray

        pending = list(refs)
        if not pending:
            return True
        _, pending = ray.wait(pending, num_returns=len(pending), timeout=timeout)
        return not pending

    @classmethod
    def _wait_rollout_drained(cls, rollout_refs, signal_name="normal") -> None:
        """Wait for the rollout tasks to finish their own shutdown on the driver side."""
        import ray

        from aura.trainer.rollout.rollout_manager import (
            DEFAULT_DRAIN_TIMEOUT,
            SHUTDOWN_WAIT_MARGIN,
        )

        rollout_refs = list(rollout_refs)
        if not rollout_refs or cls._wait_refs(rollout_refs, 0):
            return
        budget = (cls._session_drain_timeout or DEFAULT_DRAIN_TIMEOUT) + SHUTDOWN_WAIT_MARGIN
        logger.warning(
            f"[AURA_ROLLOUT] router.wait_rollout.begin signal={signal_name} "
            f"rollout_tasks={len(rollout_refs)} budget={budget}s"
        )
        drain_start = time.time()
        if cls._wait_refs(rollout_refs, budget):
            logger.warning(
                f"[AURA_ROLLOUT] router.wait_rollout.done signal={signal_name} "
                f"elapsed={time.time() - drain_start:.3f}s"
            )
            return
        logger.warning(
            f"[AURA_ROLLOUT] router.wait_rollout.timeout signal={signal_name} "
            f"elapsed={time.time() - drain_start:.3f}s budget={budget}s action=force_cancel"
        )
        for rollout_ref in rollout_refs:
            try:
                ray.cancel(rollout_ref, force=True)
            except Exception as exp:
                logger.warning(f"force cancel rollout failed: {type(exp).__name__}")

    @classmethod
    def _wait_train_drained(cls, signal_name="SIGTERM", rollout_refs=()) -> None:
        """Block until cancelled training and rollout tasks finish draining sessions."""
        from aura.trainer.rollout.rollout_manager import DEFAULT_SHUTDOWN_HARD_LIMIT

        cls._wait_rollout_drained(rollout_refs, signal_name)

        train_refs = list(cls._active_train_refs.values())
        if not train_refs:
            return
        hard_limit = cls._shutdown_hard_limit or DEFAULT_SHUTDOWN_HARD_LIMIT
        train_start = time.time()
        if cls._wait_refs(train_refs, hard_limit):
            logger.warning(
                f"[AURA_ROLLOUT] router.signal.cancel.train.done signal={signal_name} "
                f"tasks={len(train_refs)} elapsed={time.time() - train_start:.3f}s"
            )
        else:
            logger.warning(
                f"[AURA_ROLLOUT] router.signal.cancel.train.timeout signal={signal_name} "
                f"tasks={len(train_refs)} elapsed={time.time() - train_start:.3f}s hard_limit={hard_limit}s"
            )
