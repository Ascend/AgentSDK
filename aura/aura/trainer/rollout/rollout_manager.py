#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------

"""Unified rollout entry contract.

Converges the three work modes (hybrid / one_step_off / fully_async) into a
single ``create -> run -> shutdown -> status`` contract:

- one_step_off / fully_async: assemble worker + controller + executor here and
  block on ``executor.fit()`` in ``run()`` (rollout keeps its own loop).
- hybrid: do not rebuild verl assembly, only inject
  ``actor_rollout_ref.rollout.agent.agent_loop_manager_class``; ``run()``
  returns immediately because the verl trainer drives rollout.

The switch ``rollout_manager.enabled`` (default false) keeps the behavior
identical to before the refactor and emits no new logs when disabled.
"""

import socket
import time

import ray

from aura.base.log.loggers import Loggers
from aura.trainer.rollout.resource_ledger import (
    DEFAULT_RELEASE_TIMEOUT,
    ResourceEntry,
    RolloutResourceLedger,
)
from aura.trainer.rollout.session_manager import DEFAULT_STATUS_PORT, SessionManager
from aura.trainer.rollout.session_status_server import SessionStatusServer, is_train_master_process

logger = Loggers(__name__).get_logger()

MODE_HYBRID = "hybrid"
MODE_ONE_STEP_OFF = "one_step_off"
MODE_FULLY_ASYNC = "fully_async"
SUPPORTED_MODES = (MODE_HYBRID, MODE_ONE_STEP_OFF, MODE_FULLY_ASYNC)

AGENT_LOOP_MANAGER_CLASS_KEY = "actor_rollout_ref.rollout.agent.agent_loop_manager_class"
ROLLOUT_MANAGER_CONF_KEY = "extras.rollout_manager"
ROLLOUT_AGENT_LOOP_MANAGER_CLASS = (
    "aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.RolloutAgentLoopManager"
)

DEFAULT_DRAIN_TIMEOUT = 30.0
SHUTDOWN_WAIT_MARGIN = 10.0
DEFAULT_SHUTDOWN_HARD_LIMIT = 1800.0
DELEGATED_RESOURCE_NAMES = ("actor", "rollout_worker", "placement_group", "npu")


def shutdown_hard_limit(rollout_config, default=DEFAULT_SHUTDOWN_HARD_LIMIT):
    """Return the shutdown hang-guard (seconds) from the rollout config. """
    return _to_float(
        _conf_get(rollout_config, "rollout_manager.shutdown_hard_limit", default), default
    )


def _conf_get(conf, key, default=None):
    """Read a dotted key from an OmegaConf / dict config, else return default."""
    if conf is None:
        return default
    if isinstance(conf, dict):
        node = conf
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return default
    if OmegaConf.is_config(conf):
        return OmegaConf.select(conf, key, default=default)
    return default


def _to_bool(value, default=False):
    """Normalize a config switch value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _to_int(value, default):
    """Normalize a config value to int, falling back to default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default):
    """Normalize a config value to float, falling back to default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_port_bindable(port):
    """Whether ``port`` can be bound on every interface right now."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", int(port)))
        return True
    except (OSError, TypeError, ValueError):
        return False


def rollout_manager_enabled(config, default=False):
    """Return the ``rollout_manager.enabled`` switch (default false). """
    for key in ("rollout_manager.enabled", f"{ROLLOUT_MANAGER_CONF_KEY}.enabled"):
        value = _conf_get(config, key, None)
        if value is not None:
            return _to_bool(value, default)
    return default


def session_drain_timeout(rollout_config, default=DEFAULT_DRAIN_TIMEOUT):
    """Return the business drain upper bound (seconds) from the rollout config. """
    return _to_float(
        _conf_get(rollout_config, "rollout_manager.session_drain_timeout", default), default
    )


def _compact_sessions(sessions):
    """Format a session counter dict as ``{active:0,finished:0,aborted:0}``."""
    return "{" + ",".join(f"{key}:{value}" for key, value in sessions.items()) + "}"


def _create_rollout_worker(rollout_config, agent_service, infer_service, generate_config=None,
                           split_to_list=None, track_sessions=False):
    """Build a RolloutWorker with hard node affinity. """
    from aura.base.utils.pad_process import remove_padding_tensor_dict_to_dict
    from aura.trainer.rollout.rollout_worker import RolloutWorker
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    trajectory_timeout = None if track_sessions else rollout_config.trajectory_timeout

    return RolloutWorker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,
        )
    ).remote(
        generate_config=generate_config,
        train_backend=rollout_config.train_backend,
        trajectory_timeout=trajectory_timeout,
        weight_save_dir=rollout_config.weight_save_dir,
        hybrid_batch_num=rollout_config.hybrid_batch_num,
        use_on_policy=rollout_config.use_on_policy,
        wait_available_weight_timeout=rollout_config.wait_available_weight_timeout,
        n_parallel_agents=rollout_config.n_samples_per_prompt,
        actor_rollout_dispatch_size=rollout_config.actor_rollout_dispatch_size,
        validate_n_samples=rollout_config.validate_n_samples,
        traj_output_path=rollout_config.traj_output_path,
        tokenizer_name_or_path=rollout_config.tokenizer_name_or_path,
        dataset_additional_keys=rollout_config.dataset_additional_keys,
        global_batch_size=rollout_config.global_batch_size,
        remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list=split_to_list,
        service_mode="infer",
        agent_service=agent_service,
        infer_service=infer_service,
        llm_tokenizer_path=getattr(rollout_config, 'llm_tokenizer_path', None),
        track_sessions=track_sessions,
    )


def _create_rollout_controller(rollout_config, model_name):
    """Build a RolloutController from the rollout config. """
    from aura.controllers.rollout_controller.rollout_controller import RolloutController

    return RolloutController(
        enabled=rollout_manager_enabled(rollout_config),
        weight_save_dir=rollout_config.weight_save_dir,
        tokenizer_name_or_path=rollout_config.tokenizer_name_or_path,
        trust_remote_code=rollout_config.trust_remote_code,
        infer_tensor_parallel_size=rollout_config.infer_tensor_parallel_size,
        train_tensor_parallel_size=rollout_config.train_tensor_parallel_size,
        infer_expert_parallel_size=rollout_config.infer_expert_parallel_size,
        enable_version_control=rollout_config.enable_version_control,
        use_on_policy=rollout_config.use_on_policy,
        model_name=model_name,
    )


class RolloutManager:
    """Unified rollout entry for hybrid / one_step_off / fully_async."""

    def __init__(self, mode, enabled=False, status_port=DEFAULT_STATUS_PORT,
                 worker=None, controller=None, executor=None,
                 agent_loop_manager_cls=None, train_config=None, rollout_config=None,
                 session_drain_timeout=DEFAULT_DRAIN_TIMEOUT):

        self._mode = mode
        self._enabled = enabled
        self._status_port = status_port
        self._worker = worker
        self._controller = controller
        self._executor = executor
        self._agent_loop_manager_cls = agent_loop_manager_cls
        self._train_config = train_config
        self._rollout_config = rollout_config
        self._session_drain_timeout = session_drain_timeout
        self._running = False
        self._closed = False
        self._session_manager = SessionManager(enabled=enabled)
        self._status_server = None

        self._ledger = RolloutResourceLedger(log_enabled=enabled)

    @property
    def worker_handle(self):
        """RolloutWorker handle (non-hybrid only, None before ``create`` builds it)."""
        return self._worker

    @classmethod
    def create(cls, mode, **ctx):
        """Assemble a manager for the given mode. """
        if mode not in SUPPORTED_MODES:
            if _to_bool(_conf_get(ctx.get("rollout_config"), "rollout_manager.enabled", False)):
                logger.error(f"[AURA_ROLLOUT] manager.create.unsupported mode={mode}")
            raise ValueError(f"unsupported rollout mode: {mode}, supported: {SUPPORTED_MODES}")

        rollout_config = ctx.get("rollout_config")
        enabled = _to_bool(_conf_get(rollout_config, "rollout_manager.enabled", False))
        status_port = _to_int(
            _conf_get(rollout_config, "rollout_manager.status_port", DEFAULT_STATUS_PORT),
            DEFAULT_STATUS_PORT,
        )
        session_drain_timeout = _to_float(
            _conf_get(rollout_config, "rollout_manager.session_drain_timeout", DEFAULT_DRAIN_TIMEOUT),
            DEFAULT_DRAIN_TIMEOUT,
        )

        manager = cls(
            mode,
            enabled=enabled,
            status_port=status_port,
            train_config=ctx.get("train_config"),
            rollout_config=rollout_config,
            session_drain_timeout=session_drain_timeout,
        )
        if enabled:
            logger.info(f"[AURA_ROLLOUT] manager.create mode={mode} enabled={str(enabled).lower()}")
            # Fail fast instead of entering a half-started state.
            manager._validate_readiness(mode, ctx)

        try:
            if mode == MODE_HYBRID:
                manager._inject_agent_loop_manager()
                manager._register_delegated_resources()
            elif mode == MODE_ONE_STEP_OFF:
                manager._build_one_step_off(ctx["agent_service"], ctx["infer_service"])
            else:
                manager._build_fully_async(ctx["agent_service"], ctx["infer_service"])
        except BaseException:

            manager._release(reason="create_failed")
            raise
        # Self-check line so acceptance can grep status without an HTTP endpoint.
        manager.status()
        return manager

    def _validate_readiness(self, mode, ctx):
        """Verify that the resources this mode needs can actually be honored. """
        if mode == MODE_HYBRID:
            hybrid_engine = _conf_get(self._train_config, "actor_rollout_ref.hybrid_engine", None)
            if hybrid_engine is not None and not _to_bool(hybrid_engine, True):
                raise ValueError("hybrid rollout manager requires actor_rollout_ref.hybrid_engine=true")
            return
        infer_service = ctx.get("infer_service")
        if not infer_service:
            raise ValueError(f"{mode} rollout manager requires a non-empty infer_service")
        if not _is_port_bindable(self._status_port):
            raise ValueError(f"rollout status port is already in use: {self._status_port}")

    def _register_delegated_resources(self):
        """Record the hybrid resources whose lifecycle verl owns (evidence only)."""
        for name in DELEGATED_RESOURCE_NAMES:
            self._ledger.acquire(ResourceEntry(kind="delegated", name=name))

    def _register_worker(self):
        """Register the rollout worker actor right after it was created."""
        worker = self._worker

        def _release_worker():
            try:
                ray.get(worker.release.remote(), timeout=DEFAULT_RELEASE_TIMEOUT)
            finally:
                try:
                    ray.kill(worker)
                except Exception as exp:
                    if self._enabled:
                        logger.warning(
                            f"[AURA_ROLLOUT] manager.release.worker.kill.failed "
                            f"error={type(exp).__name__}"
                        )

        self._ledger.acquire(
            ResourceEntry(
                kind="actor",
                name="RolloutWorker",
                disposer=_release_worker,
            )
        )

    def _register_controller_resources(self):
        """Register the controller-owned resources (named actors + HTTP server)."""
        controller = self._controller
        self._ledger.acquire(
            ResourceEntry(
                kind="process",
                name="rollout_weight_manager",
                disposer=controller.close_weight_manager,
            )
        )
        self._ledger.acquire(
            ResourceEntry(
                kind="named_actor",
                name="rollout_queue",
                disposer=controller.close_queue,
            )
        )
        self._ledger.acquire(
            ResourceEntry(
                kind="http",
                name="rollout_server",
                disposer=controller.close_server,
            )
        )

    def run(self):
        """Start rollout according to the mode."""
        if self._enabled:
            logger.info(
                f"[AURA_ROLLOUT] manager.run mode={self._mode} driver={self._driver()} "
                f"blocking={str(self._mode != MODE_HYBRID).lower()}"
            )

            if self._mode != MODE_HYBRID:
                self._start_status_server()
        self._running = True
        if self._mode == MODE_HYBRID:
            return
        try:
            self._executor.fit()
        finally:
            self._running = False

    def shutdown(self, timeout=None):
        """Shutdown in a fixed three-segment order: drain -> release -> exit."""
        if self._closed:
            return
        self._closed = True
        drain_timeout = self._session_drain_timeout if timeout is None else float(timeout)
        start = time.time()
        if self._enabled:
            logger.info(
                f"[AURA_ROLLOUT] manager.shutdown.start mode={self._mode} timeout={float(drain_timeout)}"
            )

        drain_deferred = self._drain_sessions(drain_timeout)
        if self._enabled:
            status = self._session_status()
            active = "null" if status is None else status.get("active", 0)
            logger.info(
                f"[AURA_ROLLOUT] manager.shutdown.drain.done "
                f"active={active} elapsed={time.time() - start:.3f}"
            )
            self._log_shutdown_status()

        release = {"owned": 0, "released": 0, "failed": 0}
        if drain_deferred:
            if self._controller is not None:
                self._controller.close_server()
            if self._enabled:
                logger.warning(
                    f"[AURA_ROLLOUT] manager.shutdown.release.deferred mode={self._mode} owner=driver"
                )
        else:
            release = self._release()

        self._stop_status_server()
        if self._enabled and not drain_deferred:
            logger.info(
                f"[AURA_ROLLOUT] manager.shutdown.release.done owned={release['owned']} "
                f"released={release['released']} failed={release['failed']} "
                f"elapsed={time.time() - start:.3f}"
            )

        self._running = False

    def status(self, log=True):
        """Return the manager status, including session counters when hosted here."""
        session_status = self._session_status()
        sessions = None
        placeholder = None
        if session_status is not None:
            sessions = {key: session_status.get(key, 0) for key in ("active", "finished", "aborted")}
            placeholder = session_status.get("placeholder", 0)
        source = self._session_source()
        if self._enabled and log:
            logger.info(
                f"[AURA_ROLLOUT] manager.status mode={self._mode} "
                f"running={str(self._running).lower()} source={source} "
                f"sessions={_compact_sessions(sessions) if sessions is not None else 'null'}"
            )
        return {
            "mode": self._mode,
            "enabled": self._enabled,
            "running": self._running,
            "source": source,
            "status_port": self._status_port,
            "sessions": sessions,
            "placeholder": placeholder,
        }

    def _session_status(self):
        """Resolve session counters from the place that owns them. """
        if self._mode == MODE_HYBRID:
            return None
        if not self._enabled or self._worker is None:
            return self._session_manager.get_status()
        try:
            return ray.get(self._worker.get_session_status.remote(), timeout=5)
        except BaseException as exp:
            logger.warning(f"[AURA_ROLLOUT] manager.status.worker.failed error={type(exp).__name__}")
            return None

    def _session_source(self):
        """Where the session counters come from for the current mode."""
        if self._mode == MODE_HYBRID:
            return "trainer"
        if self._worker is not None:
            return "worker"
        return "local"

    def _start_status_server(self):
        """Serve ``GET /session_status`` on the training master node only."""
        if self._status_server is not None:
            return
        if not is_train_master_process():
            return
        mode = "hybrid" if self._mode == MODE_HYBRID else "executor"
        self._status_server = SessionStatusServer(
            self._status_port, lambda: self.status(log=False), mode=mode
        )
        self._status_server.start()
        self._ledger.acquire(
            ResourceEntry(
                kind="http",
                name="session_status",
                disposer=self._stop_status_server,
            )
        )

    def _stop_status_server(self):
        """Stop the session status endpoint and free its port (idempotent)."""
        if self._status_server is None:
            return
        self._status_server.stop()
        self._status_server = None

    def _driver(self):
        """Name of the rollout driver for the current mode."""
        return "trainer" if self._mode == MODE_HYBRID else "executor"

    def _log_shutdown_status(self):
        """Snapshot session counters before the status endpoint goes away."""
        status = self._session_status()
        if status is None:
            logger.info(
                f"[AURA_ROLLOUT] manager.shutdown.status active=null finished=null "
                f"aborted=null placeholder=null"
            )
            return
        logger.info(
            f"[AURA_ROLLOUT] manager.shutdown.status active={status.get('active', 0)} "
            f"finished={status.get('finished', 0)} aborted={status.get('aborted', 0)} "
            f"placeholder={status.get('placeholder', 0)}"
        )

    def _drain_sessions(self, timeout=DEFAULT_DRAIN_TIMEOUT):
        """Drain in-flight sessions hosted by the rollout worker."""
        if not self._enabled or self._worker is None:
            return False
        try:
            ray.get(self._worker.drain_sessions.remote(timeout), timeout=float(timeout) + 5.0)
            return False
        except (KeyboardInterrupt, ray.exceptions.TaskCancelledError):
            return True
        except BaseException as exp:
            logger.warning(
                f"[AURA_ROLLOUT] manager.shutdown.drain.failed error={type(exp).__name__}"
            )
            return False

    def _release(self, reason="shutdown"):
        """Release owned rollout resources in registration order."""
        if not self._enabled:
            return {"owned": 0, "released": 0, "failed": 0, "items": []}
        started = time.time()
        summary = self._ledger.release_all(reason)
        logger.info(
            f"[AURA_ROLLOUT] manager.release.done owned={summary['owned']} "
            f"released={summary['released']} failed={summary['failed']} "
            f"elapsed={time.time() - started:.3f}"
        )
        return summary

    def _inject_agent_loop_manager(self):
        """Point the hybrid agent loop manager to the rollout-aware subclass."""
        if not self._enabled:
            return
        if self._train_config is None:
            raise ValueError("hybrid rollout manager requires train_config for injection")
        from omegaconf import OmegaConf

        OmegaConf.update(
            self._train_config,
            AGENT_LOOP_MANAGER_CLASS_KEY,
            ROLLOUT_AGENT_LOOP_MANAGER_CLASS,
            force_add=True,
        )

        OmegaConf.update(self._train_config, f"{ROLLOUT_MANAGER_CONF_KEY}.enabled", True, force_add=True)
        OmegaConf.update(
            self._train_config,
            f"{ROLLOUT_MANAGER_CONF_KEY}.status_port",
            self._status_port,
            force_add=True,
        )

        OmegaConf.update(
            self._train_config,
            f"{ROLLOUT_MANAGER_CONF_KEY}.session_drain_timeout",
            self._session_drain_timeout,
            force_add=True,
        )
        self._agent_loop_manager_cls = ROLLOUT_AGENT_LOOP_MANAGER_CLASS
        if self._enabled:
            injected = OmegaConf.select(self._train_config, AGENT_LOOP_MANAGER_CLASS_KEY, default=None)
            logger.info(
                f"[AURA_ROLLOUT] manager.inject mode={self._mode} "
                f"key={AGENT_LOOP_MANAGER_CLASS_KEY} value={injected}"
            )

    def _build_one_step_off(self, agent_service, infer_service):
        """Assemble worker + controller + OneStepOffRolloutExecutor."""
        from aura.base.utils.pad_process import padding_dict_to_tensor_dict, put_prompts_experience
        from aura.trainer.rollout.rollout_executor import OneStepOffRolloutExecutor

        self._worker = _create_rollout_worker(
            self._rollout_config, agent_service, infer_service,
            track_sessions=self._enabled,
        )
        self._register_worker()
        ray.get(self._worker.wait_init_finished.remote(is_proxy_mode=True))

        self._controller = _create_rollout_controller(self._rollout_config, infer_service)
        self._register_controller_resources()
        self._controller.send_ready_to_train()

        self._executor = OneStepOffRolloutExecutor(
            self._controller,
            self._worker,
            train_iters=self._rollout_config.train_iters,
            padding_dict_to_tensor_dict=padding_dict_to_tensor_dict,
            put_prompts_experience=put_prompts_experience,
            dataset_additional_keys=self._rollout_config.dataset_additional_keys,
            data_optimized=self._rollout_config.data_optimized,
            n_samples_per_prompt=self._rollout_config.n_samples_per_prompt,
            hybrid_batch_num=self._rollout_config.hybrid_batch_num,
        )

    def _build_fully_async(self, agent_service, infer_service):
        """Assemble worker + SampleQueue injection + controller + fully-async executor."""
        from aura.base.utils.pad_process import padding_dict_to_tensor_dict, put_prompts_experience
        from aura.trainer.rollout.fully_async_rollout_executor import FullyAsyncRolloutExecutor

        self._worker = _create_rollout_worker(
            self._rollout_config, agent_service, infer_service,
            track_sessions=self._enabled,
        )
        self._register_worker()
        ray.get(self._worker.wait_init_finished.remote(is_proxy_mode=True))

        if isinstance(self._rollout_config, dict):
            max_required_samples = self._rollout_config.get("max_required_samples", 0)
        else:
            max_required_samples = getattr(self._rollout_config, "max_required_samples", 0)
        if max_required_samples:
            from aura.controllers.rollout_controller.sample_queue import get_sample_queue
            try:
                sample_queue = get_sample_queue()
                ray.get(self._worker.set_fully_async_config.remote(sample_queue, max_required_samples))
                logger.info(f"fully_async injected SampleQueue, max_required_samples={max_required_samples}")
            except Exception as e:
                logger.error(f"fully_async failed to inject SampleQueue: {e}", exc_info=True)
                raise

        self._controller = _create_rollout_controller(self._rollout_config, infer_service)
        self._register_controller_resources()
        self._controller.send_ready_to_train()

        self._executor = FullyAsyncRolloutExecutor(
            self._controller,
            self._worker,
            train_iters=self._rollout_config.train_iters,
            padding_dict_to_tensor_dict=padding_dict_to_tensor_dict,
            put_prompts_experience=put_prompts_experience,
            dataset_additional_keys=self._rollout_config.dataset_additional_keys,
            data_optimized=self._rollout_config.data_optimized,
            n_samples_per_prompt=self._rollout_config.n_samples_per_prompt,
            hybrid_batch_num=self._rollout_config.hybrid_batch_num,
        )
