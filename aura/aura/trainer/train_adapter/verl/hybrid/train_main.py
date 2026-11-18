# -*- coding: utf-8 -*-
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co.,Ltd.
import gc
import os
import signal
import socket
import time

import ray
try:
    from ray.exceptions import RayTaskError
except ImportError:
    RayTaskError = ()

from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

from aura.trainer.train_adapter.verl.hybrid.ray_trainer import HybridTrainer
from aura.trainer.rollout.rollout_manager import rollout_manager_enabled
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

DEFAULT_SESSION_DRAIN_TIMEOUT = 30.0


def _config_get(config, key, default=None):
    """Read a dotted key from an OmegaConf config, returning default if absent."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(config, key, default=default)
    except Exception:
        return default


def _resolve_drain_timeout(config):
    """Return the configured session drain timeout, falling back to the default."""
    value = _config_get(config, "extras.rollout_manager.session_drain_timeout", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_SESSION_DRAIN_TIMEOUT


_DATALOADER_TEARDOWN_TIMEOUT = 30.0


def _child_snapshot():
    """Describe this process's descendants without touching their exit status."""
    try:
        import psutil
    except ImportError:
        return {}
    try:
        children = psutil.Process(os.getpid()).children(recursive=True)
    except Exception:
        return {}
    snapshot = {}
    for proc in children:
        try:
            with proc.oneshot():
                snapshot[proc.pid] = (
                    proc.name(),
                    proc.status(),
                    proc.memory_info().rss / 1048576.0,
                    proc.ppid(),
                )
        except Exception:
            continue
    return snapshot


_SUPPRESS_WORKER_FAILURES = False
_SIGCHLD_PREVIOUS = None
_SIGCHLD_WATCHER = None


def _install_sigchld_watcher():
    global _SIGCHLD_PREVIOUS, _SIGCHLD_WATCHER
    current = signal.getsignal(signal.SIGCHLD)

    if _SIGCHLD_WATCHER is not None and current is _SIGCHLD_WATCHER:
        return
    previous = current
    if getattr(previous, "_aura_sigchld_watcher", False):
        return

    def _watcher(signum, frame):
        if not callable(previous):
            return None
        if _SUPPRESS_WORKER_FAILURES:

            return None
        return previous(signum, frame)

    _watcher._aura_sigchld_watcher = True
    try:
        signal.signal(signal.SIGCHLD, _watcher)
    except ValueError:

        return
    _SIGCHLD_PREVIOUS, _SIGCHLD_WATCHER = previous, _watcher


def _restore_sigchld_handler():

    global _SIGCHLD_PREVIOUS, _SIGCHLD_WATCHER
    if _SIGCHLD_WATCHER is None:
        return
    installed = signal.getsignal(signal.SIGCHLD) is _SIGCHLD_WATCHER
    previous, _SIGCHLD_PREVIOUS, _SIGCHLD_WATCHER = _SIGCHLD_PREVIOUS, None, None
    if not installed:
        return
    try:
        signal.signal(signal.SIGCHLD, previous)
    except ValueError:
        return


def _has_live_children():
    return any(info[1] != "zombie" for info in _child_snapshot().values())


def _teardown_dataloaders(trainer, timeout=_DATALOADER_TEARDOWN_TIMEOUT):

    names = [
        name
        for name in ("train_dataloader", "val_dataloader")
        if getattr(trainer, name, None) is not None
    ]
    if not names:
        return not _has_live_children()
    for name in names:
        try:
            setattr(trainer, name, None)
        except Exception:
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        gc.collect()
        if not _has_live_children():
            return True
        time.sleep(0.1)
    return False


def _drain_rollout_sessions(config, trainer, enabled):
    """Drain in-flight rollout sessions owned by the trainer process."""
    if not enabled:
        return
    rollout_manager = getattr(trainer, "async_rollout_manager", None)
    if rollout_manager is None or not hasattr(rollout_manager, "drain_sessions"):
        logger.warning("[AURA_ROLLOUT] manager.shutdown.drain.skip mode=hybrid reason=no_rollout_manager")
        return
    drain_timeout = _resolve_drain_timeout(config)
    drain_start = time.time()
    logger.warning(f"[AURA_ROLLOUT] manager.shutdown.drain.start mode=hybrid timeout={drain_timeout}s")
    try:
        rollout_manager.drain_sessions(timeout=drain_timeout)
    except Exception as exp:
        logger.warning(f"[AURA_ROLLOUT] manager.shutdown.drain.failed error={type(exp).__name__}")
    finally:
        logger.warning(
            f"[AURA_ROLLOUT] manager.shutdown.drain.done mode=hybrid elapsed={time.time() - drain_start:.3f}s"
        )
        try:
            if hasattr(rollout_manager, "log_shutdown_status"):
                rollout_manager.log_shutdown_status()
            if hasattr(rollout_manager, "stop_status_server"):
                rollout_manager.stop_status_server()
        except Exception as exp:
            logger.warning(f"[AURA_ROLLOUT] manager.shutdown.finalize.failed error={type(exp).__name__}")


@ray.remote
class HybridTaskRunner(TaskRunner):
    """Ray remote task runner for hybrid PPO training with agent loop support."""

    def run(self, config) -> None:
        """Run the hybrid PPO training pipeline.

        Args:
            config: OmegaConf configuration object for the training run.
        """
        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        logger.info(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        logger.info(f"Config: {OmegaConf.to_container(config, resolve=True)}")
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # Add reward model resource pool (v0.7.1: changed from add_reward_model_worker)
        self.add_reward_model_resource_pool(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = HybridTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        enabled = rollout_manager_enabled(config)
        global _SUPPRESS_WORKER_FAILURES
        _SUPPRESS_WORKER_FAILURES = False
        _restore_sigchld_handler()
        try:
            if enabled:
                _install_sigchld_watcher()
            trainer.init_workers()

            # Start the training process.
            trainer.fit()
            if enabled:
                _install_sigchld_watcher()
                _SUPPRESS_WORKER_FAILURES = True
                try:
                    cleared = _teardown_dataloaders(trainer)
                except Exception:
                    cleared = False
                if cleared:
                    _SUPPRESS_WORKER_FAILURES = False
                    _restore_sigchld_handler()
        except BaseException as exp:
            if enabled:
                logger.warning(f"[AURA_ROLLOUT] runner.fit.interrupted error={type(exp).__name__}")
            raise
        finally:
            try:
                _drain_rollout_sessions(config, trainer, enabled)
            finally:
                if enabled:
                    logger.info("[AURA_ROLLOUT] manager.release.delegated owner=verl status=deferred")


@ray.remote
def start_train(work_mode, train_config, rollout_config, agent_service, infer_service):
    logger.info(f"train_config={train_config}, work_mode={work_mode}")
    from time import time

    start_time = time()

    if not rollout_manager_enabled(rollout_config):
        from verl.trainer.main_ppo import run_ppo

        run_ppo(train_config, task_runner_class=HybridTaskRunner)
        logger.info(f"total time: {time() - start_time:.2f} seconds")
        return

    from aura.trainer.rollout.rollout_manager import RolloutManager, shutdown_hard_limit

    RolloutManager.create(
        "hybrid",
        train_config=train_config,
        rollout_config=rollout_config,
        agent_service=agent_service,
        infer_service=infer_service,
    ).run()

    runner = HybridTaskRunner.remote()
    runner_ref = runner.run.remote(train_config)

    enabled = rollout_manager_enabled(rollout_config)

    runner_done = False
    try:
        ray.get(runner_ref)
        runner_done = True
    except RayTaskError:
        runner_done = True
        raise
    finally:
        if enabled and not runner_done:
            try:
                ray.cancel(runner_ref, force=False)
                logger.warning(f"[AURA_ROLLOUT] runner.cancel.sent ref={runner_ref}")
            except Exception as exp:
                logger.warning(f"[AURA_ROLLOUT] runner.cancel.failed error={type(exp).__name__}")

            wait_timeout = shutdown_hard_limit(rollout_config)
            wait_start = time()
            try:
                ray.get(runner_ref, timeout=wait_timeout)
                logger.warning(f"[AURA_ROLLOUT] runner.cancel.wait.done elapsed={time() - wait_start:.3f}s")
            except Exception as exp:
                logger.warning(
                    f"[AURA_ROLLOUT] runner.cancel.wait.error elapsed={time() - wait_start:.3f}s "
                    f"error={type(exp).__name__}"
                )
    logger.info(f"total time: {time() - start_time:.2f} seconds")
