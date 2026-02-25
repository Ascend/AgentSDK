# -*- coding: utf-8 -*-
# ruff: noqa: E402
from typing import Any, Dict

import ray

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.trainer.train_adapter.verl.configs.parse_verl_config import (
    VerlConfigParser,
)
from agentic_rl.trainer.train_adapter.verl.patch.verl_vllm_model_patch import (
    apply_vllm_model_patch,
)

apply_vllm_model_patch()


logger = Loggers(__name__)


class VerlAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    pass


def _create_tokenizer(config: Dict[str, Any]):
    """Create tokenizer from config.

    Args:
        config (Dict[str, Any]): Config dict.

    Returns:
        tokenizer: Tokenizer.
    """
    trust_remote_code = config.data.get("trust_remote_code", False)
    try:
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=trust_remote_code)
    except ValueError as e:
        logger.error(f"Failed to create tokenizer: {e}")
        raise e
    return tokenizer


def _define_worker_classes(config: Dict[str, Any]):
    """Define worker classes from config.

    Args:
        config (Dict[str, Any]): Config dict.

    Returns:
        role_worker_mapping (Dict[Role, RemoteFunction]): Role to worker class mapping.
        ray_worker_group_cls (RayWorkerGroup): Ray worker group class.
    """
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        logger.error(f"actor strategy {config.actor_rollout_ref.actor.strategy} is not supported")
        raise ValueError(f"actor strategy {config.actor_rollout_ref.actor.strategy} is not supported")
    if config.critic.strategy not in {"fsdp", "fsdp2"}:
        logger.error(f"critic strategy {config.critic.strategy} is not supported")
        raise ValueError(f"critic strategy {config.critic.strategy} is not supported")
    use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
    if use_legacy_worker_impl not in {"auto", "enable"}:
        logger.error(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
        raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
    from verl.workers.fsdp_workers import CriticWorker

    actor_rollout_cls = (
        VerlAsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
    )
    ray_worker_group_cls = RayWorkerGroup
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    return role_worker_mapping, ray_worker_group_cls


def _create_resource_pool_manager(config: Dict[str, Any]):
    """Create resource pool manager from config.

    Args:
        config (Dict[str, Any]): Config dict.

    Returns:
        resource_pool_manager (ResourcePoolManager): Resource pool manager.
    """
    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


@ray.remote
def train(config: Dict[str, Any]):
    """Execute the GRPO training workflow using verl.

    Args:
        config (Dict[str, Any]): Config dict.
    """
    try:
        agentic_rl_config, input_config, verl_config, gen_config = VerlConfigParser(config).process_config()
        logger.info("Config parsed successfully")
        tokenizer = _create_tokenizer(verl_config)
        role_worker_mapping, ray_worker_group_cls = _define_worker_classes(verl_config)
        resource_pool_manager = _create_resource_pool_manager(verl_config)

        # Load the reward manager
        from verl.trainer.ppo.reward import load_reward_manager
        reward_fn = load_reward_manager(
            verl_config, tokenizer, num_examine=0, **verl_config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            verl_config, tokenizer, num_examine=1, **verl_config.reward_model.get("reward_kwargs", {})
        )
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Configuration or initialization error: {e}")
        raise RuntimeError(f"Configuration or initialization error: {e}") from e
    except OSError as e:
        logger.error(f"OS error: {e}")
        raise RuntimeError(f"OS error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise RuntimeError(f"Unexpected error: {e}") from e
    trainer = None
    try:
        from agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer import AgentGRPOTrainer
        trainer = AgentGRPOTrainer(
            config=verl_config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            ray_worker_group_cls=ray_worker_group_cls,
            resource_pool_manager=resource_pool_manager,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            tokenizer_path=input_config.get("tokenizer_name_or_path"),
            dataset_additional_keys=input_config.get("dataset_additional_keys"),
            generate_config=gen_config,
            agentic_rl_config=agentic_rl_config,
        )
        trainer.init_workers()
        trainer.fit()
        logger.info("Trainer initialized successfully")
    except (AttributeError, ValueError, TypeError) as e:
        logger.error(f"Trainer initialization error: {e}")
        raise RuntimeError(f"Trainer initialization error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during trainer fit: {e}")
        raise RuntimeError(f"Unexpected error during trainer fit: {e}") from e
    finally:
        if trainer is not None:
            trainer.shutdown()
