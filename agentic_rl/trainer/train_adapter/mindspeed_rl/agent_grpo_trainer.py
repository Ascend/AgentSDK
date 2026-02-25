#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from typing import Iterator, Dict

from codetiming import Timer

from agentic_rl.trainer.train_adapter.mindspeed_rl import patch

patch.apply_patch()

import torch
import ray
from ray.exceptions import RayError
from mindspeed_rl import RayGRPOTrainer, RLConfig, MegatronConfig, GenerateConfig, RayActorGroup, RuleReward, Metric
from mindspeed_rl.trainer.utils import compute_grpo_data_metrics
from mindspeed_rl.utils.pad_process import (
    remove_padding_tensor_dict_to_dict,
    remove_padding_and_split_to_list,
    padding_dict_to_tensor_dict
)
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.utils.utils import metrics_post_processing, metrics_sort, compute_tps

from agentic_rl.base.utils.file_utils import FileCheck
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig, GenConfig, SamplingConfig
from agentic_rl.trainer.rollout.rollout_worker import RolloutWorker

logger = Loggers(__name__)


class AgentGRPOTrainer(RayGRPOTrainer):

    def __init__(self,
                 rl_config: RLConfig,
                 actor_config: MegatronConfig,
                 generate_config: GenerateConfig,
                 agentic_rl_config: AgenticRLConfig,
                 actor_worker: RayActorGroup,
                 reference_worker: RayActorGroup,
                 reward_list: list[RuleReward],
                 tokenizer: BaseTokenizer):
        """
        AgentGPPOTrainer class. This class implements the training method to update agent perform.

        Args:
            rl_config: RLConfig Configuration for reinforcement learning (e.g., PPO settings).
            actor_config: MegatronConfig for main model training/optimization.
            generate_config: GenerateConfig Configuration for generation/inference (e.g., vLLM settings).
            agentic_rl_config: AgenticRLConfig Configuration for create trajectory.
            actor_worker: RayActorGroup, the worker as actor model.
            reference_worker: RayActorGroup, the worker as ref model.
            reward_list: list[RuleReward], reword model or reward method.
            tokenizer: BaseTokenizer = None Object to retrieve the tokenizer.
        """
        self._check_args(rl_config, actor_config, generate_config, agentic_rl_config,
                         actor_worker, reference_worker, reward_list, tokenizer)
        self.actor_config = actor_config
        self.agentic_rl_config = agentic_rl_config
        
        try:
            self.rollout_worker = RolloutWorker.remote(
                n_parallel_agents=rl_config.n_samples_per_prompt,
                max_prompt_length=rl_config.max_prompt_length,
                actor_rollout_dispatch_size=rl_config.actor_rollout_dispatch_size,
                simplify_think_content=agentic_rl_config.simplify_think_content,
                tokenizer_name_or_path=actor_config.tokenizer_name_or_path,
                dataset_additional_keys=actor_config.dataset_additional_keys,
                generate_config=self._convert_generate_config(generate_config),
                agentic_rl_config=agentic_rl_config,
                worker_group=actor_worker,
                remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
                remove_padding_and_split_to_list=remove_padding_and_split_to_list)
            ray.get(self.rollout_worker.wait_init_finished.remote())
        except AttributeError as e:
            raise AttributeError("initialize rollout worker miss attribute") from e
        except RayError as e:
            raise RuntimeError("initialize rollout worker by ray failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when initialize rollout worker") from e

        try:
            ray.get([actor.init_sharding_manager.remote() for actor in reference_worker.actor_handlers])
        except RayError as e:
            raise RuntimeError("initialize sharding manager by ray failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when initialize sharding manager") from e

        try:
            super().__init__(actor_worker=actor_worker,
                             ref_worker=reference_worker,
                             reward_list=reward_list,
                             tokenizer=tokenizer,
                             global_batch_size=actor_config.global_batch_size,
                             micro_batch_size=rl_config.adv_dispatch_size,
                             train_iters=actor_config.train_iters,
                             save_interval=actor_config.save_interval,
                             dataset_additional_keys=actor_config.dataset_additional_keys,
                             **rl_config.dict())
        except AttributeError as e:
            raise AttributeError("trainer does not have the attribute") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer initialize") from e

    @staticmethod
    def _check_args(rl_config: RLConfig,
                    actor_config: MegatronConfig,
                    generate_config: GenerateConfig,
                    agentic_rl_config: AgenticRLConfig,
                    actor_worker: RayActorGroup,
                    reference_worker: RayActorGroup,
                    reward_list: list[RuleReward],
                    tokenizer: BaseTokenizer):
        if rl_config is None or not isinstance(rl_config, RLConfig):
            raise ValueError(f"rl_config must not be none or is not an instance of {RLConfig.__name__}")
        if actor_config is None or not isinstance(actor_config, MegatronConfig):
            raise ValueError(f"actor_config must not be none or is not an instance of {MegatronConfig.__name__}")
        if generate_config is None or not isinstance(generate_config, GenerateConfig):
            raise ValueError(f"generate_config must not be none or is not an instance of {GenerateConfig.__name__}")
        if agentic_rl_config is None or not isinstance(agentic_rl_config, AgenticRLConfig):
            raise ValueError(f"agentic_rl_config must not be none or is not an instance of {AgenticRLConfig.__name__}")
        if actor_worker is None or not isinstance(actor_worker, RayActorGroup):
            raise ValueError(f"actor_worker must not be none or is not an instance of {RayActorGroup.__name__}")
        if reference_worker is None or not isinstance(reference_worker, RayActorGroup):
            raise ValueError(f"reference_worker must not be none or is not an instance of {RayActorGroup.__name__}")
        if tokenizer is not None and not isinstance(tokenizer, BaseTokenizer):
            raise ValueError(f"tokenizer must not be none or is not an instance of {BaseTokenizer.__name__}")
        if not isinstance(reward_list, list):
            raise ValueError(f"reward_list must be a list")
        for reward in reward_list:
            if not isinstance(reward, ray.actor.ActorHandle):
                raise ValueError(f"reward in reward_list must be an instance of ray.actor.ActorHandle")

    @staticmethod
    def _convert_generate_config(config: GenerateConfig) -> GenConfig:
        return GenConfig(limit_mm_image_per_prompt=config.limit_mm_image_per_prompt,
                         limit_mm_video_per_prompt=config.limit_mm_video_per_prompt,
                         tokenizer_name_or_path=config.tokenizer_name_or_path,
                         trust_remote_code=config.trust_remote_code,
                         dtype=config.dtype,
                         infer_tensor_parallel_size=config.infer_tensor_parallel_size,
                         infer_pipeline_parallel_size=config.infer_pipeline_parallel_size,
                         infer_expert_parallel_size=config.infer_expert_parallel_size,
                         max_num_seqs=config.max_num_seqs,
                         max_num_batched_tokens=config.max_num_batched_tokens,
                         max_model_len=config.max_model_len,
                         gpu_memory_utilization=config.gpu_memory_utilization,
                         offload_train_optimizer=config.offload_train_optimizer,
                         offload_train_grad=config.offload_train_grad,
                         offload_train_param=config.offload_train_param,
                         enable_prefix_caching=config.enable_prefix_caching,
                         num_scheduler_steps=config.num_scheduler_steps,
                         enforce_eager=config.enforce_eager,
                         torchair_graph=config.torchair_graph,
                         enable_expert_parallel=config.enable_expert_parallel,
                         ascend_scheduler_config_enabled=config.ascend_scheduler_config_enabled,
                         sampling_config=SamplingConfig(
                             logprobs=getattr(config.sampling_config, "logprobs", 1),
                             max_tokens=getattr(config.sampling_config, "max_tokens", 128),
                             top_p=getattr(config.sampling_config, "top_p", 1.0),
                             top_k=getattr(config.sampling_config, "top_k", 50),
                             min_p=getattr(config.sampling_config, "min_p", 0.0),
                             temperature=getattr(config.sampling_config, "temperature", 0.2),
                             detokenize=getattr(config.sampling_config, "detokenize", False),
                             seed=getattr(config.sampling_config, "seed", None),
                         ))

    def transfer_dock_init(self):
        """
        Initialize transfer dock for data communication.
        """
        dataset_additional_keys = self.dataset_additional_keys if self.dataset_additional_keys is not None else []
        self.dataset_additional_keys = dataset_additional_keys + ["response_mask"]

        try:
            super().transfer_dock_init()
        except RayError as e:
            logger.error(f"init transfer docker by ray failed: {e}")
            raise RuntimeError("init transfer docker by ray failed") from e
        except Exception as e:
            logger.error(f"Unexpected error occurred when init transfer docker: {e}")
            raise RuntimeError("Unexpected error occurred when init transfer docker") from e

        self.dataset_additional_keys = dataset_additional_keys

        ray.get(self.rollout_worker.init_data_manager.remote(self.transfer_dock))

    def fit(self, train_data_iters: Iterator, test_data_iters: Iterator):
        """
        The utils loop of GRPO.

        Args:
            train_data_iters: Iterator to get data to create Trajectories during training phase.
            test_data_iters: Iterator to get data to create Trajectories during test phase.
        """
        metrics = Metric()

        try:
            iteration = self.actor_worker.get_iteration()
        except AttributeError as e:
            raise AttributeError("trainer does not have iteration attribute") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer get iteration") from e

        if self.agentic_rl_config.test_before_train and test_data_iters is not None:
            logger.info("Start testing processing before training...")
            test_metrics = self._validate_agent(test_data_iters)
            logger.info("Testing processing done")
            for key, value in test_metrics.items():
                logger.info(f"{key}: {value}")
            if self.agentic_rl_config.test_only:
                return


        if self.blocking:
            logger.info(
                'trainer sync start grpo training at iteration: {}/{} ...'.format(iteration + 1, self.train_iters))
        else:
            logger.info(
                'trainer async start grpo training at iteration: {}/{} ...'.format(iteration + 1, self.train_iters))

        while iteration < self.train_iters:
            with Timer(name='iteration', logger=None) as all_timer:
                try:
                    batch = next(train_data_iters)
                except StopIteration:
                    logger.warning(f"iteration {iteration + 1}/{self.train_iters}, but the data is already exhausted.")
                    return

                self._prepare_for_update(batch)

                self._update_actor()

                logger.info("compute_grpo_data_metrics start ...")
                grpo_data_metrics = compute_grpo_data_metrics(self.transfer_dock,
                                                              self.global_batch_size * self.n_samples_per_prompt,
                                                              self.tokenizer,
                                                              self.global_batch_size * self.n_samples_per_prompt,
                                                              self.guarantee_order)
                metrics_result = ray.get(self.transfer_dock.get_metrics.remote())

            self._update_metrics(metrics_result, grpo_data_metrics, metrics, all_timer)

            iteration += 1
            logger.info(metrics.metric, iteration, self.train_iters)

            self._clear_transfer_dock()

            self._add_scalar(iteration, metrics)

            if iteration % self.save_interval == 0 or iteration == self.train_iters:
                self._save_checkpoint(iteration)

        if test_data_iters is not None:
            logger.info("Start testing processing after training...")
            test_metrics = self._validate_agent(test_data_iters)
            logger.info("Testing processing done")
            for key, value in test_metrics.items():
                logger.info(f"{key}: {value}")

        logger.info('grpo training finished.')

    def _add_scalar(self, iteration, metrics):
        if self.tensorboard is not None:
            for k, v in metrics.metric.items():
                try:
                    self.tensorboard.add_scalar(f"train/{k}", v, iteration)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid data for tensorboard: "
                                     f"tag=train/{k}, value={v}, step={iteration}, error={e}") from e
                except Exception as e:
                    raise RuntimeError(f"Unexpected error occurred when add scalar to tensorboard: "
                                       f"tag=train/{k}, value={v}, step={iteration}, error={e}") from e

    def _clear_transfer_dock(self):
        try:
            ray.get(self.transfer_dock.clear.remote())
        except RayError as e:
            raise RuntimeError("clear transfer dock by ray failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when clear transfer dock") from e
        logger.debug(f"trainer clear transfer dock finished.")

    def _update_actor(self):
        try:
            logger.info("update start ...")
            self.actor_worker.update(self.kl_ctrl, self.skip_actor_log_prob)
            self.actor_worker.wait_all_ref_objs_run_over()
        except RayError as e:
            raise RuntimeError("trainer update actor by ray failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer update actor") from e
        logger.info(f"trainer update actor finished.")

    def _save_checkpoint(self, iteration):
        try:
            logger.debug(f"save checkpoint at iteration {iteration}...")
            self.save_checkpoint(iteration)
        except RayError as e:
            raise RuntimeError("save checkpoint by ray failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when save checkpoint") from e
        logger.debug(f"save checkpoint finished.")

        try:
            FileCheck.check_data_path_is_valid(self.actor_config.save)
        except ValueError as e:
            raise ValueError("Permission check error for the weight save path. "
                             "Please check if the umask parameter is set.") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when check permission for weight save path") from e

    def _update_metrics(self, metrics_result, grpo_data_metrics, metrics, all_timer):
        metrics_result = metrics_post_processing(metrics_result)
        metrics_result = metrics_sort(metrics_result, all_timer.last)
        log_max_throughput = False
        tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                          all_timer.last, log_max_throughput)
        update_tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                                 metrics_result['timing/update'], log_max_throughput)
        vllm_tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                               metrics_result['timing/rollout'], log_max_throughput)
        metrics.update(value=metrics_result)
        metrics.update(value=grpo_data_metrics)
        metrics.update("tokens/p/s", tps)
        metrics.update("update_tps", update_tps)
        metrics.update("vllm_throughput", vllm_tps)

    def _prepare_for_update(self, batch):
        self._put_prompts_experience(batch)

        self._generate_sequences()

        self._compute_advantage()

        self._compute_ref_log_prob()

        self._compute_log_prob()

        try:
            logger.info("wait_all_ref_objs_run_over start ...")
            self.actor_worker.wait_all_ref_objs_run_over()
            self.ref_worker.wait_all_ref_objs_run_over()
            for _, reward in enumerate(self.reward_list):
                if hasattr(reward, 'wait_all_ref_objs_run_over'):
                    reward.wait_all_ref_objs_run_over()
        except RayError as e:
            raise RuntimeError("trainer wait all process run over failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer wait all process run over") from e

    def _compute_log_prob(self):
        logger.info(f"compute_log_prob start self.skip_actor_log_prob={self.skip_actor_log_prob}...")
        if not self.skip_actor_log_prob:
            try:
                self.actor_worker.compute_log_prob(blocking=self.blocking)
            except RayError as e:
                raise RuntimeError("trainer compute actor log prob failed") from e
            except Exception as e:
                raise RuntimeError("Unexpected error occurred when trainer compute actor log prob") from e
            logger.debug(f"trainer compute actor log prob finished.")

    def _compute_ref_log_prob(self):
        try:
            logger.info("compute_ref_log_prob start ...")
            self.ref_worker.compute_ref_log_prob(blocking=self.blocking)
        except RayError as e:
            raise RuntimeError("trainer compute ref log prob failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer compute ref log prob") from e
        logger.debug(f"trainer compute ref log prob finished.")

    def _compute_advantage(self):
        try:
            logger.info("compute_advantage start ...")
            self.compute_advantage(blocking=False, guarantee_order=self.guarantee_order)
        except RayError as e:
            raise RuntimeError("trainer compute advantage failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer compute advantage") from e
        logger.debug(f"trainer compute advantage finished.")

    def _generate_sequences(self):
        try:
            logger.info('rollout_worker start ...')
            ray.get(self.rollout_worker.generate_sequences.remote())
        except RayError as e:
            raise RuntimeError("trainer generate sequences failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer generate sequences") from e
        logger.debug(f"trainer generate sequences finished.")

    def _put_data_experience(self, batch: Dict[str, list[torch.Tensor]], n_samples_per_prompt):
        new_data = dict()

        if len(batch) == 0:
            raise ValueError("Empty batch data is not valid.")

        length = {len(v) for v in batch.values()}

        if len(length) != 1:
            raise ValueError("Batch data should have same length.")

        length = list(length)[0]

        for key in batch.keys():
            new_data[key] = []
            for value in batch[key]:
                for _ in range(n_samples_per_prompt):
                    new_data[key].append(value)

        new_size = length * n_samples_per_prompt

        indexes = [i for i in range(new_size)]

        return padding_dict_to_tensor_dict(new_data), indexes
        
    def _put_prompts_experience(self, batch):
        new_batch_data = dict()

        for key in self.dataset_additional_keys:
            new_batch_data[key] = batch[key]

        for key in new_batch_data.keys():
            for i, value in enumerate(new_batch_data[key]):
                new_batch_data[key][i] = torch.tensor(self.tokenizer.tokenize(value))

        try:
            batch_dict, indexes = self._put_data_experience(new_batch_data, self.n_samples_per_prompt)
        except ValueError as e:
            raise ValueError("trainer failed to padding batch data") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer padding batch data") from e

        try:
            ray.get(self.transfer_dock.put_experience.remote(data_dict=batch_dict, indexes=indexes, is_prompt=True))
        except RayError as e:
            raise RuntimeError("trainer put prompts experience failed") from e
        except Exception as e:
            raise RuntimeError("Unexpected error occurred when trainer put prompts experience") from e

    def _generate_validation(self, batch):
        batch, index = self._put_data_experience(batch, 1)
        return ray.get(self.rollout_worker.generate_validation.remote(batch, index))
    
    def _validate_agent(self, data_iterator):
        rewards_list = []
        step = 1
        try:
            for batch_data in data_iterator:
                logger.info(f"validate iteration: {step}")
                reward = self._generate_validation(batch_data)
                rewards_list.append(reward)
                step += 1

            if not rewards_list:
                logger.warning("Validation data iterator is empty, skipping validation.")
                return {}

            reward = torch.cat(rewards_list, dim=0).mean().item()
            logger.info(f"validation mean reward: {reward}")
            return {"test score": reward}

        except Exception as e:
            logger.error(f"Validation process failed with error: {e}, skipping validation.")
            return {}