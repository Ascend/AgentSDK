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
import os
import re
import shutil
import traceback
from threading import Lock
from multiprocessing import Process, Queue

import ray
from transformers import AutoConfig

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from aura.base.log.loggers import Loggers
from aura.base.utils.globals import ROLLOUT_WEIGHTS_PREFIX
from aura.controllers.rollout_controller.rollout_weight_loader import run_distributed_qwen3_assemble

MAX_RETAIN_WEIGHTS_VERSION = 2
PATH_ITER_PATTERN = r"iter_(\d+)"
AGGREGATE_TIMEOUT = 1800
TERMINATE_TIMEOUT = 5


def is_empty(path):
    return len(os.listdir(path)) == 0


def aggregate_worker(sharded_path, output_file, q):
    try:
        storage_reader = dcp.FileSystemReader(sharded_path)
        metadata = storage_reader.read_metadata()
        full_container = {}
        for fqn, param_metadata in metadata.state_dict_metadata.items():
            full_container[fqn] = torch.empty(
                param_metadata.size, dtype=param_metadata.properties.dtype
            )
        dcp.load(state_dict=full_container, checkpoint_id=sharded_path)
        save_file(full_container, output_file)
        q.put(None)
    except Exception as e:
        q.put(str(e))


@ray.remote
class RolloutWeightManager:
    def __init__(
        self,
        weight_save_dir,
        tokenizer_name_or_path,
        trust_remote_code,
        infer_tensor_parallel_size,
        train_tensor_parallel_size,
        infer_expert_parallel_size,
        enable_version_control,
        use_on_policy,
        model_name,
    ):
        self.logger = Loggers(__name__).get_logger()

        self.inference_save_path = weight_save_dir + ROLLOUT_WEIGHTS_PREFIX

        os.makedirs(self.inference_save_path, exist_ok=True)

        self.weights_version = 0

        self.update_lock = Lock()

        self.model_name = model_name
        self.model_path = tokenizer_name_or_path
        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        ep_mode_env = os.getenv("ONE_STEP_OFF_EP_MODE", 'false')
        self.one_step_off_ep_mode = ep_mode_env == 'true'
        if self.one_step_off_ep_mode:
            self.infer_tp = infer_tensor_parallel_size
            self.head_dim_scale = 1
        else:
            self.infer_tp = int(infer_tensor_parallel_size * (train_tensor_parallel_size / infer_tensor_parallel_size))
            self.head_dim_scale = infer_tensor_parallel_size // train_tensor_parallel_size
        self.infer_dp = infer_expert_parallel_size
        self.enable_version_control = enable_version_control
        self.max_possible_version = 0
        self.use_on_policy = use_on_policy
        self.resume_iteration = int(os.getenv("RESUME_ITERATION", '-1'))  # 断点续训的iteration_id

        self.logger.info(f"model name: {model_name}, model_path: {self.model_path}")
        self.logger.info(
            f"split_tp: {self.infer_tp}, infer_dp: {self.infer_dp}, "
            f"head_dim_scale: {self.head_dim_scale}, "
            f"one step off: {self.one_step_off_ep_mode}"
        )

    def get_weights_version(self):
        return self.weights_version

    def clean_old_weights(self):
        if self.weights_version <= MAX_RETAIN_WEIGHTS_VERSION:
            return
        weights_path = self.inference_save_path
        pattern = re.compile(r"^weights_(\d+)$")
        for entry in os.listdir(weights_path):
            match = pattern.match(entry)
            if match:
                x = int(match.group(1))
                if x < self.weights_version - MAX_RETAIN_WEIGHTS_VERSION:
                    dir_path = os.path.join(weights_path, entry)
                    if os.path.isdir(dir_path):
                        self.logger.info(f"deleting expired rollout weights: {dir_path}")
                        shutil.rmtree(dir_path, ignore_errors=True)

    def update_max_version(self, add_version_num):
        # Update the current version with the maximum predicted weight
        self.max_possible_version += add_version_num

    def _should_weights_update(self, weight_iter):
        input_weight_version = weight_iter
        if input_weight_version <= self.weights_version:
            self.logger.warning(
                f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
                f"input version: {input_weight_version}, input weight out-of-date"
            )
            return False
        if self.resume_iteration > 0 and self.resume_iteration == input_weight_version:
            # msrl 非0断点续训时，首次更新需重置max_possible_version初始值
            self.max_possible_version = input_weight_version
            self.logger.info("|rollout|resume| msrl resume training skip weight version check")
            self.logger.info(
                f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
                f"input version: {input_weight_version}, msrl resume training do update weights first"
            )
            return True
        if self.use_on_policy:
            # on_policy
            self.logger.info(
                f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
                f"input version: {input_weight_version}, on_policy always do convert weights"
            )
            return True
        # Enable off_policy version control
        if self.enable_version_control:
            # Only perform weight conversion for the versions of weights that need to be updated,
            # and generate the rollout weight file.
            # one_step_off
            required_weight_version = self.max_possible_version - 1
            self.logger.info(
                f"|perf-stat|rollout|  update_weights current weight version: {self.weights_version}, "
                f"input version: {input_weight_version}, one_step_off required version: {required_weight_version}"
            )
            if not input_weight_version == required_weight_version:
                self.logger.info(
                    f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
                    f"input version: {input_weight_version}, required version: {required_weight_version}, "
                    f"skip update weights"
                )
                return False
        self.logger.info(
            f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
            f"input version: {input_weight_version}, do update weights"
        )
        return True

    def _do_weights_update_with_assemble(self, path, weight_iter):
        self.logger.info("|perf-stat|rollout| start do converted weights ...")
        try:
            writing_weights_version = weight_iter
            kwargs = dict(
                train_save_path=path,
                hf_config=self.hf_config,
                infer_tp=self.infer_tp,
                infer_dp=self.infer_dp,
                pattern="pp*_tp*_ep*.safetensors",
                target_dtype=torch.bfloat16,
                inference_save_path=self.inference_save_path,
                weights_version=writing_weights_version,
                head_dim_scale=self.head_dim_scale,
                use_simple_ep_mode=self.one_step_off_ep_mode,
            )

            run_distributed_qwen3_assemble(**kwargs)
            self.weights_version = writing_weights_version
            self.logger.info(f"|perf-stat|rollout| converted weights succeed, weights version: {self.weights_version}")
        except Exception as e:
            self.logger.error(f"failed to synchronize model weights: {e}, current version: {self.weights_version}")
            traceback.print_exc()

    def aggregate_hf_weights(self, sharded_path, output_file):
        q = Queue()
        p = Process(target=aggregate_worker, args=(sharded_path, output_file, q))
        self.logger.info(f"aggregate weights, sharded_path: {sharded_path}, output_file: {output_file}")
        p.start()
        p.join(timeout=AGGREGATE_TIMEOUT)
        self.logger.info(f"aggregate weights succeed, output file: {output_file}")

        error_msg = None

        if p.is_alive():
            p.terminate()
            p.join(timeout=TERMINATE_TIMEOUT)
            if p.is_alive():
                p.kill()
                p.join()
            error_msg = (
                f"aggregate_worker subprocess timed out after {AGGREGATE_TIMEOUT}s"
            )
        elif p.exitcode != 0:
            error_msg = (
                f"aggregate_worker subprocess exited with code {p.exitcode}"
            )
        else:
            try:
                err = q.get(timeout=5)
            except Exception:
                error_msg = "aggregate_worker subprocess failed (no result returned)"
            else:
                if err is not None:
                    error_msg = err

        if error_msg is not None:
            if os.path.exists(output_file):
                os.remove(output_file)
            raise RuntimeError(error_msg)

        self.logger.info("aggregate subprocess finished")

    def _move_weights(self, src_dir, weight_iter):
        self.logger.info("|perf-stat|rollout| start do converted weights ...")
        try:
            writing_weights_version = weight_iter
            dst_dir = os.path.join(self.inference_save_path, f"weights_{writing_weights_version}")
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir, ignore_errors=True)
            os.makedirs(dst_dir, exist_ok=True)

            self.logger.info(f"moving the weight file: {src_dir=}, {dst_dir=}, {os.listdir(src_dir)=}")

            # Traverse all the files in the source directory
            moved_files = []
            for filename in os.listdir(src_dir):
                if not filename.endswith(".safetensors"):
                    continue
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)

                # Make sure to only move the files (excluding subdirectories)
                if os.path.isfile(src_path):
                    shutil.move(src_path, dst_path)
                    moved_files.append(filename)
                self.logger.info(f"moved the weight file: {moved_files}")

            self.weights_version = writing_weights_version
            self.logger.info(f"|perf-stat|rollout| converted weights succeed, weights version: {self.weights_version}")
        except Exception as e:
            self.logger.error(f"failed to synchronize model weights: {e}, current version: {self.weights_version}")
            traceback.print_exc()

    def _do_weights_update_with_megatron(self, path, weight_iter):
        self.logger.info("|perf-stat|rollout| start do converted weights with megatron...")
        src_dir = path
        if not os.path.exists(src_dir) or is_empty(src_dir):
            self.logger.error(f"src dir {src_dir} does not exist or empty, skip update weights")
            return
        self._move_weights(src_dir, weight_iter)

    def _do_weights_update_with_fsdp(self, path, weight_iter):
        self.logger.info("|perf-stat|rollout| start do converted weights with fsdp...")
        src_dir = path
        if not os.path.exists(src_dir) or is_empty(src_dir):
            self.logger.error(f"src dir {src_dir} does not exist or empty, skip update weights")
            return
        output_file = os.path.join(src_dir, "model.safetensors")
        self.aggregate_hf_weights(src_dir, output_file)
        self._move_weights(src_dir, weight_iter)

    def _do_weights_update(self, path, weight_iter):
        if os.getenv("WEIGHT_SAVE_STRATEGY") == "megatron":
            self._do_weights_update_with_megatron(path, weight_iter)
        elif os.getenv("WEIGHT_SAVE_STRATEGY") == "fsdp":
            self._do_weights_update_with_fsdp(path, weight_iter)
        else:
            self._do_weights_update_with_assemble(path, weight_iter)

    def sync_weights_update(self, path):
        self.clean_old_weights()
        search_res = re.search(PATH_ITER_PATTERN, path)
        weight_iter = int(search_res.group(1))

        self.logger.info(f"|perf-stat|rollout| start update weights iter: {weight_iter}")
        with self.update_lock:
            if not self._should_weights_update(weight_iter):
                return
            self._do_weights_update(path, weight_iter)

    def init_done(self):
        pass
