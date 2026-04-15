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

import ray
from transformers import AutoConfig

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.globals import ROLLOUT_WEIGHTS_PREFIX

MAX_RETAIN_WEIGHTS_VERSION = 2
PATH_ITER_PATTERN = r"iter_(\d+)"


@ray.remote
class RolloutWeightManager:
    def __init__(self,
                 weight_save_dir,
                 tokenizer_name_or_path,
                 trust_remote_code,
                 infer_tensor_parallel_size,
                 train_tensor_parallel_size,
                 infer_expert_parallel_size,
                 enable_version_control,
                 use_on_policy,
                 model_name):
        self.logger = Loggers(__name__).get_logger()

        self.inference_save_path = weight_save_dir + ROLLOUT_WEIGHTS_PREFIX

        os.makedirs(self.inference_save_path, exist_ok=True)

        self.weights_version = 0

        self.update_lock = Lock()

        self.model_name = model_name
        self.model_path = tokenizer_name_or_path
        self.hf_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code
        )
        ep_mode_env = os.getenv("ONE_STEP_OFF_EP_MODE", 'false')
        self.one_step_off_ep_mode = ep_mode_env == 'true'
        if self.one_step_off_ep_mode:
            self.infer_tp = infer_tensor_parallel_size
            self.head_dim_scale = 1
        else:
            self.infer_tp = int(infer_tensor_parallel_size * (
                    train_tensor_parallel_size / infer_tensor_parallel_size))
            self.head_dim_scale = infer_tensor_parallel_size // train_tensor_parallel_size
        self.infer_dp = infer_expert_parallel_size
        self.enable_version_control = enable_version_control
        self.max_possible_version = 0
        self.use_on_policy = use_on_policy

        self.logger.info(f"model name: {model_name}, model_path: {self.model_path}")
        self.logger.info(
            f"split_tp: {self.infer_tp}, infer_dp: {self.infer_dp}, head_dim_scale: {self.head_dim_scale}, "
            f"one step off: {self.one_step_off_ep_mode}")

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
                        self.logger.info(f"deleting expired weights: {dir_path}")
                        shutil.rmtree(dir_path, ignore_errors=True)

    def update_max_version(self, add_version_num):
        # Update the current version with the maximum predicted weight
        self.max_possible_version += add_version_num

    def _should_weights_update(self, weight_iter):
        input_weight_version = weight_iter + 1
        if input_weight_version <= self.weights_version:
            self.logger.warning(f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
                                f"input version: {input_weight_version}, input weight out-of-date")
            return False
        if self.use_on_policy:
            # on_policy
            self.logger.info(f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
                             f"input version: {input_weight_version}, on_policy always do convert weights")
            return True
        # Enable off_policy version control
        if self.enable_version_control:
            # Only perform weight conversion for the versions of weights that need to be updated,
            # and generate the rollout weight file.
            # one_step_off
            required_weight_version = self.max_possible_version - 1
            self.logger.info(f"|perf-stat|rollout|  update_weights current weight version: {self.weights_version}, "
                             f"input version: {input_weight_version}, one_step_off required version: {required_weight_version}")
            if not input_weight_version == required_weight_version:
                self.logger.info(f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
                                 f"input version: {input_weight_version}, required version: {required_weight_version}, "
                                 f"skip convert weights")
                return False
        self.logger.info(f"|perf-stat|rollout| update_weights current weight version: {self.weights_version}, "
                         f"input version: {input_weight_version}, do convert weights")
        return True

    def _do_weights_update(self, path, weight_iter):
        self.logger.info(f"|perf-stat|rollout| start do converted weights ...")
        try:
            writing_weights_version = (weight_iter + 1)
            import os
            import shutil

            # Define the source directory and the target directory
            src_dir = path
            dst_dir = os.path.join(self.inference_save_path, f"weights_{writing_weights_version}")
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir, ignore_errors=True)
            os.makedirs(dst_dir, exist_ok=True)

            self.logger.info(f"######## {src_dir=}, {dst_dir=}, {os.listdir(src_dir)=}")

            # Traverse all the files in the source directory
            for filename in os.listdir(src_dir):
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)

                # Make sure to only move the files (excluding subdirectories)
                if os.path.isfile(src_path):
                    shutil.move(src_path, dst_path)
                    self.logger.info(f"Moved: {filename}")

            # run_distributed_qwen3_assemble(**kwargs)
            self.weights_version = writing_weights_version
            self.logger.info(f"|perf-stat|rollout| converted weights succeed, weights version: {self.weights_version}")
        except Exception as e:
            self.logger.error(f"failed to synchronize model weights: {e}, "
                              f"current version: {self.weights_version}")
            traceback.print_exc()

    def sync_weights_update(self, path):
        self.clean_old_weights()
        search_res = re.search(PATH_ITER_PATTERN, path)
        weight_iter = int(search_res.group(1))

        self.logger.info(f"|perf-stat|rollout| start converted weights iter: {weight_iter}")
        with self.update_lock:
            if not self._should_weights_update(weight_iter):
                return
            self._do_weights_update(path, weight_iter)

    def init_done(self):
        pass
