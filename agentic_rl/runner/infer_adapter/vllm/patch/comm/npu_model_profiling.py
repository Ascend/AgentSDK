#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -------------------------------------------------------------------------

import os
import re

import torch
import torch_npu

PROF_SAVE_PATH = os.environ.get("PROFILING_SAVE_PATH", "./")


def run_model_with_profiling(model, input_ids, positions, intermediate_tensors, inputs_embeds, stat_step, process_name):
    process_name_new = re.sub(r'[^a-zA-Z0-9_]', '_', process_name)
    process_name_new = f'{process_name_new}_step_{stat_step}'
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        export_type=torch_npu.profiler.ExportType.Text)
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU
        ],
        with_stack=False,
        record_shapes=True,
        profile_memory=False,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            dir_name=PROF_SAVE_PATH + "_generate",
            worker_name=process_name_new)
    )
    prof.start()

    hidden_states = model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
    )
    prof.step()

    return hidden_states