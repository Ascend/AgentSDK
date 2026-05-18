#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# -------------------------------------------------------------------------

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_PROMPTS = {
    "search_r1": "search_r1_system_prompt",
}


def _get_prompts(key: str, prompt_dir=CURRENT_DIR):
    with open(os.path.join(prompt_dir, f"{_PROMPTS[key]}.txt"), 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return "".join(lines)


SEARCH_R1_PROMPT = _get_prompts("search_r1")
