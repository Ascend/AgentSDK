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

import re

# Roles
SUMMARY = "summary"
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

ROLE_TOKEN_OVERHEAD = 4

# Tag constants
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_TAG_START = "<answer>"
ANSWER_TAG_END = "</answer>"

# Summary formatting
SUMMARY_HISTORY_PREFIX = "\n\n[Historical Information Summary]:\n\n"
SUMMARY_HISTORY_SUFFIX = "\n\n"

# Buffer for chat parsing and tokenization
CHUNK_BUFFER_LENGTH = 100
