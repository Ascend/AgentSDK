#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

import torch


def is_seq_like(x, B):
    if isinstance(x, (list, tuple)):
        return len(x) == B
    if torch.is_tensor(x):
        return x.dim() > 0 and x.size(0) == B
    return False


def _find_key(d, candidates):
    for k in candidates:
        if k in d: return k
    return None


def _len0(x):
    return x.size(0) if torch.is_tensor(x) else len(x)


def deserialize_and_split(file):
    # load buffer directly
    outputs = torch.load(file, map_location="cpu")
    return outputs
