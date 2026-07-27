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

from pathlib import Path


def _repo_root() -> Path:
    return next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "configs" / "train").exists()
    )


def test_webwalker_train_configs_reference_registered_agent():
    train_dir = _repo_root() / "aura" / "configs" / "train"
    config_paths = [
        train_dir / "msrl_train_hybrid_A3_t8_qwen3_8b_webwalker.yaml",
        train_dir / "verl_train_hybrid_A3_t8_qwen3_8b_webwalker_fsdp.yaml",
    ]

    for path in config_paths:
        text = path.read_text(encoding="utf-8")
        assert "name: webwalker" in text
        assert "trajectory_generation_method" in text
        assert "page_cache_path" in text
