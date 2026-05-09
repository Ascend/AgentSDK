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

# Gemini Vertex AI Config (for dataset preprocessing).
GCP_PROJECT_ID = "cloud-llm-test"
GCP_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-1.5-pro-002"
OAI_RM_MODEL = "gpt-4o-mini"

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

# SWEBench Harness Config
SWEBENCH_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
MAX_WORKERS = 4
FORCE_REBUILD = False
CACHE_LEVEL = "None"
CLEAN = False
OPEN_FILE_LIMIT = 4096
TIMEOUT = 1_800
NAMESPACE = None
REWRITE_REPORTS = False
SPLIT = "test"
INSTANCE_IMAGE_TAG = "latest"
REPORT_DIR = "../.."

ROLLOUT_WEIGHTS_PREFIX = "/rollout"

TRAIN_CLUSTER = "train"
ROLLOUT_CLUSTER = "rollout"


def set_cluster_mode(mode):
    os.environ["CLUSTER_MODE"] = mode


def get_cluster_mode():
    return os.environ["CLUSTER_MODE"]


def is_pd_separate():
    return bool(int(os.getenv("USE_PD", "0")))
