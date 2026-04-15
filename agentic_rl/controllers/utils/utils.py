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

import time

import ray
import requests
import torch

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

MIN_RETRY_COUNT = 1
DEFAULT_RETRY_COUNT = 3
DEFAULT_BACKOFF_FACTOR = 30.0
MIN_BACKOFF_FACTOR = 5.0

MIN_SLEEP_TIME = 0.1
DEFAULT_SLEEP_TIME = 2
DEFAULT_TIMEOUT = 30
READ_TIMEOUT = 600
MAX_TIMEOUT = 1800
HEALTH_CHECK_TIMEOUT = 300

DEFAULT_URL_METHOD = "http"

DEFAULT_REPLICAS = 1
MAX_ONGOING_REQUESTS = 64

DEFAULT_CPUS = 1
MAX_CPUS = 4
MAX_CONCURRENCY = 128


def post_with_url(url: str, retry: int = MIN_RETRY_COUNT, backoff: float = DEFAULT_BACKOFF_FACTOR):
    for attempt in range(1, retry + 1):
        try:
            response = requests.post(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < retry:
                # log and back off
                logger.warning(f"Attempt {attempt}/{retry} failed sending to {url}: {e!r}. "
                               f"Waiting {backoff}s before retry ...")
                time.sleep(backoff)
            else:
                # all retries exhausted
                logger.error(f"All {retry} attempts failed for {url}.")
                raise e


def tensor_item(x):
    return x.item() if torch.is_tensor(x) else int(x)


def create_actor(
        *,
        name: str,
        cls,
        namespace: str = None,
        lifetime: str = "detached",
        options: dict | None = None,
        actor_args: tuple = (),
        actor_kwargs: dict | None = None,
):
    options = options or {}
    actor_kwargs = actor_kwargs or {}

    try:
        a = ray.get_actor(name, namespace=namespace)
        ray.kill(a)
        return cls.options(
            name=name, namespace=namespace, lifetime=lifetime, **options
        ).remote(*actor_args, **actor_kwargs)
    except ValueError:
        return cls.options(
            name=name, namespace=namespace, lifetime=lifetime, **options
        ).remote(*actor_args, **actor_kwargs)


def collator(features, dataset_additional_keys=None):
    if dataset_additional_keys is None:
        dataset_additional_keys = []
    features_dict = {"prompts": [torch.tensor(value['input_ids']) for value in features],
                     "responses": [torch.tensor(value['response_ids']) for value in features]}

    for add_key in dataset_additional_keys:
        features_dict[add_key] = [torch.tensor(value[add_key]) for value in features]

    return features_dict
