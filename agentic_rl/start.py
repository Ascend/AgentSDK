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


import asyncio
import json
import os
import time
import uuid

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
import ray
from ray import serve
from ray.serve import HTTPOptions

from agentic_rl.base.conf.conf import AgenticRLConf
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.utils.utils import DEFAULT_SLEEP_TIME
from agentic_rl.runner.infer_manager import get_or_create_infer_manager, destroy_infer_manager

logger = Loggers(__name__).get_logger()


def start_direct_mode(conf: DictConfig):
    from agentic_rl.runner.agent_manager import get_or_create_agent_manager, destroy_agent_manager

    async def _init_manager():
        await get_or_create_agent_manager()
        # Training mode currently starts rollout inference processes via msrl and temporarily doesn't support starting via open-source vllm
        await get_or_create_infer_manager()

    def _exit_manager():
        destroy_agent_manager()
        # Training mode currently starts rollout inference processes via msrl and temporarily doesn't support starting via open-source vllm
        destroy_infer_manager()

    asyncio.run(_init_manager())

    async def submit_job(job_conf):
        job_dict = OmegaConf.to_container(job_conf, resolve=True)
        job_type = job_dict.get("job_type", "<unknown>")
        job_name = job_dict.get("job_name", "<unknown>")
        logger.info(f"[START] job_type={job_type}, job_name={job_name}, job={job_dict}")

        if job_type == "train":
            from agentic_rl.trainer.train_router import TrainRouter
            router = await TrainRouter.create()
            try:
                await router.train(name=job_name, **job_dict.get("job_kwargs", {}))
                logger.info(f"[FINISH] Successful job_type={job_type}, job_name={job_name}")
            except Exception as e:
                logger.error(f"[FINISH] Failed job_type={job_type}, job_name={job_name}, reason={e}")
        else:
            logger.warning(f"[FINISH] Unknown job_type={job_type}, job_name={job_name}, skipping.")

    async def run_all_jobs():
        entrypoints = list(conf.direct_conf.entrypoints)
        logger.info(f"Submitting {len(entrypoints)} jobs concurrently...")
        tasks = [asyncio.create_task(submit_job(job)) for job in entrypoints]
        await asyncio.gather(*tasks)
        logger.info("All jobs completed.")

    asyncio.run(run_all_jobs())

    # Stop all managers
    _exit_manager()


def start_serve_mode(conf: DictConfig):
    from agentic_rl.runner.agent_manager import get_or_create_agent_manager
    from agentic_rl.runner.infer_manager import get_or_create_infer_manager

    async def _init_manager():
        await get_or_create_agent_manager()
        # Serve mode can support starting inference via open-source vllm
        await get_or_create_infer_manager()

    asyncio.run(_init_manager())

    serve.start(
        http_options=HTTPOptions(
            **{
                "host": conf.serve_conf.host,
                "port": conf.serve_conf.port,
                "location": "EveryNode",
            }
        )
    )

    try:
        serve.get_app_handle("agentic_rl")
    except Exception as e:
        logger.info(f"No exists app: [agentic_rl], exception={e}.")
        from agentic_rl.serve.serve import deployment
        serve.run(
            target=deployment,
            blocking=False,
            name="agentic_rl",
        )
    else:
        logger.info(f"Find exists app: [agentic_rl].")

    while True:
        # Blocking wait
        time.sleep(DEFAULT_SLEEP_TIME)


@hydra.main(version_base=None, config_path="../configs", config_name="")
def main(conf: DictConfig):
    conf = AgenticRLConf.load_config(OmegaConf.to_yaml(conf, resolve=True))
    ray.init(
        namespace=str("agentic_raygroup"),
        runtime_env={"env_vars": {AgenticRLConf.CONF_ENV: OmegaConf.to_yaml(conf, resolve=True)} | os.environ}
    )
    os.environ[AgenticRLConf.CONF_ENV] = OmegaConf.to_yaml(conf, resolve=True)
    logger.info(f"Start the service in {conf.agentic_ai.mode} mode.")

    logger.info(f"{'=' * 20} Agentic AI Config Start {'=' * 20}\n"
                f"{json.dumps(OmegaConf.to_container(conf, resolve=True), indent=2, ensure_ascii=False)}"
                f"{'=' * 20} Agentic AI Config End {'=' * 20}")

    if conf.agentic_ai.mode == "serve":
        start_serve_mode(conf)
        return

    if conf.agentic_ai.mode == "direct":
        start_direct_mode(conf)
        return

    raise ValueError(f"{conf.name} not supported.")


if __name__ == "__main__":
    mlflow.set_tracking_uri("xxxx") # sensitive info
    main()

