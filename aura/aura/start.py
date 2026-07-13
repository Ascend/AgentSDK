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
import os
import time

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from ray import serve
from ray.serve import HTTPOptions

from aura.base.conf.conf import AgenticRLConf
from aura.base.log.loggers import Loggers
from aura.controllers.utils.utils import DEFAULT_SLEEP_TIME
from aura.runner.infer_manager import get_or_create_infer_manager, destroy_infer_manager

logger = Loggers(__name__).get_logger()


def start_direct_mode(conf: DictConfig):
    from aura.runner.agent_manager import get_or_create_agent_manager, destroy_agent_manager
    from aura.trainer.trainer_register.train_register import get_train_actor
    from aura.data_manager.data_registry import get_data_manager_actor
    from aura.controllers.utils.utils import kill_actor

    train_register_actor = get_train_actor()
    data_manager_actor = get_data_manager_actor()

    def get_work_mode():
        train_instances = list(conf.train_instances)
        for train_instance in train_instances:
            work_mode = train_instance.executor_kwargs.work_mode
            logger.info(f"=============work mode: {work_mode}===============")
            return work_mode
        return "one_step_off"

    async def _init_manager():
        await get_or_create_agent_manager()
        # Training mode currently starts rollout inference processes via msrl and temporarily doesn't support starting via open-source vllm
        work_mode = get_work_mode()
        if work_mode == "hybrid":
            return
        await get_or_create_infer_manager()

    def _exit_manager():
        destroy_agent_manager()
        # Training mode currently starts rollout inference processes via msrl and temporarily doesn't support starting via open-source vllm
        work_mode = get_work_mode()
        if work_mode == "hybrid":
            return
        destroy_infer_manager()

    asyncio.run(_init_manager())

    def register_train_backend(job_type):
        if job_type == "msrl_train":
            ray.get(train_register_actor.registry_msrl_train.remote())
            ray.get(data_manager_actor.registry_msrl_data_manager.remote())
            logger.info("registry msrl train and data manager")
        elif job_type == "verl_train":
            ray.get(train_register_actor.registry_verl_train.remote())
            ray.get(data_manager_actor.registry_verl_data_manager.remote())
            logger.info("registry verl train and data manager")
        else:
            logger.error(f"unknown job_type={job_type}, skipping.")
            return False
        return True

    async def submit_job(job_conf):
        job_dict = OmegaConf.to_container(job_conf, resolve=True)
        job_type = job_dict.get("job_type", "<unknown>")
        job_name = job_dict.get("job_name", "<unknown>")
        logger.info(f"[START] job_type={job_type}, job_name={job_name}, job={job_dict}")

        if not register_train_backend(job_type):
            return

        from aura.trainer.train_router import TrainRouter

        router = await TrainRouter.create()
        try:
            await router.train(name=job_name, **job_dict.get("job_kwargs", {}))
            logger.info(f"[FINISH] Successful job_type={job_type}, job_name={job_name}")
        except Exception as e:
            logger.error(f"[FINISH] Failed job_type={job_type}, job_name={job_name}, reason={e}")
            raise e

    async def run_all_jobs():
        entrypoints = list(conf.direct_conf.entrypoints)
        logger.info(f"Submitting {len(entrypoints)} jobs concurrently...")
        tasks = [asyncio.create_task(submit_job(job)) for job in entrypoints]
        await asyncio.gather(*tasks)
        logger.info("All jobs completed.")

    try:
        asyncio.run(run_all_jobs())
    finally:
        _exit_manager()
        kill_actor(train_register_actor)
        kill_actor(data_manager_actor)


def start_serve_mode(conf: DictConfig):
    from aura.runner.agent_manager import get_or_create_agent_manager
    from aura.runner.infer_manager import get_or_create_infer_manager

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
        serve.get_app_handle("aura")
    except Exception as e:
        logger.info(f"No exists app: [aura], exception={e}.")
        from aura.serve.serve import deployment

        serve.run(
            target=deployment,
            blocking=False,
            name="aura",
        )
    else:
        logger.info("Find exists app: [aura].")

    while True:
        # Blocking wait
        time.sleep(DEFAULT_SLEEP_TIME)


def start_direct_mode_with_serve(conf: DictConfig):
    from aura.runner.agent_manager import get_or_create_agent_manager, destroy_agent_manager
    from aura.trainer.trainer_register.train_register import get_train_actor
    from aura.data_manager.data_registry import get_data_manager_actor
    from aura.controllers.utils.utils import kill_actor

    train_register_actor = get_train_actor()
    data_manager_actor = get_data_manager_actor()

    def get_work_mode():
        train_instances = list(conf.train_instances)
        for train_instance in train_instances:
            work_mode = train_instance.executor_kwargs.work_mode
            logger.info(f"=============work mode: {work_mode}===============")
            return work_mode
        return "one_step_off"

    async def _init_manager():
        await get_or_create_agent_manager()
        # The training mode currently starts the rollout inference process via the msrl method,
        # and does not yet support starting via the open-source vllm method
        # The shared mode does not need to start the infer manager
        work_mode = get_work_mode()
        if work_mode == "hybrid":
            return
        await get_or_create_infer_manager()

    def _exit_manager():
        destroy_agent_manager()
        # In training mode, the rollout inference process is currently started through the msrl method,
        # and starting through the open-source vllm method is not yet supported
        # In full-card mode, there is no need to start the infer manager, and likewise, no need to destroy it
        work_mode = get_work_mode()
        if work_mode == "hybrid":
            return
        destroy_infer_manager()

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
        serve.get_app_handle("aura")
    except Exception as e:
        logger.info(f"No exists app: [aura], exception={e}.")
        from aura.serve.serve import deployment

        serve.run(
            target=deployment,
            blocking=False,
            name="aura",
        )
    else:
        logger.info("Find exists app: [aura].")

    def register_train_backend(job_type):
        if job_type == "msrl_train":
            ray.get(train_register_actor.registry_msrl_train.remote())
            ray.get(data_manager_actor.registry_msrl_data_manager.remote())
            logger.info("registry msrl train and data manager")
        elif job_type == "verl_train":
            ray.get(train_register_actor.registry_verl_train.remote())
            ray.get(data_manager_actor.registry_verl_data_manager.remote())
            logger.info("registry verl train and data manager")
        elif job_type == "omni_rl":
            ray.get(train_register_actor.registry_omnirl_train.remote())
            ray.get(data_manager_actor.registry_omnirl_data_manager.remote())
            logger.info("registry verl train and data manager")
        else:
            logger.error(f"unknown job_type={job_type}, skipping.")
            return False
        return True

    async def submit_job(job_conf):
        job_dict = OmegaConf.to_container(job_conf, resolve=True)
        job_type = job_dict.get("job_type", "<unknown>")
        job_name = job_dict.get("job_name", "<unknown>")
        logger.info(f"[START] job_type={job_type}, job_name={job_name}, job={job_dict}")

        if not register_train_backend(job_type):
            return

        from aura.trainer.train_router import TrainRouter

        router = await TrainRouter.create()
        try:
            await router.train(name=job_name, **job_dict.get("job_kwargs", {}))
            logger.info(f"[FINISH] Successful job_type={job_type}, job_name={job_name}")
        except Exception as e:
            logger.error(f"[FINISH] Failed job_type={job_type}, job_name={job_name}, reason={e}")
            raise e

    async def run_all_jobs():
        entrypoints = list(conf.direct_conf.entrypoints)
        logger.info(f"Submitting {len(entrypoints)} jobs concurrently...")
        tasks = [asyncio.create_task(submit_job(job)) for job in entrypoints]
        await asyncio.gather(*tasks)
        logger.info("All jobs completed.")

    try:
        asyncio.run(run_all_jobs())
    finally:
        # stop all manager
        _exit_manager()
        kill_actor(train_register_actor)
        kill_actor(data_manager_actor)


@hydra.main(version_base=None, config_path="../configs/train", config_name="")
def main(conf: DictConfig):
    runtime_env_vars = {
        AgenticRLConf.CONF_ENV: OmegaConf.to_yaml(conf, resolve=True),
    }
    # Propagate NPU/HCCL env so Ray workers inherit the same comm settings as the driver.
    for env_key in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "HCCL_BUFFSIZE",
        "HCCL_HOST_SOCKET_PORT_RANGE",
        "HCCL_NPU_SOCKET_PORT_RANGE",
        "HCCL_IF_BASE_PORT",
        "PYTORCH_NPU_ALLOC_CONF",
        "VLLM_ASCEND_ENABLE_NZ",
    ):
        env_val = os.environ.get(env_key)
        if env_val:
            runtime_env_vars[env_key] = env_val
    # Inject hybrid PYTHONPATH so Ray workers load sitecustomize (NPU device remap; verl + msrl).
    hybrid_pythonpath = os.environ.get("VERL_HYBRID_PYTHONPATH")
    if hybrid_pythonpath:
        base_pythonpath = os.environ.get("PYTHONPATH", "")
        runtime_env_vars["PYTHONPATH"] = (
            hybrid_pythonpath if not base_pythonpath
            else f"{hybrid_pythonpath}{os.pathsep}{base_pythonpath}"
        )

    ray.init(
        namespace=str("agentic_raygroup"),
        runtime_env={"env_vars": runtime_env_vars},
    )
    os.environ[AgenticRLConf.CONF_ENV] = OmegaConf.to_yaml(conf, resolve=True)

    logger.info(f"Start the service in {conf.agentic_ai.mode} mode.")

    if conf.agentic_ai.mode == "serve":
        start_serve_mode(conf)
    elif conf.agentic_ai.mode == "direct":
        start_direct_mode(conf)
    elif conf.agentic_ai.mode == "direct_with_serve":
        start_direct_mode_with_serve(conf)
    else:
        raise ValueError(f"{conf.name} not supported.")


if __name__ == "__main__":
    main()
