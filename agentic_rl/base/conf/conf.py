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

from omegaconf import OmegaConf

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class AgenticRLConf:
    CONF_ENV: str = "AGENTIC_RL_CONF"

    # Define the whitelist: Only allow these first-level keys
    WHITELIST_KEYS = {
        "agentic_ai",
        "serve_conf",
        "direct_conf",
        "train_instances",
        "agent_instances",
        "infer_instances"
    }

    @classmethod
    def load_config(cls, conf_str=None):
        conf_str = os.environ.get(cls.CONF_ENV) if conf_str is None else conf_str
        if not conf_str:
            logger.warning(f"Environment variable {cls.CONF_ENV} is empty.")
            return OmegaConf.create({})

        # Load the original complete configuration
        full_conf = OmegaConf.create(conf_str)

        # Filter configuration: Only retain the first-level key in the whitelist and its sub-configurations
        filtered_dict = {
            k: v
            for k, v in full_conf.items()
            if k in cls.WHITELIST_KEYS
        }

        # Repackage as an OmegaConf object
        # while maintaining its DictConfig characteristics, such as dot notation access
        conf = OmegaConf.create(filtered_dict)
        logger.debug(f"AgenticRLConf: {conf}")

        return conf
