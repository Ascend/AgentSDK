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

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class AgenticRLConf:
    CONF_ENV: str = "AURA_CONF"

    @classmethod
    def load_config(cls):
        conf_str = os.environ.get(cls.CONF_ENV)
        if not conf_str:
            logger.warning(f"Environment variable {cls.CONF_ENV} is empty.")
            return OmegaConf.create({})

        # Load the original complete configuration
        conf = OmegaConf.create(conf_str)

        logger.debug(f"AgenticRLConf: {conf}")

        return conf
