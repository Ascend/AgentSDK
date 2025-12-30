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

from unittest.mock import patch, MagicMock


class TestLoggerPatch:

    def test_patch_logger(self):
        with patch("logging.Logger.setLevel"), \
                patch("logging.Logger.addHandler"), \
                patch("warnings.filterwarnings"):
            from agentic_rl.base.utils.logger_patch import patch as logger_patch

            logger_patch()

            import logging

            logger = MagicMock()

            logger.name = "agentic_rl"
            logging.Logger.setLevel(logger, "DEBUG")
            logging.Logger.addHandler(logger, MagicMock())

            logger.name = "abc"
            logging.Logger.setLevel(logger, "DEBUG")
            logging.Logger.addHandler(logger, MagicMock())
