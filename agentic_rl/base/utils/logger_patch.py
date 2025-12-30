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

import logging
import sys

_patched = False


def patch():
    global _patched

    if _patched:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.ERROR,
    )

    _original_setLevel = logging.Logger.setLevel
    _original_addHandler = logging.Logger.addHandler

    def is_valid_package(name):
        return name.startswith('agentic_rl') or name == "__main__"

    def setLevel_patch(self, level):
        if is_valid_package(self.name):
            _original_setLevel(self, level)
        else:
            level = logging.ERROR
            _original_setLevel(self, level)

    def addHandler_patch(self, handler):
        if is_valid_package(self.name):
            _original_addHandler(self, handler)
        else:
            pass

    logging.Logger.setLevel = setLevel_patch
    logging.Logger.addHandler = addHandler_patch

    import warnings

    warnings.filterwarnings("ignore")

    def filterwarnings_hook(action, message='', category=Warning, module='', lineno=0, append=False):
        pass

    warnings.filterwarnings = filterwarnings_hook

    _patched = True
