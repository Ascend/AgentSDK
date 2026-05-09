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
import importlib
from typing import Any


def load_object_by_path(path: str) -> Any:
    """
    Load classes or functions dynamically based on the complete path string
    path: "package.module.ClassOrFuncName"
    Return the corresponding class or function object
    """
    try:
        # Separate the module path and the object name
        module_path, obj_name = path.rsplit(".", 1)

        # Dynamic import module
        module = importlib.import_module(module_path)

        # Obtain the object (class or function)
        obj = getattr(module, obj_name)
        return obj
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load object '{path}': {e}") from e
