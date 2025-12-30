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
import importlib.util
import inspect
import pathlib
from typing import Type

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params
from agentic_rl.base.utils.file_utils import FileCheck

logger = Loggers(__name__)


@validate_params(
    base_class=dict(validator=lambda x: isinstance(x, type), message="base_class must be a class")
)
def load_subclasses_from_file(file_path: str, base_class: Type) -> Type:
    """
    Load subclass from given files, just load one subclass and raise error while not only one subclass is found.

    Args:
        file_path: path for py file for loading subclass.
        base_class: a baseclass, used for search subclass.

    Raises:
        ImportError or ValueError
    """
    FileCheck.check_data_path_is_valid(file_path)
    file_path = pathlib.Path(file_path).resolve()
    module_name = file_path.stem

    # Validate module_name is a valid Python identifier
    if not module_name:
        raise ValueError(f"Module name cannot be empty for file: {file_path}")
    
    if not module_name.isidentifier():
        raise ValueError(f"Invalid module name '{module_name}'. Module name must be a valid Python identifier.")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (AttributeError, RuntimeError) as e:
        raise ImportError(f"Failed to execute module {module_name}: {str(e)}") from e
    except Exception as e:
        raise ImportError(f"Unexpected error occurred during executing module {module_name}: {str(e)}") from e

    subclasses = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, base_class) and obj is not base_class:
            subclasses.append(obj)

    if len(subclasses) == 0:
        raise ImportError(f"{file_path} found no class inherited from {base_class.__name__}.")

    if len(subclasses) > 1:
        raise ImportError(f"{file_path} found multiply classes inherited from {base_class.__name__}, please "
                          f"ensure there is just one subclass.")

    return subclasses[0]
