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

from typing import Dict, Any
import copy
from pydantic import ValidationError
from agentic_rl.trainer.train_adapter.schema import GlobalConfig
from agentic_rl.base.utils.checker import validate_params
 
 
class ConfigParser:
    """Configuration parser using Pydantic for validation.
 
    This class is responsible for parsing and validating the configuration dictionary.
    """

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) for k in x.keys()),
            message="config must be a dictionary with string keys",
        )
    )
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ConfigParser with the provided configuration dictionary."""
        self.raw_config = copy.deepcopy(config)
        self.global_config: GlobalConfig | None = None
 
    def process_config(self) -> Dict[str, Any]:
        """Process and validate the configuration.
 
        This base implementation only validates and returns a normalized dictionary.
        Backend-specific parsers can override this method to generate additional objects/configs.
 
        Returns:
            dict[str, Any]: A dictionary containing the parsed and validated configuration.
        """
        global_config = self._validate_config()
        return global_config.model_dump()
 
    def _validate_config(self) -> GlobalConfig:
        """Validate the configuration dictionary using Pydantic model.
 
        Raises:
            ValueError: If validation fails.
        """
        try:
            self.global_config = GlobalConfig(**self.raw_config)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed:\n{e}") from e
        return self.global_config