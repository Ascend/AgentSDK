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

from typing import Any


def validate_message(message: Any) -> bool:
    """Validate that the message is a dictionary with string keys or a list of such dictionaries.

    The dictionary must contain "role" and "content" keys.

    Args:
        message (Any): The message object to validate.

    Returns:
        True if the message is a dict[str, Any] or list[dict[str, Any]] containing 'role' and 'content' keys,
        False otherwise.
    """
    if isinstance(message, dict):
        return all(isinstance(k, str) for k in message.keys()) and "role" in message and "content" in message
    if isinstance(message, list):
        return all(
            isinstance(x, dict) and all(isinstance(k, str) for k in x.keys()) and "role" in x and "content" in x
            for x in message
        )
    return False
