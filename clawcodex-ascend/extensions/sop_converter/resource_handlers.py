#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

# pylint: disable=undefined-loop-variable
# Backward-compatibility stub — re-exports from core/resource_handlers.py
from extensions.sop_converter.core.resource_handlers import (
    RESOURCE_TYPE_UNREGISTERED,
    ResourceHandler,
    ResourceHandlerError,
    ensure_builtin_handlers,
    get_resource_handler,
    normalize_resource_type,
    register_resource_handler,
    registered_resource_types,
    require_resource_handler,
)

__all__ = [
    "RESOURCE_TYPE_UNREGISTERED",
    "ResourceHandler",
    "ResourceHandlerError",
    "ensure_builtin_handlers",
    "get_resource_handler",
    "normalize_resource_type",
    "register_resource_handler",
    "registered_resource_types",
    "require_resource_handler",
]

from extensions.sop_converter.core import resource_handlers as _impl

for _name, _value in vars(_impl).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value, _impl
