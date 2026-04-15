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

import functools

from fastapi import HTTPException


def async_raise_http_exception(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            import ray
            ray.serve_context = ray.serve.context._serve_request_context
            return await func(*args, **kwargs)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Key error: {e.args[0]}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {e.args[0]}")

    return wrapper
