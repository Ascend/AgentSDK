#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

from clawcodex_ext.providers.base import BaseProvider, ChatResponse


class CatalogProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test", model="configured-model")
        self.client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[
                        SimpleNamespace(id="account-model-a"),
                        {"id": "account-model-b"},
                    ]
                )
            )
        )

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        raise NotImplementedError

    def chat_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    def get_available_models(self) -> list[str]:
        return ["configured-model"]


def test_base_provider_discovers_account_models_from_sdk_catalog() -> None:
    assert CatalogProvider().discover_available_models() == [
        "account-model-a",
        "account-model-b",
    ]
