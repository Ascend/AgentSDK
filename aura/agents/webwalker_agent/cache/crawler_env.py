# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
#
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
#
"""Lightweight WebWalker env for offline page crawling."""

from __future__ import annotations

from agents.webwalker_agent.environment.webwalker_env import WebWalkerEnvironment


class CrawlerEnv(WebWalkerEnvironment):
    """Minimal env that reuses fetch + link-extraction without heavy init.

    Bypasses tokenizer / critic-client setup; only the bits needed to fetch a
    page and extract in-domain buttons are wired up.
    """

    def __init__(self, root_url: str) -> None:  # noqa: D401 - intentionally light
        self.root_url = str(root_url or "")
        self.current_page_url = self.root_url
        self.button_url_dict: dict[str, str] = {}
        self._page_cache: dict = {}
        self.cache_mode = "off"
        self.page_cache_path = ""
        self._page_store = None
        self.tokenizer = None
