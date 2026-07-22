#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
webwalker_tools_v1 = {
    "visit_page": {
        "name": "visit_page",
        "description": "A tool analyzes the content of a webpage and extracts buttons associated with sublinks. Simply input the button which you want to explore, and the tool will return both the markdown-formatted content of the corresponding page of button and a list of new clickable buttons found on the new page.",
        "parameters": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "description": "the button you want to click (e.g., '<button>About Us</button>' or simply 'About Us')"
                }
            },
            "required": [
                "button"
            ]
        }
    },
    "finish": {
        "name": "finish",
        "description": "Call this tool once you want to stop exploring and provide a final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "the final answer or summary you want to provide"
                }
            },
            "required": [
                "response"
            ]
        }
    }
}

webwalker_explorer_tools = {
    "visit_page": webwalker_tools_v1["visit_page"],
    "finish": webwalker_tools_v1["finish"],
}

webwalker_tools = webwalker_explorer_tools

TOOLS = {
    "webwalker_tools_v1": webwalker_tools_v1,
    "webwalker_explorer_tools": webwalker_explorer_tools,
}
