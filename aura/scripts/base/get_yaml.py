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

"""
Description:
Author: Aura Team
"""

import yaml
import sys


def get_val():
    file = sys.argv[1]
    param_path = sys.argv[2].split('.')
    with open(file) as f:
        data = yaml.safe_load(f)
        for key in param_path:
            data = data.get(key, {})
        if data == {}:
            print("")
        elif isinstance(data, bool):
            print(str(data).lower())
        else:
            print(data)


get_val()
