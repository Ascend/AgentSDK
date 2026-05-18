# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
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
