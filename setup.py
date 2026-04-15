#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import os
from setuptools import setup, find_packages


# Read dependencies from requirements.txt file
def parse_requirements(filename):
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            # Remove whitespace and comments from each line
            requirements = [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.startswith('#')
            ]
    return requirements

setup(
    name='agentic_rl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
