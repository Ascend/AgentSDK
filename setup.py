#!/usr/bin/env python3
# coding=utf-8
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
import re
from pathlib import Path

from setuptools import setup, find_packages

this_directory = Path(__file__).parent


def get_ci_version_info():
    """
    Get version information from configfile

    Return: version number
    """
    with open("agentic_rl/__init__.py", "r") as f:
        content = f.read()
    version = re.search(r'__version__ = "(.*?)"', content).group(1)
    version = str(version).lower()
    return version


setup(
    name="agentic_rl",
    version=get_ci_version_info(),
    author="AgentSDK",
    license="Apache 2.0",
    description="AgenticRL",
    python_requires=">=3.10",
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "agentic_rl=agentic_rl.trainer.main:main",
        ]
    }
)
