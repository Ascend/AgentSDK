#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Media generation providers — image and video generation.

Architecture::

    media/
        __init__.py          ← exports
        base.py              ← MediaProvider / ImageProvider / VideoProvider ABCs
        registry.py          ← MediaProviderRegistry + media_registry singleton
        image/
            __init__.py
            agnes.py         ← AgnesImageProvider
        video/
            __init__.py
            agnes.py         ← AgnesVideoProvider

New providers are registered by importing the registry and calling
``media_registry.register_image(...)`` or
``media_registry.register_video(...)`` at module level (same pattern as
``register_provider(...)`` in ``clawcodex_ext/providers/__init__.py``).
"""

from clawcodex_ext.providers.media.base import (
    ImageProvider,
    ImageResult,
    MediaProvider,
    VideoProvider,
    VideoResult,
    VideoStatus,
    VideoTask,
)
from clawcodex_ext.providers.media.registry import MediaProviderRegistry, media_registry

__all__ = [
    # Base classes
    "ImageProvider",
    "ImageResult",
    "MediaProvider",
    "VideoProvider",
    "VideoResult",
    "VideoStatus",
    "VideoTask",
    # Registry
    "MediaProviderRegistry",
    "media_registry",
]
