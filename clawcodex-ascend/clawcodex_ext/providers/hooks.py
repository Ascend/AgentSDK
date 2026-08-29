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

"""Model discovery hooks dynamically registered at import time.

This module is imported from ``clawcodex_ext/__init__.py`` so the hooks
are registered before any ``ModelRegistry`` instance is created.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _codex_api_discovery() -> list[str]:
    """Fetch available models from the OpenAI Codex backend API.

    This is called lazily by ``ModelRegistry.available_models()`` each time
    the model list is requested.  If the OAuth token is not available or the
    API call fails, an empty list is returned (the static baseline from
    ``PROVIDER_INFO`` is still available).

    Register with::

        register_discovery_hook("openai-codex", _codex_api_discovery)
    """
    try:
        from src.auth.codex_oauth import resolve_codex_runtime_credentials

        credentials = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        from clawcodex_ext.providers.codex_models import get_codex_model_ids

        return get_codex_model_ids(credentials.api_key)
    except Exception:
        logger.debug("Codex API model discovery failed (non-fatal)", exc_info=True)
        return []
