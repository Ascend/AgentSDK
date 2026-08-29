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

"""Gemini API key authentication."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GeminiAuth:
    """Gemini API key authentication handler."""

    def load_api_key(self) -> str | None:
        """Load Gemini API key from environment or config."""
        # Env vars
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = os.environ.get(var)
            if key:
                return key

        # Config
        try:
            from src.config import load_config

            config = load_config()
            key = config.get("providers", {}).get("gemini", {}).get("api_key", "")
            if key:
                return key
        except Exception:  # nosec B110
            pass  # Intentional best-effort path; the surrounding fallback remains valid.

        return None

    def is_configured(self) -> bool:
        """Check if a Gemini API key is available."""
        return self.load_api_key() is not None
