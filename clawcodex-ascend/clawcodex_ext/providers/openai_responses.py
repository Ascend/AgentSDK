"""Compatibility bridge to the upstream OpenAI Responses helpers."""

from __future__ import annotations

import sys

from src.providers import openai_responses as _module

sys.modules[__name__] = _module
