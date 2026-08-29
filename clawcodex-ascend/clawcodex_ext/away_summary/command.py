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

"""The /recap slash command."""

from __future__ import annotations

import logging
from typing import Any

from clawcodex_ext.away_summary.config import (  # pylint: disable=no-name-in-module
    load_away_summary_config,
)
from clawcodex_ext.away_summary.messages import (  # pylint: disable=no-name-in-module
    format_away_summary_for_display,
)
from clawcodex_ext.away_summary.service import AwaySummaryService  # pylint: disable=no-name-in-module
from src.command_system.types import LocalCommand, LocalCommandResult

logger = logging.getLogger(__name__)


def build_recap_command() -> LocalCommand:
    command = LocalCommand(
        name="recap",
        description="Generate a short recap of the current session",
        argument_hint="",
        aliases=["away", "catchup"],
        supports_non_interactive=False,
        run_in_thread=True,
        is_enabled=lambda: load_away_summary_config().recap_command_enabled,
    )
    command.set_call(_recap_call)
    return command


def _recap_call(args: str, context: Any) -> LocalCommandResult:
    del args
    provider = getattr(context, "provider", None)
    runtime = getattr(context, "runtime_context", None)
    if provider is None and runtime is not None:
        provider = getattr(runtime, "provider", None)
    if provider is None:
        return LocalCommandResult(type="text", value="Recap requires an active provider.")

    model = getattr(provider, "model", None)
    if runtime is not None:
        model = getattr(getattr(runtime, "options", None), "model", None) or model
    session = getattr(context, "session", None)
    if session is None and runtime is not None:
        session = getattr(runtime, "session", None)
    conversation = getattr(context, "conversation", None)
    if conversation is None:
        return LocalCommandResult(type="text", value="No conversation is available to recap.")

    cfg = load_away_summary_config(cwd=getattr(context, "cwd", None))

    # Try to share the parent query loop's prompt-cache prefix. The main
    # loop snapshots CacheSafeParams after each turn; the recap can
    # replay them so the recap request piggybacks on Anthropic's
    # prompt-cache instead of issuing an independent cold call.
    cache_safe_params = None
    if cfg.enable_recap_cache:
        cache_safe_params = _try_get_last_cache_safe_params()
        if cache_safe_params is not None:
            logger.debug("/recap: using cached CacheSafeParams (cache hit)")
        else:
            logger.debug("/recap: no CacheSafeParams cached, using fresh provider call")

    try:
        result = AwaySummaryService(
            conversation=conversation,
            provider=provider,
            model=model,
            session=session,
            config=cfg,
        ).generate(
            trigger="manual",
            cache_safe_params=cache_safe_params,
        )
    except Exception:
        logger.exception("Manual Away Summary generation failed")
        return LocalCommandResult(type="text", value="Recap failed. Check logs for details.")

    if not result.generated:
        return LocalCommandResult(type="text", value=result.reason)
    return LocalCommandResult(
        type="text",
        value=format_away_summary_for_display(result.summary),
    )


def _try_get_last_cache_safe_params() -> Any | None:
    """Read the most recent CacheSafeParams snapshot, if the fork
    primitive is importable. Returns ``None`` for any failure — the
    recap will then fall back to a fresh ``provider.chat`` call.
    """
    try:
        from clawcodex_ext.agent.forked_agent import get_last_cache_safe_params

        return get_last_cache_safe_params()
    except Exception:
        return None
